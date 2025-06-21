import argparse
import jiwer
import math
import torch

import pandas as pd
import torch.nn.functional as F
import torch.nn as nn

from lhotse import CutSet
from lhotse.dataset import DynamicBucketingSampler
from lhotse.dataset.collation import TokenCollater
from lhotse.dataset.sampling.utils import find_pessimistic_batches
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchaudio.models import hubert_base
from transformers.trainer_utils import set_seed
from tqdm import tqdm

class mHuBERTFinetuneModel(torch.nn.Module):

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.pad_token_index = self.vocab.index("<pad>")
        self.transformer = hubert_base()
        self.lm_head = nn.Linear(768, len(vocab), bias=False)

    @classmethod
    def from_pretrained(self, vocab):
        model = mHuBERTFinetuneModel(vocab)

        from transformers import HubertModel
        from torchaudio.models.wav2vec2.utils import import_huggingface_model
        pretrained_model = import_huggingface_model(HubertModel.from_pretrained("utter-project/mHuBERT-147"))
        
        model.transformer.load_state_dict(pretrained_model.state_dict())

        return model

    @classmethod
    def from_finetuned(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        vocab = checkpoint["metadata"]["vocab"]

        model = mHuBERTFinetuneModel(vocab)
        model.load_state_dict(checkpoint["weights"])

        return model

    def save_checkpoint(self, output_path, metadata={}):
        # Add obligatory metadata
        checkpoint_metadata = metadata | { 
            "vocab" : self.vocab
        }

        torch.save({ "metadata" : checkpoint_metadata, "weights" : model.state_dict() }, output_path)

    def freeze_encoder(self):
        # Keep encoder frozen while final layer is warmed up
        self.transformer.encoder.requires_grad_(False)

    def unfreeze_encoder(self):
        # Function to unfreeze encoder once final layer warmed up
        self.transformer.encoder.requires_grad_(True)

    def forward(self, audio_padded, audio_lengths=None, labels_padded=None, labels_lengths=None, **kwargs):
        hidden_feats, hidden_lengths = self.transformer(audio_padded, audio_lengths)
        logits = self.lm_head(hidden_feats)
        logprobs = F.log_softmax(logits, dim=-1)

        if labels_padded is None:
            # Infer labels if none provided
            return self.decode(logprobs)

        else:
            # Compute loss if labels are provided
            loss = F.ctc_loss(
                logprobs.transpose(0, 1), # Model outputs (B, T, C) but ctc_loss expects (T, B, C)
                labels_padded,
                hidden_lengths,
                labels_lengths,
                blank=self.pad_token_index,
                reduction="sum", # Keep unnormalized loss, ctc_loss defaults to normalizing by target lengths
                zero_infinity=True
            )
            # Normalize by batch size, see https://github.com/NVIDIA/NeMo/issues/68#issuecomment-546026714
            loss /= audio_padded.size(0)
            return loss

    def decode(self, logprobs):
        indices = torch.argmax(logprobs, dim=-1)

        predictions = []

        for p in list(indices):
            unique_indices = torch.unique_consecutive(p, dim=-1)
            prediction = "".join([ self.vocab[i] for i in unique_indices if i != self.pad_token_index ])
            predictions.append(prediction)

        return predictions

class TriStageLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear learning rate scheduler with warmup, hold, and decay."""
    # Taken from Zhaoheng Ni's HuBERT training code:
    # https://github.com/pytorch/audio/blob/4e94321c54617dd738a05bfedfc28bc0fa635b5c/examples/hubert/lightning_modules.py#L53

    def __init__(
        self,
        optimizer,
        warmup_updates: int,
        hold_updates: int,
        decay_updates: int,
        init_lr_scale: float = 0.01,
        final_lr_scale: float = 0.05,
        last_epoch: int = -1
    ):
        self.warmup_updates = warmup_updates
        self.hold_updates = hold_updates
        self.decay_updates = decay_updates
        self.init_lr_scale = init_lr_scale
        self.final_lr_scale = final_lr_scale

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            return [
                base_lr * (self.init_lr_scale + self._step_count / self.warmup_updates * (1 - self.init_lr_scale))
                for base_lr in self.base_lrs
            ]
        elif self.warmup_updates < self._step_count <= (self.warmup_updates + self.hold_updates):
            return list(self.base_lrs)
        elif self._step_count <= (self.warmup_updates + self.hold_updates + self.decay_updates):
            return [
                base_lr
                * math.exp(
                    math.log(self.final_lr_scale)
                    * (self._step_count - self.warmup_updates - self.hold_updates)
                    / self.decay_updates
                )
                for base_lr in self.base_lrs
            ]
        else:
            return [base_lr * self.final_lr_scale for base_lr in self.base_lrs]

class ASRFinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, for_eval=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.for_eval = for_eval

    def __getitem__(self, cuts: CutSet) -> dict:
        cuts = cuts.sort_by_duration()

        audio_unpadded = []

        for cut in cuts:
            maybe_stereo_audio = cut.load_audio()
        
            if maybe_stereo_audio.shape[0] == 1:
                # Mono signal
                mono_audio = torch.Tensor(maybe_stereo_audio)[0]
            else:
                # Stereo signal
                kept_channel = cut.supervisions[0].custom['kept_channel']
                mono_audio = torch.Tensor(maybe_stereo_audio[kept_channel, :])

            with torch.no_grad():
                # Normalize audio if your base model is trained with fairseq
                # https://github.com/facebookresearch/fairseq/issues/3277
                mono_audio = torch.nn.functional.layer_norm(mono_audio, mono_audio.shape)
                audio_unpadded.append(mono_audio)

        audio_padded = pad_sequence(audio_unpadded, batch_first=True)
        audio_lengths = torch.LongTensor([ a.size(0) for a in audio_unpadded ])

        if not self.for_eval:

            labels_padded, labels_lengths = self.tokenizer(cuts)
            
            return {
                "_ids" : [ c.id for c in cuts ],
                "audio_padded": audio_padded,
                "audio_lengths": audio_lengths,
                # Return tokenized labels for computing CTC loss
                "labels_padded": labels_padded,
                "labels_lengths": labels_lengths
            }

        else:

            return {
                "_ids" : [ c.id for c in cuts ],
                # Return plain text for computing word/character error rate
                "_texts" : [ c.supervisions[0].text for c in cuts ],
                "audio_padded": audio_padded,
                "audio_lengths": audio_lengths
            }

def load_cuts_lazy(manifest_path):
    return CutSet.from_jsonl_lazy(manifest_path).filter(lambda cut: 1.0 <= cut.duration <= 10.0).resample(16_000)

def setup_data(train_manifest, eval_manifest, sampler_max_duration=75, sampler_quadratic_duration=40):
    
    train_cuts = load_cuts_lazy(train_manifest)
    eval_cuts  = load_cuts_lazy(eval_manifest)

    # Function that maps characters to integers (e.g.: 'a' -> 1, 'q' -> 5, etc.)
    tokenizer = TokenCollater(train_cuts + eval_cuts, add_bos=False, add_eos=False)
    # List of unique characters ['a', 'b', ..., ]
    vocab = list(tokenizer.idx2token)

    # Train dataset returns integer labels for computing CTC loss
    # Eval dataset returns texts for computing word/character error rates
    train_dset = ASRFinetuneDataset(tokenizer, for_eval=False)
    eval_dset  = ASRFinetuneDataset(tokenizer, for_eval=True)
    
    # Set to train_cuts to repeat to yield infinite batches (stop based on max_steps, not epoch)
    train_sampler = DynamicBucketingSampler(train_cuts.repeat(), shuffle=True, drop_last=True, max_duration=sampler_max_duration, quadratic_duration=sampler_quadratic_duration)
    eval_sampler = DynamicBucketingSampler(eval_cuts, shuffle=False, drop_last=False, max_duration=sampler_max_duration, quadratic_duration=sampler_quadratic_duration)

    train_loader = DataLoader(train_dset, sampler=train_sampler, batch_size=None, num_workers=4)    
    eval_loader = DataLoader(eval_dset, sampler=eval_sampler, batch_size=None, num_workers=4)

    return train_loader, eval_loader, vocab

def setup_model(vocab, max_steps, learning_rate):

    model = mHuBERTFinetuneModel.from_pretrained(vocab)
    # Freeze feature extractor for full run
    model.transformer.feature_extractor.requires_grad_(False)
    model.cuda()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler = TriStageLRScheduler(
        optimizer,
        warmup_updates=int(max_steps*0.1),
        hold_updates=int(max_steps*0.4),
        decay_updates=int(max_steps*0.5)
    )

    # Keep transformer frozen while final (vocab) layer is warmed up
    freeze_encoder_updates = int(0.5 * max_steps)

    return model, optimizer, scheduler, freeze_encoder_updates

def move_tensors_to_cuda(batch):
    return dict({ 
        k:v.cuda() if not k.startswith("_") and type(v) is torch.Tensor else v
        for (k,v) in batch.items()
    })

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('run_name')
    parser.add_argument('train_manifest')
    parser.add_argument('eval_manifest')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases to track experiment')
    parser.add_argument('--max-steps', type=int, default=30_000)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--sampler-max-duration', type=int, default=75, help="Max batch size in seconds.")
    parser.add_argument('--sampler-quad-duration', type=int, default=40, help="Penalize very long utterances (transformers have quadratic length cost).")
    args = parser.parse_args()

    train_loader, eval_loader, vocab = setup_data(
        args.train_manifest,
        args.eval_manifest,
        sampler_max_duration=args.sampler_max_duration,
        sampler_quadratic_duration=args.sampler_quad_duration
    )

    model, optimizer, scheduler, freeze_encoder_updates = setup_model(
        vocab,
        args.max_steps,
        args.learning_rate
    )

    if args.wandb:
        import wandb
        wandb.init(project="finetune-mhubert-147", name = args.run_name, save_code=True)

    # Note: because of train_cuts.repeat() setting above
    # next(batches) will yield batches indefinitely (as intended)
    batches = iter(train_loader)

    for global_step in (pbar := tqdm(range(args.max_steps), total=args.max_steps, dynamic_ncols=True)):

        batch = next(batches)

        if global_step == 0:
            # Print first batch for spot check
            print(batch)
            model.freeze_encoder()

        if global_step == freeze_encoder_updates:
            print(f"Unfreeze encoder at {freeze_encoder_updates} steps")
            model.unfreeze_encoder()

        batch = move_tensors_to_cuda(batch)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Forward pass
        loss = model(**batch)
        # Backward pass
        loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Update
        optimizer.step()

        lr_at_update=scheduler.get_last_lr()[0]
        scheduler.step()

        metrics = {
            "global_step" : global_step,
            "loss" : loss.item(),
            "grad_norm": norm.item(),
            "lr" : lr_at_update
        }

        pbar.set_postfix(metrics)

        # Run eval code at:
        #   - global_step 0 just to make sure it won't crash mid-run later!
        #   - but don't run eval until after final layer warmed up
        #   - do run eval at final step before finishing up
        if global_step % 5_000 == 0 and (global_step == 0 or global_step >= 10_000 or global_step == args.max_steps-1):

            print("Running eval ...")

            model.eval()

            results = []

            for batch in eval_loader:
                batch = move_tensors_to_cuda(batch)

                with torch.no_grad():
                    predictions = model(**batch)

                results.append(pd.DataFrame({
                    "id" : batch["_ids"],
                    "reference" : batch["_texts"],
                    "prediction" : predictions
                }))

            results = pd.concat(results)
            # Remove empty references before eval (jiwer will raise an error)
            results = results[ results.reference.str.strip().str.len() > 0 ].copy()

            wer_cer = {
                'wer' : round(jiwer.wer(results.reference.to_list(), results.prediction.to_list()) * 100, 2),
                'cer' : round(jiwer.cer(results.reference.to_list(), results.prediction.to_list()) * 100, 2)
            }

            print(results)
            print(wer_cer)

            metrics.update(wer_cer)

            model.train()

        if args.wandb and (global_step % 100 == 0 or 'cer' in metrics):
            wandb.log(metrics)

    # Save final checkpoint
    model.save_checkpoint(f"/workspace/tmp/{args.run_name}.pt", metrics)
