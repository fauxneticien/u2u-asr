FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    tmux && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    git config --global --add safe.directory /workspace

RUN pip install \
    datasets==3.6.0 \
    jupyterlab==4.4.3 \
    jiwer==4.0.0 \
    librosa==0.11.0 \
    lhotse==1.30.3 \
    matplotlib==3.10.3 \
    pandas==2.3.0 \
    pympi-ling==1.70.2 \
    transformers==4.52.4 \
    wandb==0.20.1

WORKDIR /workspace
