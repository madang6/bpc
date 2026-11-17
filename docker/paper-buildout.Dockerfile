FROM nvidia/cuda:12.2.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libgl1 libglib2.0-0 libsm6 libx11-6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*
ADD bpc_environment.tar.gz /opt/bpc/env
RUN find /opt/bpc/env -type f ! -perm -004 -exec chmod a+r {} +

ENV PATH=/opt/bpc/env/bin:$PATH
RUN python /opt/bpc/env/bin/conda-unpack
RUN python -m pip install --no-cache-dir umap-learn==0.5.9.post2 wandb==0.25.1 && \
    python -m pip install --no-cache-dir --force-reinstall typing_extensions==4.15.0
