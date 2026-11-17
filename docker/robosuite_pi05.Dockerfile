FROM ghcr.io/astral-sh/uv:0.5.1 AS uv
FROM bpc-robosuite:latest

COPY --from=uv /uv /uvx /bin/
COPY openpi /opt/openpi
WORKDIR /opt/openpi
ENV UV_LINK_MODE=copy UV_NO_CACHE=1 UV_PROJECT_ENVIRONMENT=/opt/openpi/.venv
RUN uv venv --python "$(command -v python)" --system-site-packages \
    "$UV_PROJECT_ENVIRONMENT" && \
    uv pip install augmax==0.4.1 beartype==0.19.0 flax==0.10.2 \
    'jax[cuda12]==0.5.3' jaxtyping==0.2.36 orbax-checkpoint==0.11.13 \
    sentencepiece==0.2.0 transformers==4.53.2
ENV PATH=/opt/openpi/.venv/bin:$PATH \
    PYTHONPATH=/opt/openpi/src:/opt/robosuite:/opt/robomimic:/opt/mimicgen:/opt/robosuite-task-zoo
