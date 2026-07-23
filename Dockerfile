# Deployment image. Used by Hugging Face Spaces (Docker SDK) and by any host
# that runs a container. Not needed to run the CLI locally.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/huggingface

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt requirements-web.txt ./
RUN pip install --upgrade pip && pip install -r requirements-web.txt

COPY --chown=user:user . .
RUN mkdir -p data/index && chown -R user:user /home/user
USER user

# Bake the encoders into the image so the first request is not a model download.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
