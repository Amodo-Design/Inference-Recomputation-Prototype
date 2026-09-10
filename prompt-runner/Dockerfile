FROM python:3.12-slim

WORKDIR /app

# promptbench's dependency tree pins sentencepiece 0.1.99, which ships no
# cp312 wheel and builds from source — it needs a C++ toolchain + cmake.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake pkg-config \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first so promptbench's transformers dependency resolves
# against it instead of pulling the multi-GB CUDA build from PyPI.
RUN python3 -m pip install --no-cache-dir torch==2.7.1 \
      --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

# PromptBench downloads its datasets (sst2/cola/gsm8k, all small) on first
# use; keep the cache inside the container FS (or mount HF_HOME for reuse).
ENV HF_HOME=/data/hf
EXPOSE 8200

CMD ["python3", "-m", "uvicorn", "app.service:app", "--host", "0.0.0.0", "--port", "8200"]
