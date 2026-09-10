FROM python:3.12-slim

WORKDIR /app

# CUDA torch, version-matched to the verifier vLLM image's torch
# (vllm/vllm-openai:v0.25.1 pins torch==2.11.0; the stack runs v0.25.1 with
# VLLM_USE_V2_MODEL_RUNNER=0 so the V1 sampler difr replays stays active):
# the Gumbel replay draws the prover's per-request exponential noise from a
# CUDA Philox generator, so the runner needs (a) CUDA torch on a GPU — CPU
# MT19937 noise never matches — and (b) the same torch version, since the
# Philox offset stride of CUDA distribution kernels is not guaranteed stable
# across torch releases. Bump in lockstep with ORCH_VLLM_IMAGE.
RUN python3 -m pip install --no-cache-dir torch==2.11.0

COPY inf-ver-runner/requirements.txt /app/inf-ver-runner/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/inf-ver-runner/requirements.txt

# Only the torch-only verification core is used from difr (difr.gumbel_verify);
# it is put on PYTHONPATH rather than pip-installed so difr's research
# dependencies (vllm, datasets, ...) are never pulled in.
COPY difr/difr/gumbel_verify.py /app/difr/difr/gumbel_verify.py

COPY inf-ver-runner /app/inf-ver-runner
WORKDIR /app/inf-ver-runner

ENV PYTHONPATH=/app/inf-ver-runner:/app/difr
# Batch process: drains its model's pending events and exits (no HTTP server).
CMD ["python3", "-m", "app.main"]
