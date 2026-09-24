# Built from the REPO ROOT (like inf-ver-runner), because it copies
# inf-proxy's pure extractors so both taps render prompts/tokens identically:
#   docker build -t <registry-host>/infver_images/frame-processor:latest -f frame-processor/Dockerfile .
FROM python:3.12-slim

WORKDIR /app

COPY frame-processor/requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

# Shared, pure module from inf-proxy (loaded by app/extract.py by file path).
COPY inf-proxy/app/capture.py /app/inf_proxy_capture.py

COPY frame-processor/app /app/app

# tools/ rides in this image but is NOT this service. Nothing the CMD above
# reaches imports either file; both run only when a command is overridden.
#
#   account.py            takes the account by hand — `--live` alongside the
#                         deployed tap on the capture host, or `--ledger` to
#                         file a stored capture as capture_window rows. Drives
#                         app/, so it needs the PYTHONPATH set below.
#   tapped_link_health.py the thing frame-processor observes, not an observer. It
#                         ships here so the bytes it puts on the wire and the
#                         digest the tap expects move together — one image,
#                         one payload, no transcription step between them.
#
# The health check must run as its own workload with its own lifecycle, never
# as a sidecar in this pod. A check that dies whenever the observer dies
# cannot tell "the capture went blind" from "the check stopped", which is the
# one distinction it exists to make. Override the command to run either:
#   command: ["python3", "/app/tools/tapped_link_health.py", "respond", ...]
#   command: ["python3", "/app/tools/account.py", "--live", "<monitor-iface-a>"]
COPY frame-processor/tools /app/tools

ENV PYTHONPATH=/app
# Passive observer: no ports exposed, no server. Live capture needs NET_RAW
# and the host network namespace (set in the Kubernetes manifest).
CMD ["python3", "-m", "app.main"]
