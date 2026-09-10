#!/usr/bin/env bash
# Creates the backing directories for the statically provisioned local
# PersistentVolumes defined in manifests/local-storage/persistent-volumes.yaml.
#
# Run on each worker node (requires root):
#   sudo ./create-local-storage-dirs.sh
#
# The node is detected from its hostname; pass a node name to override, e.g.
# when hostname doesn't match the Kubernetes node name:
#   sudo ./create-local-storage-dirs.sh gpu-node-2
set -euo pipefail

BASE_DIR=/mnt/local-storage
NODE="${1:-$(hostname)}"

case "$NODE" in
  gpu-node-1)
    DIRS=(monitoring-prometheus monitoring-grafana hf-model-cache)
    ;;
  gpu-node-2)
    DIRS=(harbor-registry harbor-database harbor-redis harbor-jobservice harbor-trivy)
    ;;
  gpu-node-3)
    DIRS=(infver-postgres infver-open-webui)
    ;;
  *)
    echo "ERROR: unknown node '$NODE'." >&2
    echo "Expected one of: gpu-node-1, gpu-node-2, gpu-node-3." >&2
    echo "If this node's hostname differs from its Kubernetes node name," >&2
    echo "pass the node name explicitly: $0 <node-name>" >&2
    exit 1
    ;;
esac

if [[ $EUID -ne 0 ]]; then
  echo "ERROR: must run as root (sudo $0)" >&2
  exit 1
fi

echo "Creating local-storage directories for node '$NODE':"
for dir in "${DIRS[@]}"; do
  path="$BASE_DIR/$dir"
  mkdir -p "$path"
  chmod 755 "$path"
  echo "  $path"
done
echo "Done."
