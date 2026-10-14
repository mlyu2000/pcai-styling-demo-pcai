#!/usr/bin/env bash
# PCAI Styling Demo — build & deploy scripts (run from the Windows host, drive the bastion)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CS1_DIR="$(cd "$ROOT/../cs1_platform" && pwd)"

SSH="ssh -i $HOME/.ssh/id_ed25519_hermes -o IdentitiesOnly=yes -o BatchMode=yes ml@10.90.2.183"
REG=registry.ctc.sg.lab:5000
NS=styling-demo
TAG=${TAG:-v1}

# Registry CA (already installed on bastion under /etc/docker/certs.d, but keep idempotent)
ensure_bastion_ready() {
  $SSH 'test -f /etc/docker/certs.d/registry.ctc.sg.lab:5000/ca.crt' && return 0
  echo "WARNING: registry CA missing on bastion, re-installing"
  $SSH 'sudo mkdir -p /etc/docker/certs.d/registry.ctc.sg.lab:5000'
  scp -i "$HOME/.ssh/id_ed25519_hermes" -o IdentitiesOnly=yes - <<'X' 2>/dev/null || true
X
  echo "  (re-run scripts/install-registry-ca.sh if needed)"
}

build_frontend() {
  ensure_bastion_ready
  $SSH 'mkdir -p /home/ml/pcai/src/frontend'
  scp -i "$HOME/.ssh/id_ed25519_hermes" -o IdentitiesOnly=yes "$ROOT/frontend/index.html" "$ROOT/frontend/Dockerfile" "$ROOT/frontend/nginx.conf" ml@10.90.2.183:/home/ml/pcai/src/frontend/
  $SSH 'cd /home/ml/pcai/src/frontend && docker build -f Dockerfile -t '"$REG"'/pcai-styling-frontend:'"$TAG"' . && docker push '"$REG"'/pcai-styling-frontend:'"$TAG"' && docker rmi '"$REG"'/pcai-styling-frontend:'"$TAG"''
  echo "frontend image pushed"
}

build_backend() {
  ensure_bastion_ready
  $SSH 'mkdir -p /home/ml/pcai/src/backend/sample_data'
  scp -i "$HOME/.ssh/id_ed25519_hermes" -o IdentitiesOnly=yes "$ROOT/backend/main.py" "$ROOT/backend/Dockerfile" "$ROOT/backend/requirements.txt" ml@10.90.2.183:/home/ml/pcai/src/backend/
  scp -i "$HOME/.ssh/id_ed25519_hermes" -o IdentitiesOnly=yes "$ROOT/backend/sample_data/"* ml@10.90.2.183:/home/ml/pcai/src/backend/sample_data/
  $SSH 'cd /home/ml/pcai/src/backend && docker build --build-arg BUILD_TAG='"$TAG"' -f Dockerfile -t '"$REG"'/pcai-styling-backend:'"$TAG"' . && docker push '"$REG"'/pcai-styling-backend:'"$TAG"' && docker rmi '"$REG"'/pcai-styling-backend:'"$TAG"''
  echo "backend image pushed"
}

deploy() {
  export KUBECONFIG="$(cygpath -m "$CS1_DIR/kubeconfig-cs1.conf")"
  K="$(cygpath -m "$CS1_DIR/kubectl.exe")"
  "$K" apply -f "$ROOT/manifests"
  echo "deployed (TAG=$TAG) to ns/$NS"
  "$K" -n "$NS" wait --for=condition=ready pod -l app=styling-frontend --timeout=180s
  "$K" -n "$NS" wait --for=condition=ready pod -l app=styling-backend  --timeout=180s
}

verify() {
  export KUBECONFIG="$(cygpath -m "$CS1_DIR/kubeconfig-cs1.conf")"
  K="$(cygpath -m "$CS1_DIR/kubectl.exe")"
  echo "=== pods ==="; "$K" -n "$NS" get pods -o wide
  echo "=== endpoints ==="; "$K" -n "$NS" get endpoints
  echo "=== virtualservice ==="; "$K" -n "$NS" get virtualservice
  "$K" -n "$NS" exec deploy/styling-backend -- curl -s http://localhost:8000/healthz | head -c 400; echo
  echo "=== UI (in-cluster) ==="
  "$K" -n "$NS" exec deploy/styling-backend -- curl -s http://styling-frontend.styling-demo.svc.cluster.local:80/ | head -c 200; echo
  echo "=== public ingress ==="
  curl -s -o /dev/null -w "https://styling-demo.aie.cs1.ctc.sg.lab -> %{http_code}\n" https://styling-demo.aie.cs1.ctc.sg.lab
}

gpu_metrics() {
  export KUBECONFIG="$(cygpath -m "$CS1_DIR/kubeconfig-cs1.conf")"
  K="$(cygpath -m "$CS1_DIR/kubectl.exe")"
  $SSH 'curl -sk --cert /etc/kubernetes/pki/tls/apiserver-kubelet-client.crt --key /etc/kubernetes/pki/tls/apiserver-kubelet-client.key https://10.90.2.41:6443 --cacert /etc/kubernetes/pki/ca.crt' >/dev/null 2>&1 || true
  # use DCGM exporter via a probe pod (kubeconfig from host)
  "$K" -n "$NS" get pod -o name | head -1 >/dev/null
  echo "(DCGM metrics via probe pod — see scripts/gpu-report.sh)"
}

case "${1:-}" in
  frontend) build_frontend ;;
  backend)  build_backend ;;
  images)   build_backend; build_frontend ;;
  deploy)   deploy ;;
  verify)   verify ;;
  *) echo "usage: $0 {backend|frontend|images|deploy|verify}" ;;
esac
