#!/usr/bin/env bash
# Create the cluster secrets the demo needs (registry pull creds, LiteLLM key, HF token).
# Run from the Windows host with CS1 access.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CS1_DIR="$(cd "$ROOT/../cs1_platform" && pwd)"
export KUBECONFIG="$(cygpath -m "$CS1_DIR/kubeconfig-cs1.conf")"
K="$(cygpath -m "$CS1_DIR/kubectl.exe")"
NS=styling-demo

# 1) registry pull credentials (admin creds of registry.ctc.sg.lab:5000)
REG_JSON=$( "$K" get secret registry-config-creds -n default -o json | python -c "
import json,sys,base64
d=json.load(sys.stdin)['data']['.dockerconfigjson']
a=json.loads(base64.b64decode(d))['auths']
e=a.get('registry.ctc.sg.lab:5000') or list(a.values())[0]
print(json.dumps(e))
" )
REG_USER=$( python -c "import json,sys,base64; e=json.loads('''$REG_JSON'''); print(e.get('username','') or base64.b64decode(e['auth']).decode().split(':',1)[0])" )
REG_PASS=$( python -c "import json,sys,base64; e=json.loads('''$REG_JSON'''); print(e.get('password','') or base64.b64decode(e['auth']).decode().split(':',1)[1])" )
"$K" -n "$NS" create secret docker-registry regcred \
  --docker-server=registry.ctc.sg.lab:5000 \
  --docker-username="$REG_USER" \
  --docker-password="$REG_PASS" --dry-run=client -o yaml | "$K" apply -f -
echo "regcred secret set (user=$REG_USER)"

# 2) LiteLLM master key (read live from the litellm deployment)
LLM_KEY=$( "$K" -n project-user-aieadmin exec deploy/litellm-helm -- sh -c 'printenv PROXY_MASTER_KEY' 2>/dev/null )
"$K" -n "$NS" create secret generic litellm-key --from-literal="key=$LLM_KEY" --dry-run=client -o yaml | "$K" apply -f -
echo "litellm-key secret set"

# 3) HuggingFace token (optional; set HF_TOKEN env var in your shell to use)
if [ -n "${HF_TOKEN:-}" ]; then
  "$K" -n "$NS" create secret generic hf-token --from-literal="token=$HF_TOKEN" --dry-run=client -o yaml | "$K" apply -f -
  echo "hf-token secret set"
else
  echo "hf-token: skipped (no HF_TOKEN in env; public weights will download unauthenticated)"
fi

# 4) demo API key for the backend
API_KEY=$(python -c "import secrets; print('sk-pcai-' + secrets.token_hex(16))")
"$K" -n "$NS" create secret generic styling-api-key --from-literal="key=$API_KEY" --dry-run=client -o yaml | "$K" apply -f -
echo "styling-api-key secret set (value: $API_KEY)"
