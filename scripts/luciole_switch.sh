#!/usr/bin/env bash
# Switch the Luciole vLLM serving profile on gpu-ubuntu.
#
# The L4 (24 GB) hosts ONE serving profile at a time; this helper enforces
# the exclusive profile and waits for readiness. The 1B pair backs the
# LiteLLM gateway (ports 8010/8011); the 8B/23B services bind the Tailscale
# interface (100.90.203.88:8013 / :8015) and are called directly, bypassing
# the model-scoped LiteLLM key.
#
# Usage (from athena):
#   bash scripts/luciole_switch.sh 1b|8b|23b|stop|status
#
# 8B/23B corpus generation needs the dedicated venv:
#   ~/.venv-vllm-serve  (see scripts/setup_gpu_env.sh history)
set -euo pipefail

PROFILE="${1:-status}"
REMOTE="${LUCIOLE_SWITCH_HOST:-gpu-ubuntu}"
SSH_KEY="${LUCIOLE_SWITCH_KEY:-$HOME/.ssh/gpu_ubuntu}"
TS_IP=100.90.203.88

remote() { ssh -o BatchMode=yes -i "$SSH_KEY" "$REMOTE" "$@"; }

wait_ready() { # $1 = host:port on the remote box
    remote "
      for i in \$(seq 1 120); do
        code=\$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://$1/v1/models 2>/dev/null || echo 000)
        [ \"\$code\" = '200' ] && exit 0
        sleep 5
      done; echo 'TIMEOUT waiting for $1' >&2; exit 1"
}

case "$PROFILE" in
    status)
        remote "systemctl is-active luciole-1b-base luciole-1b-instruct \
                luciole-8b-instruct luciole-23b-instruct 2>&1; \
                nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"
        exit 0
        ;;
    stop)
        remote "sudo systemctl stop luciole-1b-base luciole-1b-instruct \
                luciole-8b-instruct luciole-23b-instruct 2>/dev/null; echo stopped"
        exit 0
        ;;
    1b)
        remote "sudo systemctl stop luciole-8b-instruct luciole-23b-instruct 2>/dev/null; \
                sudo systemctl start luciole-1b-base && sudo systemctl start luciole-1b-instruct"
        wait_ready 127.0.0.1:8011
        echo "profile 1b ready (litellm gateway unchanged)"
        ;;
    8b)
        remote "sudo systemctl stop luciole-1b-base luciole-1b-instruct luciole-23b-instruct 2>/dev/null; \
                sudo systemctl start luciole-8b-instruct"
        wait_ready "$TS_IP:8013"
        echo "profile 8b ready on http://$TS_IP:8013/v1 (model: luciole-8b-instruct)"
        ;;
    23b)
        remote "sudo systemctl stop luciole-1b-base luciole-1b-instruct luciole-8b-instruct 2>/dev/null; \
                sudo systemctl start luciole-23b-instruct"
        wait_ready "$TS_IP:8015"
        echo "profile 23b ready on http://$TS_IP:8015/v1 (model: luciole-23b-instruct, nf4)"
        ;;
    *)
        echo "usage: $0 1b|8b|23b|stop|status" >&2
        exit 2
        ;;
esac
