#!/usr/bin/env bash
# Provision the binoculars-eu scoring environment on gpu-ubuntu (OVH L4).
#
# Idempotent — safe to re-run. Run from the repo root on athena:
#
#     bash scripts/setup_gpu_env.sh
#
# What it does, remotely:
#   1. creates ~/.venv-binoculars-eu and installs torch 2.5.1 (cu124 wheels,
#      per PRD §8.2 and protocol §8.1 pins),
#   2. rsyncs this repository to ~/projects/binoculars-eu,
#   3. installs requirements-eval.txt (protocol §8.1) + the [api] stack,
#      then the package itself with --no-deps so the pins are untouched,
#   4. copies the HuggingFace token (private Luciole-1B-SFT-1.0 repo) and
#      pre-downloads the Base + SFT weights used by the `fr` profile,
#   5. runs a GPU smoke scoring test (Binoculars.for_language("fr")).
#
# Note: vLLM/LiteLLM are NOT touched here — the calibration scorer needs
# logits (transformers), not the OpenAI-compatible text API.

set -euo pipefail

REMOTE=gpu-ubuntu

step() { echo; echo "== $* =="; }

step "1/5 venv + torch cu124 pin"
ssh "$REMOTE" 'bash -s' <<'EOS'
set -euo pipefail
if [ ! -d ~/.venv-binoculars-eu ]; then
  python3.12 -m venv ~/.venv-binoculars-eu
fi
source ~/.venv-binoculars-eu/bin/activate
pip install -q --upgrade pip setuptools wheel
pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
EOS

step "2/5 rsync repo"
ssh "$REMOTE" 'mkdir -p ~/projects'
rsync -az --delete \
  --exclude .venv --exclude .env --exclude __pycache__ \
  --exclude .pytest_cache --exclude .ruff_cache --exclude .mypy_cache \
  ./ "$REMOTE:projects/binoculars-eu/"

step "3/5 eval pins + package"
ssh "$REMOTE" 'bash -s' <<'EOS'
set -euo pipefail
source ~/.venv-binoculars-eu/bin/activate
cd ~/projects/binoculars-eu
pip install -q -r requirements-eval.txt
pip install -q fastapi "uvicorn[standard]" pydantic requests httpx
pip install -q -e . --no-deps
python -c "import transformers, sklearn, numpy; print('transformers', transformers.__version__, '| sklearn', sklearn.__version__, '| numpy', numpy.__version__)"
EOS

step "4/5 HF token + weights (Base + SFT-1.0)"
ssh "$REMOTE" 'mkdir -p ~/.cache/huggingface && chmod 700 ~/.cache/huggingface'
scp -q ~/.cache/huggingface/token "$REMOTE:~/.cache/huggingface/token"
ssh "$REMOTE" 'chmod 600 ~/.cache/huggingface/token'
ssh "$REMOTE" 'bash -s' <<'EOS'
set -euo pipefail
source ~/.venv-binoculars-eu/bin/activate
hf download OpenLLM-France/Luciole-1B-Base --quiet
hf download OpenLLM-France/Luciole-1B-SFT-1.0 --quiet
echo "weights present:"
du -sh ~/.cache/huggingface/hub/models--OpenLLM-France--Luciole-1B-* 2>/dev/null
EOS

step "5/5 GPU smoke scoring test"
ssh "$REMOTE" 'bash -s' <<'EOS'
set -euo pipefail
source ~/.venv-binoculars-eu/bin/activate
cd ~/projects/binoculars-eu
python - <<'PY'
from binoculars_eu import Binoculars

detector = Binoculars.for_language("fr")
score = detector.compute_score(
    "Dans le paysage numérique en constante évolution, il est crucial de "
    "tirer parti des synergies pour naviguer dans un écosystème complexe."
)
print("SMOKE score:", score, "| devices:", detector.device_1, "/", detector.device_2)
PY
EOS

echo
echo "SETUP_DONE"
