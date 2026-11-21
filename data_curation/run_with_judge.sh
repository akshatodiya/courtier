#!/bin/zsh
# Start the unchanged archived Claude CLI shim (Haiku 5.5 judge) on 8767, run the given command, then always stop it.
# The shim log goes to the Aya data folder so the archived TinyAya tree is not written to.
# usage: curation/run_with_judge.sh <command...>
set -u
COURTIER=~/courtier/tiny_aya_3_35b/courtier
LOG=~/courtier/aya_expanse_8b/data/story_scenes/claude_shim.log
if nc -z 127.0.0.1 8767 2>/dev/null; then echo "port 8767 busy; refusing to start a second shim" >&2; exit 2; fi
export NODE_EXTRA_CA_CERTS=~/courtier/learn/.venv/lib/python3.12/site-packages/certifi/cacert.pem
(cd "$COURTIER" && exec .venv/bin/python -u scripts/claude_shim.py 8767 ${CLAUDE_SLOTS:-4}) >> "$LOG" 2>&1 &
SHIM=$!
trap 'kill $SHIM 2>/dev/null; wait $SHIM 2>/dev/null' EXIT INT TERM
for i in {1..50}; do nc -z 127.0.0.1 8767 2>/dev/null && break; sleep 0.2; done
"$@"
