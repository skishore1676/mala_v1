#!/bin/bash
# Wrapper invoked by launchd for Mala's nightly regime matrix scout.
# Matches the pattern used by Bhiksha's launchd wrappers (cd to repo root,
# exec the uv-managed Python, let launchd capture stdout/stderr via the
# plist's StandardOutPath / StandardErrorPath).
#
# Install once:
#   launchctl load -w ~/Library/LaunchAgents/com.mala.nightly_regime_matrix.plist
# Unload:
#   launchctl unload ~/Library/LaunchAgents/com.mala.nightly_regime_matrix.plist

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Make sure uv is on PATH even when launchd hands us a minimal environment.
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

# Log the invocation with a timestamp so the log file is self-describing.
echo "=== mala nightly regime matrix @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "PATH=$PATH"
echo "uv=$(command -v uv || echo missing)"

exec uv run python scripts/run_nightly_regime_matrix.py \
    --config config/nightly_regime_matrix.yaml
