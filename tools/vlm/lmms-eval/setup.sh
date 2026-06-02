#!/usr/bin/env bash
# Stitch local lmms-eval additions into the upstream submodule via symlinks.
#
# Layout (everything relative to tools/vlm/lmms-eval/):
#
#   ext/    overlays. For each file at  ext/<rel>,  setup.sh creates a
#           symlink at  repo/lmms_eval/<rel>  pointing back to it.
#           Custom tasks land in repo/lmms_eval/tasks/<name>/ and get
#           picked up by the standard task registry — no --include_path
#           flag, and `register_model` decorators in custom model files
#           are honored because the files live at the canonical path.
#
#   scripts/  human-run shell scripts (Step 5 launcher etc.). NOT
#             symlinked; invoke them directly from this directory.
#
#   repo/   git submodule (lmms-eval upstream). The only modifications
#           inside are the symlinks this script installs.
#
# Adapted from the equivalent pattern in motion-affordance's lmms-eval overlay.
#
# Usage:
#   git submodule update --init tools/vlm/lmms-eval/repo
#   tools/vlm/lmms-eval/setup.sh

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$HERE/repo" ] || [ -z "$(ls -A "$HERE/repo" 2>/dev/null)" ]; then
    echo "Error: $HERE/repo is missing or empty."
    echo ""
    echo "Initialize the lmms-eval submodule first:"
    echo "  git submodule update --init tools/vlm/lmms-eval/repo"
    exit 1
fi

cd "$HERE"

# 1. Sweep any existing symlinks under repo/ whose target points into our
#    ext/ tree. Idempotent — re-running setup.sh after moving an ext file
#    leaves no dangling links behind.
swept=0
while IFS= read -r -d '' link; do
    target="$(readlink "$link" 2>/dev/null || true)"
    case "$target" in
        */ext/*)
            rm -f "$link"
            swept=$((swept+1))
            ;;
    esac
done < <(find repo -type l -print0 2>/dev/null)

# 2. Install fresh symlinks. One rule:  ext/<rel>  →  repo/lmms_eval/<rel>.
count=0
overrides=0
while IFS= read -r -d '' src; do
    rel="${src#ext/}"
    dst="repo/lmms_eval/$rel"
    # Number of '../' needed for the relative symlink target: each segment
    # of (lmms_eval + rel) is one level deeper than tools/vlm/lmms-eval/.
    rel_slashes=$(awk -F/ '{print NF-1}' <<< "$rel")
    ups=$((rel_slashes + 2))   # +2 for repo/ and lmms_eval/
    prefix=""
    for ((i=0; i<ups; i++)); do prefix+="../"; done
    target="${prefix}ext/${rel}"
    if [ -e "$dst" ] && [ ! -L "$dst" ]; then
        overrides=$((overrides+1))
    fi
    mkdir -p "$(dirname "$dst")"
    ln -sfn "$target" "$dst"
    count=$((count+1))
done < <(find ext -type f -print0 2>/dev/null)

echo "Swept $swept stale symlink(s); installed $count new (of which $overrides replace upstream files)."
