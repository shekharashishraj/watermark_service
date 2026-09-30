#!/bin/bash
# Clone O-SCD and MV3DCD at the commits the plan was written against and apply our patches.
# Run on a login node (needs internet).      bash setup/01_fetch_code.sh
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/sol.env"

fetch() {  # url dir commit patch
    local url="$1" dir="$2" commit="$3" patch="$4"
    if [ ! -d "$dir/.git" ]; then
        git clone "$url" "$dir"
    fi
    git -C "$dir" fetch --quiet origin
    if ! git -C "$dir" diff --quiet; then
        echo "[scd] $dir has local changes; leaving it alone (git -C $dir stash to reset)" >&2
        return 1
    fi
    git -C "$dir" checkout --quiet "$commit"
    git -C "$dir" apply --check "$patch"
    git -C "$dir" apply "$patch"
    echo "[scd] $dir @ $(git -C "$dir" rev-parse --short HEAD) + $(basename "$patch")"
}

fetch https://github.com/Chumsy0725/O-SCD  "$SCD_ROOT/code/O-SCD"  3abfeaa "$SCD_REPO/patches/oscd.patch"
fetch https://github.com/Chumsy0725/MV3DCD "$SCD_ROOT/code/MV3DCD" b45606c "$SCD_REPO/patches/mv3dcd.patch"
echo "[scd] patched code ready under $SCD_ROOT/code (git diff shows the patch; git checkout -- . reverts it)"
