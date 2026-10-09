#!/usr/bin/env bash
# Snapshot upstream cactus-compute/needle at the pinned commit into original/needle (git-ignored).
#
# Idempotent: re-running with the checkout already at the pin does nothing.
#
# Usage: scripts/fetch-upstream.sh

set -euo pipefail

URL="https://github.com/cactus-compute/needle.git"
PIN="af2654dffa4987f7e2cbbacdf59c2f854fd18302"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/original/needle"

if [[ -d "${DEST}/.git" ]] && [[ "$(git -C "${DEST}" rev-parse HEAD 2>/dev/null)" == "${PIN}" ]]; then
	echo "ok      original/needle already at ${PIN}"
	exit 0
fi

mkdir -p "${DEST}"
if [[ ! -d "${DEST}/.git" ]]; then
	git -C "${DEST}" init -q
	git -C "${DEST}" remote add origin "${URL}"
fi
git -C "${DEST}" fetch -q --depth 1 origin "${PIN}"
git -C "${DEST}" checkout -q --detach FETCH_HEAD

head="$(git -C "${DEST}" rev-parse HEAD)"
if [[ ${head} != "${PIN}" ]]; then
	echo "original/needle is at ${head}, want ${PIN}" >&2
	exit 1
fi
echo "ok      original/needle at ${PIN}"
