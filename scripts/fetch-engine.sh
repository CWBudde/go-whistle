#!/usr/bin/env bash
# Fetch the closed-source Needle 3 engine (the test oracle) into engine/<platform>/.
#
# The engine is used locally as a black-box oracle only: never commit or redistribute it.
# Every oracle run must disable upstream telemetry:
#
#   NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 engine/<platform>/needle ...
#
# Layout on HF Cactus-Compute/needle3 (see PLAN.md, Phase 0):
#   <platform>/needle, <platform>/needle.h            CLI + C header
#   python/cactus_needle-3.2.0-py3-none-<tag>.whl     the only place libneedle3.{dylib,so} ships
#
# The loose binaries on the HF main branch are replaced in place, so everything is pinned to one
# repo revision and verified by SHA-256.
#
# Usage: scripts/fetch-engine.sh [--platform macos-arm64|linux-x86_64|linux-arm64]

set -euo pipefail

REPO="Cactus-Compute/needle3"
REVISION="2ae11323dc000f5e70c49f7403efa6af12ba9e67"
ENGINE_VERSION="3.2.0"
BASE_URL="https://huggingface.co/${REPO}/resolve/${REVISION}"
HEADER_SHA256="914bbd00423939b02fc349f91cdb7c12531b3f5bb7105b743e0e93c7e409d7b1"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

detect_platform() {
	local os arch
	os="$(uname -s)"
	arch="$(uname -m)"
	case "${os}/${arch}" in
	Darwin/arm64) echo "macos-arm64" ;;
	Linux/x86_64) echo "linux-x86_64" ;;
	Linux/aarch64 | Linux/arm64) echo "linux-arm64" ;;
	*)
		echo "unsupported platform: ${os}/${arch}" >&2
		return 1
		;;
	esac
}

sha256_of() {
	if command -v sha256sum >/dev/null 2>&1; then
		sha256sum "$1" | cut -d' ' -f1
	else
		shasum -a 256 "$1" | cut -d' ' -f1
	fi
}

# fetch <url> <dest> <sha256>: download unless dest already has the expected digest.
fetch() {
	local url="$1" dest="$2" want="$3" got
	if [[ -f ${dest} && "$(sha256_of "${dest}")" == "${want}" ]]; then
		echo "ok      ${dest#"${ROOT}"/}"
		return 0
	fi
	echo "fetch   ${url}"
	curl -fL --retry 3 --progress-bar -o "${dest}.part" "${url}"
	got="$(sha256_of "${dest}.part")"
	if [[ ${got} != "${want}" ]]; then
		rm -f "${dest}.part"
		echo "sha256 mismatch for ${url}: got ${got}, want ${want}" >&2
		return 1
	fi
	mv "${dest}.part" "${dest}"
	echo "ok      ${dest#"${ROOT}"/}"
}

platform=""
while [[ $# -gt 0 ]]; do
	case "$1" in
	--platform)
		platform="${2:?--platform needs a value}"
		shift 2
		;;
	-h | --help)
		sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
		exit 0
		;;
	*)
		echo "unknown argument: $1" >&2
		exit 2
		;;
	esac
done
[[ -n ${platform} ]] || platform="$(detect_platform)"

case "${platform}" in
macos-arm64)
	wheel_tag="macosx_11_0_arm64"
	lib_name="libneedle3.dylib"
	needle_sha256="342fa2c6f140e702354a99c4201c9057535ec908eed35c7382e911a19d1d2724"
	wheel_sha256="3b0887a43cd6e9a99009fabf35b231c11bb3a978ab8af94b8119b2eb76e19832"
	;;
linux-x86_64)
	wheel_tag="manylinux2014_x86_64"
	lib_name="libneedle3.so"
	needle_sha256="f38dc4b0345d66b4e385734ad0f12af43ac6e2cfa1752d0795af5c137200c8e4"
	wheel_sha256="0e8a3bce4e52968ee14e8e9d47098e65ed06dd5b7c18cef3a7fa1375756736f9"
	;;
linux-arm64)
	wheel_tag="manylinux2014_aarch64"
	lib_name="libneedle3.so"
	needle_sha256="6fc25a97def475e1c7d1933a4a219d4e37076be77974e76dd331d9e0cd97f3cb"
	wheel_sha256="80871585c84a277eff24b43a9366958996be38f6f920c9d1f8a0d2a07f3ae51f"
	;;
*)
	echo "unsupported platform: ${platform}" >&2
	exit 2
	;;
esac

dest="${ROOT}/engine/${platform}"
mkdir -p "${dest}"

fetch "${BASE_URL}/${platform}/needle" "${dest}/needle" "${needle_sha256}"
chmod +x "${dest}/needle"
fetch "${BASE_URL}/${platform}/needle.h" "${dest}/needle.h" "${HEADER_SHA256}"

wheel="cactus_needle-${ENGINE_VERSION}-py3-none-${wheel_tag}.whl"
fetch "${BASE_URL}/python/${wheel}" "${dest}/${wheel}" "${wheel_sha256}"
unzip -o -q -j "${dest}/${wheel}" "needle/${lib_name}" -d "${dest}"
chmod 644 "${dest}/${lib_name}"
echo "ok      engine/${platform}/${lib_name} (from ${wheel})"

{
	echo "repo=${REPO}"
	echo "revision=${REVISION}"
	echo "engine_version=${ENGINE_VERSION}"
	echo "platform=${platform}"
} >"${dest}/PIN"

echo "engine ${ENGINE_VERSION} (${platform}) ready in engine/${platform}/"
