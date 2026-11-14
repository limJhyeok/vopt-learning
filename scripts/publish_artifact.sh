#!/usr/bin/env bash
# Colab에서 받은 빌드 산출물(.engine 등)을 GitHub Release에 올린다.
# 노트북의 vopt.cache가 같은 파일명으로 자동 다운로드하므로 이후 study 때는 빌드를 건너뛴다.
#
# 사용법:
#   1) VS Code 명령 팔레트 → "Colab: Download..." 로 /content/vopt-cache/*.engine 을 artifacts/ 에 받기
#   2) scripts/publish_artifact.sh artifacts/*.engine
#
# 파일명(캐시 키)을 바꾸지 말 것. 이름에 GPU 아키텍처/TRT 버전/해시가 들어 있다.
set -euo pipefail

TAG="${VOPT_RELEASE_TAG:-artifacts}"
[[ $# -ge 1 ]] || { echo "usage: $0 FILE..." >&2; exit 1; }

if ! gh release view "$TAG" >/dev/null 2>&1; then
  gh release create "$TAG" --title "Prebuilt artifacts" \
    --notes "Prebuilt TensorRT engines etc. Filenames encode GPU arch + TensorRT version." --latest=false
fi
gh release upload "$TAG" "$@" --clobber
echo "uploaded to release '$TAG':"; printf '  %s\n' "$@"
