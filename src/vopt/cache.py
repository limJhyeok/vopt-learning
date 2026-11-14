"""빌드 산출물(TensorRT engine 등) 캐시.

조회 순서: 로컬 캐시 디렉토리 → GitHub Release 다운로드 → 직접 빌드(후 캐시에 저장)

- 캐시 디렉토리: $VOPT_CACHE > Google Drive(마운트돼 있으면) > /content/vopt-cache
- GitHub Release: https://github.com/<repo>/releases/download/<tag>/<filename>
  업로드는 로컬에서 scripts/publish_artifact.sh 로 한다.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

GITHUB_REPO = "limJhyeok/vopt-learning"
RELEASE_TAG = "artifacts"
DRIVE_DIR = Path("/content/drive/MyDrive/vopt-cache")


def cache_dir() -> Path:
    if env := os.environ.get("VOPT_CACHE"):
        d = Path(env)
    elif DRIVE_DIR.parent.exists():  # Drive가 마운트된 경우: 런타임이 끊겨도 유지됨
        d = DRIVE_DIR
    elif Path("/content").exists():
        d = Path("/content/vopt-cache")
    else:
        d = Path.home() / ".cache" / "vopt"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fingerprint(*files: str | Path, **config) -> str:
    """입력 파일 내용 + 빌드 설정의 해시(8자). 입력이나 설정이 바뀌면 다른 캐시 키가 된다."""
    h = hashlib.sha256()
    for f in files:
        with open(f, "rb") as fp:
            for chunk in iter(lambda: fp.read(1 << 20), b""):
                h.update(chunk)
    h.update(json.dumps(config, sort_keys=True, default=str).encode())
    return h.hexdigest()[:8]


def release_url(filename: str, repo: str = GITHUB_REPO, tag: str = RELEASE_TAG) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{filename}"


def download(url: str, dst: Path) -> bool:
    """성공하면 True, 404 등으로 없으면 False."""
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        if e.code == 404:
            return False
        raise
    tmp.replace(dst)
    return True


def get_or_create(
    filename: str,
    build_fn: Callable[[Path], None],
    *,
    use_release: bool = True,
    force_rebuild: bool = False,
) -> Path:
    """`filename` 산출물 경로를 돌려준다. 없으면 release에서 받고, 그래도 없으면 build_fn(path)로 만든다."""
    path = cache_dir() / filename
    if path.exists() and not force_rebuild:
        print(f"[cache] hit: {path}")
        return path
    if use_release and not force_rebuild:
        url = release_url(filename)
        if download(url, path):
            print(f"[cache] downloaded: {url}")
            return path
        print(f"[cache] not in release: {filename}")
    print(f"[cache] building: {filename}")
    tmp = path.with_suffix(path.suffix + ".part")
    build_fn(tmp)
    tmp.replace(path)
    print(f"[cache] saved: {path}")
    return path
