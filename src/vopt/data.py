"""벤치마크/정확도 비교용 샘플 이미지 배치.

이미지 출처: Lorem Picsum(https://picsum.photos). 사진은 Unsplash에서 가져온 것이며
Unsplash License(https://unsplash.com/license)로 상업/비상업 용도 모두 무료로 쓸 수 있다.

`/id/{id}` 고정 URL을 쓰므로 매번 같은 사진이 내려온다(재현 가능). engine과 같은 vopt.cache 경로를 쓴다:
로컬/Drive 캐시 → GitHub Release(`scripts/publish_artifact.sh`로 올린 사진) → picsum 다운로드.
picsum이 불안정할 때를 대비해 한 번 받은 `picsum-*.jpg`를 Release에 올려두면 좋다(Unsplash License상 재배포 가능).
"""

from __future__ import annotations

from pathlib import Path

from . import cache

PICSUM_URL = "https://picsum.photos/id/{id}/{size}/{size}.jpg"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _fetch(image_id: int, size: int) -> Path | None:
    """picsum 사진 하나의 로컬 경로. 캐시 → GitHub Release → picsum 순서로 찾는다. 실패하면 None."""

    def download_from_picsum(dst: Path) -> None:
        url = PICSUM_URL.format(id=image_id, size=size)
        if not cache.download(url, dst):
            raise FileNotFoundError(f"picsum에 없는 id: {image_id}")

    try:
        return cache.get_or_create(f"picsum-{image_id}-{size}.jpg", download_from_picsum)
    except FileNotFoundError as e:  # picsum에 빠진 id → 다음 id로
        print(f"[data] skip: {e}")
        return None
    except OSError as e:  # 5xx, timeout 등 서비스 장애: id를 바꿔도 소용없으므로 바로 중단
        raise RuntimeError(f"picsum 다운로드 실패({e}). 잠시 후 다시 시도하세요.") from e


def sample_images(n: int, *, start_id: int = 10, size: int = 256) -> list[Path]:
    """picsum id를 start_id부터 차례로 시도해 n장을 모은다(빠진 id는 건너뜀)."""
    paths: list[Path] = []
    image_id = start_id
    while len(paths) < n:
        if image_id > start_id + 4 * n + 50:
            raise RuntimeError(f"이미지 {n}장을 받지 못했습니다({len(paths)}장 성공).")
        if (p := _fetch(image_id, size)) is not None:
            paths.append(p)
        image_id += 1
    return paths


def imagenet_batch(batch_size: int, *, device: str = "cuda", start_id: int = 10):
    """ResNet 표준 전처리(Resize 256 → CenterCrop 224 → Normalize)를 거친 [B, 3, 224, 224] fp32 텐서."""
    import torch
    from PIL import Image
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    paths = sample_images(batch_size, start_id=start_id)
    tensors = [preprocess(Image.open(p).convert("RGB")) for p in paths]
    return torch.stack(tensors).to(device)
