"""벤치마크/정확도 비교용 샘플 이미지 배치.

데이터셋: Oxford-IIIT Pet(Hugging Face `timm/oxford-iiit-pet`, test split), 라이선스 CC BY-SA 4.0.
  Parkhi et al., "Cats and Dogs", CVPR 2012. https://www.robots.ox.ac.uk/~vgg/data/pets/
  개/고양이 품종 사진이라 ImageNet 클래스와 겹쳐서 top-1 비교가 의미 있다.

전체(~800MB)를 받지 않고 Hugging Face datasets-server의 `/rows` API로 필요한 행(이미지)만 받는다.
test split은 이미 섞여 있어서 앞에서부터 n장만 받아도 여러 품종이 나온다.
받은 jpg는 캐시 디렉토리에 저장해서 다음 실행부터는 네트워크 없이 읽는다.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

from . import cache

HF_DATASET = "timm/oxford-iiit-pet"
HF_CONFIG = "default"
HF_SPLIT = "test"
ROWS_API = "https://datasets-server.huggingface.co/rows"
MAX_ROWS_PER_REQUEST = 100  # datasets-server 제한

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _image_path(idx: int) -> Path:
    d = cache.cache_dir() / "images" / HF_DATASET.replace("/", "--") / HF_SPLIT
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{idx:05d}.jpg"


def _fetch_rows(offset: int, length: int) -> list[dict]:
    query = urllib.parse.urlencode(
        {"dataset": HF_DATASET, "config": HF_CONFIG, "split": HF_SPLIT, "offset": offset, "length": length}
    )
    with urllib.request.urlopen(f"{ROWS_API}?{query}", timeout=60) as r:
        return json.load(r)["rows"]


def sample_images(n: int, *, offset: int = 0) -> list[Path]:
    """데이터셋 [offset, offset + n) 행의 이미지 경로. 캐시에 없는 것만 다운로드한다."""
    indices = range(offset, offset + n)
    missing = [i for i in indices if not _image_path(i).exists()]
    if missing:
        print(f"[data] {HF_DATASET} ({HF_SPLIT}) 이미지 {len(missing)}장 다운로드 중...")
        lo, hi = missing[0], missing[-1] + 1
        for start in range(lo, hi, MAX_ROWS_PER_REQUEST):
            for row in _fetch_rows(start, min(MAX_ROWS_PER_REQUEST, hi - start)):
                path = _image_path(row["row_idx"])
                # 이미지 URL은 서명된 임시 URL이므로 캐시에 바로 저장해 둔다.
                if not path.exists() and not cache.download(row["row"]["image"]["src"], path):
                    raise RuntimeError(f"이미지 다운로드 실패: row {row['row_idx']}")
    return [_image_path(i) for i in indices]


def imagenet_batch(batch_size: int, *, device: str = "cuda", offset: int = 0):
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
    paths = sample_images(batch_size, offset=offset)
    tensors = [preprocess(Image.open(p).convert("RGB")) for p in paths]
    return torch.stack(tensors).to(device)
