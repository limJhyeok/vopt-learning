"""실행 환경 정보. 블로그 글에 'T4 / TRT 10.x 기준' 같은 재현 정보를 남기는 용도."""

from __future__ import annotations

import platform
import re


def gpu_tag() -> str:
    """engine 캐시 키용 GPU 식별자. 예: 'tesla-t4-sm75'.

    TensorRT engine은 GPU 아키텍처(compute capability)마다 호환되지 않으므로 키에 반드시 포함한다.
    """
    import torch

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"{slug}-sm{major}{minor}"


def env_info() -> dict[str, str]:
    info = {"python": platform.python_version()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda (torch)"] = str(torch.version.cuda)
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu tag"] = gpu_tag()
        else:
            info["gpu"] = "N/A (런타임 유형에서 GPU를 선택했는지 확인)"
    except ImportError:
        info["torch"] = "not installed"
    try:
        import tensorrt as trt

        info["tensorrt"] = trt.__version__
    except ImportError:
        info["tensorrt"] = "not installed"
    return info


def print_env() -> None:
    info = env_info()
    width = max(map(len, info))
    for k, v in info.items():
        print(f"{k:<{width}} : {v}")
