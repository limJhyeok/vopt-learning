"""TensorRT(10.x) 빌드/로드/추론 래퍼. torch 텐서를 그대로 입출력으로 쓴다."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from . import cache, env

Shape = Sequence[int]


def _logger(verbose: bool = False):
    import tensorrt as trt

    return trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)


def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    fp16: bool = False,
    workspace_gb: float = 4,
    shapes: dict[str, tuple[Shape, Shape, Shape]] | None = None,
    verbose: bool = False,
) -> Path:
    """ONNX → serialized engine 파일.

    shapes: 동적 입력용 {input_name: (min, opt, max)}. 정적 shape 모델이면 None.
    """
    import tensorrt as trt

    logger = _logger(verbose)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # TRT 10: explicit batch가 기본
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if shapes:
        profile = builder.create_optimization_profile()
        for name, (mn, opt, mx) in shapes.items():
            profile.set_shape(name, mn, opt, mx)
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build failed (verbose=True로 로그 확인)")
    engine_path = Path(engine_path)
    engine_path.write_bytes(serialized)
    return engine_path


def engine_filename(name: str, onnx_path: str | Path, **build_kwargs) -> str:
    """캐시 키 = 이름 + (ONNX 내용, 빌드 옵션) 해시 + GPU 아키텍처 + TRT 버전."""
    import tensorrt as trt

    fp = cache.fingerprint(onnx_path, **build_kwargs)
    return f"{name}-{fp}-{env.gpu_tag()}-trt{trt.__version__}.engine"


def get_engine(
    name: str,
    onnx_path: str | Path,
    *,
    force_rebuild: bool = False,
    use_release: bool = True,
    **build_kwargs,
) -> Path:
    """캐시/Release에 있으면 그것을, 없으면 빌드해서 engine 경로를 돌려준다."""
    filename = engine_filename(name, onnx_path, **build_kwargs)
    return cache.get_or_create(
        filename,
        lambda dst: build_engine(onnx_path, dst, **build_kwargs),
        use_release=use_release,
        force_rebuild=force_rebuild,
    )


def load_engine(engine_path: str | Path):
    import tensorrt as trt

    runtime = trt.Runtime(_logger())
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    if engine is None:
        raise RuntimeError(
            f"deserialize failed: {engine_path} (다른 GPU/TRT 버전에서 빌드된 engine일 수 있음)"
        )
    return engine


class TRTModule:
    """engine을 torch 모듈처럼 호출: outputs = mod(x) 또는 mod(input_name=x)."""

    def __init__(self, engine_or_path):
        import tensorrt as trt

        self.engine = (
            load_engine(engine_or_path) if isinstance(engine_or_path, (str, Path)) else engine_or_path
        )
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        mode = self.engine.get_tensor_mode
        self.input_names = [n for n in names if mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in names if mode(n) == trt.TensorIOMode.OUTPUT]

    def _torch_dtype(self, name: str):
        import tensorrt as trt
        import torch

        table = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.bfloat16: torch.bfloat16,
            trt.int8: torch.int8,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
            trt.bool: torch.bool,
            trt.uint8: torch.uint8,
        }
        return table[self.engine.get_tensor_dtype(name)]

    def __call__(self, *args, **kwargs):
        import torch

        feeds = dict(zip(self.input_names, args)) | kwargs
        stream = torch.cuda.current_stream()
        for name in self.input_names:
            t = feeds[name].contiguous()
            feeds[name] = t  # 실행 끝날 때까지 참조 유지
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())
        outputs = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            out = torch.empty(shape, dtype=self._torch_dtype(name), device="cuda")
            self.context.set_tensor_address(name, out.data_ptr())
            outputs[name] = out
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        vals = list(outputs.values())
        return vals[0] if len(vals) == 1 else outputs
