"""vopt: study 본문과 무관한 boilerplate(환경 출력, engine 캐시, 추론 래퍼, 벤치마크)를 모아둔 헬퍼.

GPU 패키지(torch, tensorrt)는 함수 안에서 lazy import 하므로 GPU가 없는 로컬에서도 `import vopt`는 된다.
"""

__all__ = ["env", "cache", "trt_utils", "bench", "data"]
