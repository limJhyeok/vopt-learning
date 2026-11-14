"""GPU latency 벤치마크 (CUDA event 기반)."""

from __future__ import annotations

from typing import Callable


def benchmark(fn: Callable[[], object], *, warmup: int = 20, iters: int = 100) -> dict[str, float]:
    """fn()을 반복 실행하고 latency(ms) 통계를 돌려준다."""
    import torch

    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
    t = torch.tensor(times)
    return {
        "mean_ms": t.mean().item(),
        "p50_ms": t.median().item(),
        "p90_ms": t.quantile(0.9).item(),
        "min_ms": t.min().item(),
    }


def compare(results: dict[str, dict[str, float]], baseline: str | None = None) -> None:
    """{name: benchmark결과} 를 표로 출력. baseline 기준 speedup 포함."""
    baseline = baseline or next(iter(results))
    base = results[baseline]["mean_ms"]
    print(f"{'name':<20}{'mean(ms)':>10}{'p50':>10}{'p90':>10}{'speedup':>10}")
    for name, r in results.items():
        print(
            f"{name:<20}{r['mean_ms']:>10.3f}{r['p50_ms']:>10.3f}{r['p90_ms']:>10.3f}"
            f"{base / r['mean_ms']:>9.2f}x"
        )
