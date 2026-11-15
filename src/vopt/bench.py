"""GPU latency 벤치마크 (CUDA event 기반)."""

from __future__ import annotations

from typing import Callable


def benchmark(fn: Callable[[], object], *, warmup: int = 20, iters: int = 100) -> dict[str, object]:
    """fn()을 반복 실행하고 latency(ms) 통계와 매 반복의 출력(outs)을 돌려준다."""
    import torch

    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        outs = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
            # clone(측정 구간 밖): CUDA graph(torch.compile mode="reduce-overhead" 등)는
            # 매 호출 같은 출력 버퍼를 재사용하므로 복사하지 않으면 모든 outs가 마지막 값이 된다.
            outs.append(out.clone())
    t = torch.tensor(times)
    return {
        "mean_ms": t.mean().item(),
        "p50_ms": t.median().item(),
        "p90_ms": t.quantile(0.9).item(),
        "min_ms": t.min().item(),
        "outs": outs,
    }


def compare(results: dict[str, dict[str, object]], baseline: str | None = None) -> None:
    """{name: benchmark결과}를 표로 출력. baseline 대비 speedup과 출력 오차를 함께 보여준다.

    모든 반복(iters)의 출력을 torch.stack으로 묶어 baseline과 비교한다([iters, B, num_classes]).
      max_diff : 최대 절대 오차
      rel_err  : 상대 오차 ||cur - base|| / ||base|| (logit 크기와 무관한 비교)
      cos_sim  : 샘플별 logit 벡터 코사인 유사도의 최솟값
      top1     : top-1 예측 일치율(%)
    """
    import torch
    import torch.nn.functional as F

    baseline = baseline or next(iter(results))
    base_mean = results[baseline]["mean_ms"]
    base_outs = torch.stack(results[baseline]["outs"]).float()
    base_preds = base_outs.argmax(dim=-1)

    print(
        f"{'name':<24}{'mean(ms)':>10}{'p50':>10}{'p90':>10}{'speedup':>10}"
        f"{'max_diff':>11}{'rel_err':>11}{'cos_sim':>10}{'top1':>9}"
    )
    print("-" * 105)

    for name, r in results.items():
        cur_outs = torch.stack(r["outs"]).float()
        max_diff = (cur_outs - base_outs).abs().max().item()
        rel_err = ((cur_outs - base_outs).norm() / base_outs.norm()).item()
        cos_sim = F.cosine_similarity(cur_outs, base_outs, dim=-1).min().item()
        match_rate = (cur_outs.argmax(dim=-1) == base_preds).float().mean().item() * 100.0
        speedup = base_mean / r["mean_ms"]
        print(
            f"{name:<24}"
            f"{r['mean_ms']:>10.3f}"
            f"{r['p50_ms']:>10.3f}"
            f"{r['p90_ms']:>10.3f}"
            f"{speedup:>9.2f}x"
            f"{max_diff:>11.4f}"
            f"{rel_err:>11.2e}"
            f"{cos_sim:>10.6f}"
            f"{match_rate:>8.1f}%"
        )
