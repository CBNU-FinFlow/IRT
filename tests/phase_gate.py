#!/usr/bin/env python3
"""
IRT 평가 결과를 사전 정의된 검증 게이트와 비교하는 도구.

사용법:
    python tests/phase_gate.py --eval logs/irt/<time>/evaluation_insights.json

`scripts/evaluate.py`가 생성한 `evaluation_insights.json`을 읽어들인 뒤,
가능한 경우 `env_meta.json`과 비교해 학습·평가 설정이 일치하는지도 확인한다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class GateResult:
    gate: str
    description: str
    passed: bool
    value: Any
    threshold: str
    details: str = ""


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _try_load_env_meta(eval_path: Path) -> Optional[Dict[str, Any]]:
    candidates = [
        eval_path.parent / "env_meta.json",
        eval_path.parent.parent / "env_meta.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return _load_json(candidate)
            except Exception:
                return None
    return None


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.4f}"
    return str(value)


def check_gates(
    insights: Dict[str, Any], env_meta: Optional[Dict[str, Any]] = None
) -> List[GateResult]:
    summary = insights.get("summary", {})
    risk_metrics = insights.get("risk_metrics", {})
    prototype = insights.get("prototype_analysis", {})
    irt_decomp = insights.get("irt_decomposition", {})
    tcell = insights.get("tcell_insights", {})

    crisis_pct = float(tcell.get("crisis_regime_pct", 0.0) or 0.0)

    alpha_std = float(irt_decomp.get("std_alpha_c", 0.0) or 0.0)
    alpha_min = float(irt_decomp.get("min_alpha_c", 0.0) or 0.0)
    alpha_max = float(irt_decomp.get("max_alpha_c", 0.0) or 0.0)

    avg_entropy = float(prototype.get("avg_entropy", 0.0) or 0.0)
    max_proto_weight = float(prototype.get("max_prototype_weight", 0.0) or 0.0)

    turnover_target = float(risk_metrics.get("turnover_target", 0.0) or 0.0)
    turnover_actual = float(risk_metrics.get("turnover_actual", 0.0) or 0.0)
    turnover_ratio = float(risk_metrics.get("turnover_transmission_rate", 0.0) or 0.0)

    contribution_abs_sum = float(
        insights.get("top_holdings_contribution_abs_sum", 0.0) or 0.0
    )
    contribution_sum = float(insights.get("top_holdings_contribution_sum", 0.0) or 0.0)

    reward_type_eval = summary.get("reward_type")
    reward_scale_eval = summary.get("reward_scaling")

    reward_type_train = env_meta.get("reward_type") if env_meta else None
    reward_scale_train = env_meta.get("reward_scaling") if env_meta else None

    gates: List[GateResult] = []

    # Gate A: Crisis coverage >= 10%
    gates.append(
        GateResult(
            gate="A",
            description="Crisis regime coverage ≥ 10%",
            passed=crisis_pct >= 0.10,
            value=crisis_pct,
            threshold=">= 0.10",
            details="tcell_insights.crisis_regime_pct",
        )
    )

    # Gate B: Alpha variability present
    gates.append(
        GateResult(
            gate="B",
            description="alpha_c variance ≥ 0.05 and min < max",
            passed=(alpha_std >= 0.05) and (alpha_max - alpha_min > 1e-6),
            value={"std": alpha_std, "min": alpha_min, "max": alpha_max},
            threshold="std >= 0.05 & max>min",
        )
    )

    # Gate C: Prototype selectivity
    gates.append(
        GateResult(
            gate="C",
            description="Prototype entropy in [1.4, 1.9] and max weight > 0.2",
            passed=(1.4 <= avg_entropy <= 1.9) and (max_proto_weight > 0.2),
            value={"avg_entropy": avg_entropy, "max_weight": max_proto_weight},
            threshold="avg_entropy ∈ [1.4,1.9], max_weight > 0.2",
        )
    )

    # Gate D: Turnover alignment absolute difference within 10%
    if turnover_target <= 1e-6:
        turnover_pass = abs(turnover_actual) <= 1e-6
    else:
        turnover_pass = abs(turnover_actual - turnover_target) <= 0.10 * turnover_target
    gates.append(
        GateResult(
            gate="D",
            description="Turnover actual vs target within 10%",
            passed=turnover_pass,
            value={"actual": turnover_actual, "target": turnover_target},
            threshold="|actual-target| <= 10% of target (or both ≈0)",
        )
    )

    # Gate E: Reward config parity
    reward_parity = False
    if reward_type_eval is not None and reward_type_train is not None:
        if reward_scale_eval is not None and reward_scale_train is not None:
            try:
                reward_parity = (
                    reward_type_eval == reward_type_train
                    and abs(float(reward_scale_eval) - float(reward_scale_train))
                    <= 1e-6
                )
            except (TypeError, ValueError):
                reward_parity = False
    gates.append(
        GateResult(
            gate="E",
            description="Reward type & scale match checkpoint",
            passed=reward_parity,
            value={
                "eval_type": reward_type_eval,
                "train_type": reward_type_train,
                "eval_scale": reward_scale_eval,
                "train_scale": reward_scale_train,
            },
            threshold="match required",
            details="env_meta vs evaluation summary",
        )
    )

    # Gate F: Turnover transmission ratio close to 1 (±10%)
    gates.append(
        GateResult(
            gate="F",
            description="Turnover transmission ratio within [0.9, 1.1]",
            passed=0.9 <= turnover_ratio <= 1.1,
            value=turnover_ratio,
            threshold="0.9 ≤ ratio ≤ 1.1",
        )
    )

    # Gate G: Normalised top holding contributions sum to ≤ 1 (allowing small epsilon)
    gates.append(
        GateResult(
            gate="G",
            description="Top holding contributions properly normalised",
            passed=(contribution_abs_sum <= 1.02) and (abs(contribution_sum) <= 1.02),
            value={"sum": contribution_sum, "abs_sum": contribution_abs_sum},
            threshold="abs(sum) ≤ 1.02",
        )
    )

    return gates


def format_report(path: Path, results: List[GateResult]) -> str:
    lines = [f"Evaluation: {path}"]
    for res in results:
        status = "PASS" if res.passed else "FAIL"
        line = f"  [{res.gate}] {status:<4} | {res.description} | value={_fmt(res.value)} | threshold={res.threshold}"
        if res.details:
            line += f" ({res.details})"
        lines.append(line)
    passed = sum(1 for r in results if r.passed)
    lines.append(f"  Result: {passed}/{len(results)} gates passed")
    return "\n".join(lines)


def emit_json_report(path: Path, results: List[GateResult]) -> Dict[str, Any]:
    return {
        "evaluation": str(path),
        "summary": {
            "passed": sum(1 for r in results if r.passed),
            "total": len(results),
        },
        "gates": [
            {
                "gate": r.gate,
                "description": r.description,
                "passed": r.passed,
                "value": r.value,
                "threshold": r.threshold,
                "details": r.details,
            }
            for r in results
        ],
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="IRT 평가 게이트 검증 도구")
    parser.add_argument(
        "--eval",
        dest="eval_paths",
        action="append",
        required=True,
        help="evaluation_insights.json 경로 (여러 번 지정 가능)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="사람이 읽기 쉬운 리포트 대신 JSON 요약 출력",
    )

    args = parser.parse_args(argv)

    reports: List[str] = []
    json_payload: List[Dict[str, Any]] = []
    exit_code = 0

    for eval_path_str in args.eval_paths:
        eval_path = Path(eval_path_str).expanduser().resolve()
        try:
            insights = _load_json(eval_path)
        except Exception as exc:
            reports.append(f"Evaluation: {eval_path}\n  ERROR: {exc}")
            exit_code = 1
            continue

        env_meta = _try_load_env_meta(eval_path)
        results = check_gates(insights, env_meta)

        if not all(r.passed for r in results):
            exit_code = 2

        if args.json:
            json_payload.append(emit_json_report(eval_path, results))
        else:
            reports.append(format_report(eval_path, results))

    if args.json:
        json.dump(json_payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write("\n\n".join(reports) + "\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
