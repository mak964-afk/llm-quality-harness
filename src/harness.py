"""Прогон набора кейсов через агента: оркестрация, кэш, сравнение с baseline.

Точка входа:
    python -m src.harness --cases cases/support_agent.yaml --report out/report.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from . import groundedness as g
from . import judge as j
from . import metrics as m
from . import report as rep
from .providers import get_llm
from .retrieval import Agent, DemoAgent


def load_cases(path: Path) -> list[dict]:
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError(f"{path}: ожидался список кейсов")

    seen = set()
    for c in cases:
        if "id" not in c:
            raise ValueError(f"{path}: кейс без id: {c}")
        if c["id"] in seen:
            # Дубль id ломает сравнение с baseline: один кейс перезатрёт
            # другой, и регрессия останется незамеченной. Падаем сразу.
            raise ValueError(f"{path}: дублирующийся id кейса: {c['id']}")
        seen.add(c["id"])
    return cases


def run_case(case: dict, agent: Agent, llm, k: int = 5) -> m.CaseResult:
    started = time.perf_counter()
    answer = agent.answer(case["question"])
    latency_ms = (time.perf_counter() - started) * 1000

    retrieved_ids = [d.id for d in answer.docs]
    should_refuse = case.get("expected_behavior") == "refuse"

    # Негативный кейс: эталона нет, проверяем только - промолчал или выдумал.
    if should_refuse:
        refused = j.is_refusal(llm, answer.text)
        return m.CaseResult(
            case_id=case["id"],
            passed=refused,          # для негативного кейса отказ И ЕСТЬ успех
            retrieved_ids=retrieved_ids,
            refused=refused,
            should_refuse=True,
            latency_ms=latency_ms,
            cost_usd=answer.cost_usd,
            tags=case.get("tags", []),
        )

    verdict = j.judge(llm, case["question"], case["expected"], answer.text)
    ground = g.check(llm, answer.text, answer.docs)

    cited = bool(answer.citations)
    citation_valid = cited and all(c in retrieved_ids for c in answer.citations)

    # must_cite: ответ без валидной ссылки на источник = провал,
    # даже если фактически верен. Для регламентов, политик и юридических
    # вопросов непроверяемый ответ бесполезен: пользователь не сможет
    # сослаться на него в споре.
    passed = verdict.passed
    if case.get("must_cite") and not citation_valid:
        passed = False

    return m.CaseResult(
        case_id=case["id"],
        passed=passed,
        retrieved_ids=retrieved_ids,
        expected_source=case.get("expected_source"),
        grounded_claims=ground.grounded,
        total_claims=ground.total,
        cited=cited,
        citation_valid=citation_valid,
        refused=verdict.refused,
        should_refuse=False,
        latency_ms=latency_ms,
        cost_usd=answer.cost_usd,
        tags=case.get("tags", []),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Эвал-харнесс для LLM-агентов и RAG")
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--model", default="demo")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--report", type=Path)
    p.add_argument("--baseline", type=Path, help="сравнить с прошлым прогоном")
    p.add_argument("--save-baseline", type=Path, help="сохранить прогон как baseline")
    p.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="выйти с кодом 1, если pass@k ниже порога (для CI)",
    )
    args = p.parse_args(argv)

    cases = load_cases(args.cases)
    llm = get_llm(args.model)
    agent: Agent = DemoAgent()

    results = [run_case(c, agent, llm, args.k) for c in cases]
    summary = m.summarize(results, args.k)
    tags = m.by_tag(results, args.k)

    baseline_summary = None
    broken: list[str] = []
    if args.baseline and args.baseline.exists():
        data = json.loads(args.baseline.read_text(encoding="utf-8"))
        baseline_summary = data["summary"]
        baseline_results = [m.CaseResult(**r) for r in data["results"]]
        broken = m.regressions(results, baseline_results)

    text = rep.render(results, summary, tags, baseline_summary, broken)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
        print(f"Отчёт: {args.report}")
    else:
        print(text)

    if args.save_baseline:
        args.save_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.save_baseline.write_text(
            json.dumps(
                {"summary": summary, "results": [asdict(r) for r in results]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # Сломанные кейсы валят прогон независимо от средних цифр.
    # Именно так харнесс встаёт в CI: правка промпта, ломающая рабочий
    # сценарий, не должна доехать до прода только потому, что средняя
    # метрика осталась приличной.
    if broken:
        print(f"\nРЕГРЕССИЯ: сломано кейсов - {len(broken)}: {', '.join(broken)}")
        return 1

    if args.fail_under is not None and summary["pass@k"] < args.fail_under:
        print(f"\npass@k {summary['pass@k']:.2f} ниже порога {args.fail_under}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
