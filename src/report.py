"""Отчёт о прогоне.

Отчёт пишется для двух читателей одновременно, и это осознанный компромисс:
инженеру нужны сломанные кейсы и разрез по тегам, руководителю - одна строка
"можно выпускать или нет". Поэтому вердикт стоит первым, детали ниже.
"""

from __future__ import annotations

from typing import Sequence

from .metrics import CaseResult

_LABELS = {
    "pass@k": "Верных ответов",
    "hit@5": "RAG нашёл нужный документ",
    "mrr": "Качество ранжирования (MRR)",
    "groundedness": "Опора на источники (анти-галлюцинации)",
    "citation_accuracy": "Ссылки ведут на реальные документы",
    "correct_refusal": "Честно молчит, когда не знает",
    "over_refusal": "Молчит, когда знает (плохо)",
    "latency_p95_ms": "Задержка p95, мс",
    "cost_per_case_usd": "Стоимость обращения, $",
    "cases": "Кейсов в наборе",
}

# Пороги - не истина, а стартовая точка для разговора с заказчиком.
# Для юридического или медицинского контура их поднимают, для внутреннего
# поиска по вики - опускают. Цифры обязаны обсуждаться, а не наследоваться молча.
_THRESHOLDS = {
    "pass@k": 0.85,
    "groundedness": 0.95,      # самый жёсткий: галлюцинация дороже незнания
    "citation_accuracy": 0.98, # выдуманная ссылка хуже отсутствия ссылки
    "correct_refusal": 0.90,
    "over_refusal": 0.10,      # чем МЕНЬШЕ, тем лучше
}

_LOWER_IS_BETTER = {"over_refusal", "latency_p95_ms", "cost_per_case_usd"}


def _fmt(key: str, value: float) -> str:
    if key in ("cases",):
        return str(int(value))
    if key == "latency_p95_ms":
        return f"{value:.0f}"
    if key == "cost_per_case_usd":
        return f"{value:.4f}"
    return f"{value:.1%}"


def _verdict(summary: dict[str, float]) -> tuple[str, list[str]]:
    problems = []
    for key, threshold in _THRESHOLDS.items():
        if key not in summary:
            continue
        value = summary[key]
        failed = value > threshold if key in _LOWER_IS_BETTER else value < threshold
        if failed:
            problems.append(
                f"{_LABELS.get(key, key)}: {_fmt(key, value)} "
                f"(порог {_fmt(key, threshold)})"
            )

    if not problems:
        return "ГОТОВ К ВЫПУСКУ", []
    return "НЕ ГОТОВ", problems


def render(
    results: Sequence[CaseResult],
    summary: dict[str, float],
    tags: dict[str, dict[str, float]],
    baseline: dict[str, float] | None = None,
    broken: Sequence[str] = (),
) -> str:
    verdict, problems = _verdict(summary)

    out: list[str] = ["# Отчёт о качестве агента", "", f"## Вердикт: {verdict}", ""]

    if problems:
        out.append("Не проходят пороги:")
        out.append("")
        for p in problems:
            out.append(f"- {p}")
        out.append("")

    if broken:
        out += [
            f"## РЕГРЕССИЯ: сломано кейсов - {len(broken)}",
            "",
            "Эти кейсы проходили в прошлом прогоне и не проходят сейчас. "
            "Средние метрики могли при этом не сдвинуться - смотрите сюда, а не на них.",
            "",
        ]
        for case_id in broken:
            out.append(f"- `{case_id}`")
        out.append("")

    out += ["## Метрики", ""]
    header = "| Метрика | Значение |" + (" Было | Дельта |" if baseline else "")
    sep = "|---|---|" + ("---|---|" if baseline else "")
    out += [header, sep]

    for key, value in summary.items():
        row = f"| {_LABELS.get(key, key)} | {_fmt(key, value)} |"
        if baseline and key in baseline:
            delta = value - baseline[key]
            good = delta < 0 if key in _LOWER_IS_BETTER else delta > 0
            sign = "+" if delta > 0 else ""
            mark = "" if abs(delta) < 1e-9 else (" ✅" if good else " ⚠️")
            row += f" {_fmt(key, baseline[key])} | {sign}{_fmt(key, delta)}{mark} |"
        out.append(row)

    if tags:
        out += [
            "",
            "## В разрезе тегов",
            "",
            "Общая цифра усредняет и потому лжёт: агент может держать 90% в целом "
            "и валить все критичные кейсы про деньги и сроки.",
            "",
            "| Тег | Верных | Опора на источники | Кейсов |",
            "|---|---|---|---|",
        ]
        for tag, s in tags.items():
            out.append(
                f"| {tag} | {_fmt('pass@k', s['pass@k'])} | "
                f"{_fmt('groundedness', s['groundedness'])} | {int(s['cases'])} |"
            )

    failed = [r for r in results if not r.passed]
    if failed:
        out += ["", f"## Провалившиеся кейсы ({len(failed)})", ""]
        for r in failed:
            reasons = []
            if r.total_claims and r.grounded_claims < r.total_claims:
                reasons.append(
                    f"галлюцинации: {r.total_claims - r.grounded_claims} из {r.total_claims} утверждений без опоры"
                )
            if r.expected_source and r.expected_source not in r.retrieved_ids:
                reasons.append("RAG не нашёл нужный документ")
            if r.cited and not r.citation_valid:
                reasons.append("ссылка на несуществующий источник")
            if r.should_refuse and not r.refused:
                reasons.append("ответил там, где обязан был признать незнание")
            out.append(f"- `{r.case_id}` - {'; '.join(reasons) or 'ответ не соответствует эталону'}")

    return "\n".join(out) + "\n"
