"""Метрики качества LLM-агента и RAG.

Философия модуля: одна цифра ничего не значит. pass@k без groundedness
говорит только о том, что агент угадал - но не о том, что он не выдумал.
Поэтому качество меряется парой: "верно ли" и "откуда взял".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable, Sequence


@dataclass
class CaseResult:
    """Результат прогона одного кейса."""

    case_id: str
    passed: bool                      # вердикт судьи: ответ соответствует эталону
    retrieved_ids: list[str] = field(default_factory=list)   # что нашёл RAG, по порядку
    expected_source: str | None = None
    grounded_claims: int = 0          # утверждений, подтверждённых источниками
    total_claims: int = 0             # всего утверждений в ответе
    cited: bool = False               # ответ содержит ссылку на источник
    citation_valid: bool = False      # ссылка ведёт на реально найденный документ
    refused: bool = False             # агент признал незнание
    should_refuse: bool = False       # по кейсу он был обязан это сделать
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    tags: list[str] = field(default_factory=list)


def pass_at_k(results: Sequence[CaseResult]) -> float:
    """Доля кейсов с верным ответом.

    Базовая метрика. Отвечает на вопрос "можно ли вообще выпускать",
    но НЕ отвечает на вопрос "не врёт ли он" - для этого groundedness.
    """
    if not results:
        return 0.0
    return mean(1.0 if r.passed else 0.0 for r in results)


def hit_at_k(results: Sequence[CaseResult], k: int = 5) -> float:
    """Доля кейсов, где нужный документ попал в топ-k выдачи RAG.

    Зачем отдельно от pass@k: разделяет два разных провала.
    Низкий Hit@k -> виноват поиск (чанкинг, эмбеддинги, база неполна).
    Высокий Hit@k при низком pass@k -> документ нашли, но модель не смогла
    им воспользоваться (виноват промпт или модель).
    Без этого разделения команда неделями крутит промпт там, где сломан индекс.
    """
    scored = [r for r in results if r.expected_source]
    if not scored:
        return 0.0
    return mean(
        1.0 if r.expected_source in r.retrieved_ids[:k] else 0.0 for r in scored
    )


def mrr(results: Sequence[CaseResult]) -> float:
    """Mean Reciprocal Rank - насколько высоко нужный документ в выдаче.

    Hit@5 не отличает "первое место" от "пятого". MRR отличает.
    Это важно, потому что в контекст модели влезает не всё: документ на
    пятой позиции может быть обрезан, и агент его не увидит.
    """
    scored = [r for r in results if r.expected_source]
    if not scored:
        return 0.0

    ranks = []
    for r in scored:
        if r.expected_source in r.retrieved_ids:
            ranks.append(1.0 / (r.retrieved_ids.index(r.expected_source) + 1))
        else:
            ranks.append(0.0)
    return mean(ranks)


def groundedness(results: Sequence[CaseResult]) -> float:
    """Доля утверждений в ответах, подтверждённых найденными источниками.

    ЭТО ПРЯМАЯ МЕРА ГАЛЛЮЦИНАЦИЙ, и главная метрика всего харнесса.

    Агент может дать верный ответ, выдумав его из головы - pass@k этого
    не поймает. Но выдумавший однажды выдумает снова, и в следующий раз
    неверно. Groundedness ловит проблему ДО того, как она стоила денег.
    """
    scored = [r for r in results if r.total_claims > 0]
    if not scored:
        return 0.0
    return mean(r.grounded_claims / r.total_claims for r in scored)


def citation_accuracy(results: Sequence[CaseResult]) -> float:
    """Доля ответов, где ссылка на источник ведёт на реально найденный документ.

    Модель умеет выдумывать не только факты, но и ссылки на них.
    Ответ со ссылкой на несуществующий документ опаснее ответа без ссылки:
    он выглядит проверяемым и потому вызывает ложное доверие.
    """
    cited = [r for r in results if r.cited]
    if not cited:
        return 0.0
    return mean(1.0 if r.citation_valid else 0.0 for r in cited)


def refusal_rate(results: Sequence[CaseResult]) -> float:
    """Доля кейсов, где агент честно признал незнание."""
    if not results:
        return 0.0
    return mean(1.0 if r.refused else 0.0 for r in results)


def correct_refusal_rate(results: Sequence[CaseResult]) -> float:
    """Доля НЕГАТИВНЫХ кейсов, где агент правильно отказался отвечать.

    Набор кейсов, где агент обязан только отвечать, не проверяет ничего.
    Половина ценности харнесса - в проверке, что он молчит там, где не знает.
    Уверенный неверный ответ на вопрос вне базы знаний - худший из отказов
    системы, потому что пользователь ему поверит.
    """
    negatives = [r for r in results if r.should_refuse]
    if not negatives:
        return 0.0
    return mean(1.0 if r.refused else 0.0 for r in negatives)


def over_refusal_rate(results: Sequence[CaseResult]) -> float:
    """Доля кейсов, где агент отказался, хотя ответ был в базе.

    Обратная сторона: перекрученные ограничители превращают агента
    в бесполезного труса. Меряем оба края, а не один.
    """
    positives = [r for r in results if not r.should_refuse]
    if not positives:
        return 0.0
    return mean(1.0 if r.refused else 0.0 for r in positives)


def latency_p95(results: Sequence[CaseResult]) -> float:
    """95-й перцентиль задержки, мс. Среднее прячет хвост, а живёт пользователь в хвосте."""
    if not results:
        return 0.0
    values = sorted(r.latency_ms for r in results)
    idx = min(int(len(values) * 0.95), len(values) - 1)
    return values[idx]


def cost_per_case(results: Sequence[CaseResult]) -> float:
    """Средняя стоимость одного обращения. Экономика внедрения решается здесь."""
    if not results:
        return 0.0
    return mean(r.cost_usd for r in results)


def summarize(results: Sequence[CaseResult], k: int = 5) -> dict[str, float]:
    """Полная сводка. Именно этот словарь ложится в отчёт и в сравнение с baseline."""
    return {
        "pass@k": pass_at_k(results),
        f"hit@{k}": hit_at_k(results, k),
        "mrr": mrr(results),
        "groundedness": groundedness(results),
        "citation_accuracy": citation_accuracy(results),
        "correct_refusal": correct_refusal_rate(results),
        "over_refusal": over_refusal_rate(results),
        "latency_p95_ms": latency_p95(results),
        "cost_per_case_usd": cost_per_case(results),
        "cases": float(len(results)),
    }


def by_tag(results: Sequence[CaseResult], k: int = 5) -> dict[str, dict[str, float]]:
    """Сводка в разрезе тегов.

    Общая цифра усредняет и потому лжёт: агент может держать 90% pass@k
    в целом и при этом валить ВСЕ критичные кейсы про деньги и сроки.
    Разрез по тегам показывает, где именно он ломается.
    """
    tags = {t for r in results for t in r.tags}
    return {
        tag: summarize([r for r in results if tag in r.tags], k) for tag in sorted(tags)
    }


def compare(current: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    """Дельта между прогонами. Ответ на вопрос "стало лучше или хуже"."""
    return {
        key: current.get(key, 0.0) - baseline.get(key, 0.0)
        for key in set(current) | set(baseline)
    }


def regressions(
    current: Iterable[CaseResult], baseline: Iterable[CaseResult]
) -> list[str]:
    """Кейсы, которые проходили раньше и сломались сейчас.

    Самая полезная строка отчёта. Сводные метрики могут почти не сдвинуться,
    а пять критичных кейсов - молча упасть. Правка промпта, которая чинит
    один сценарий и ломает три, выглядит в средних цифрах как улучшение.
    """
    was = {r.case_id: r.passed for r in baseline}
    return sorted(
        r.case_id for r in current if was.get(r.case_id) and not r.passed
    )
