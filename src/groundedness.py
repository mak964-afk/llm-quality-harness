"""Детекция галлюцинаций через проверку опоры ответа на источники.

Идея простая и старая, как аудит: не верь выводу, проверь каждое утверждение
по первичному документу.

Ответ модели разбирается на атомарные утверждения, и каждое сверяется с теми
документами, которые RAG реально отдал агенту. Утверждение, которое не следует
ни из одного источника, - галлюцинация, даже если оно случайно верно.

Почему "даже если верно": модель, угадавшая факт из своих весов, угадала его
и в следующий раз - но там ошибётся. Мы ловим не ошибку, а МЕХАНИЗМ,
который её порождает.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, Sequence


class LLM(Protocol):
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str: ...


@dataclass
class Doc:
    id: str
    text: str


@dataclass
class Claim:
    text: str
    supported: bool
    source_id: str | None
    reason: str


@dataclass
class GroundednessResult:
    claims: list[Claim]

    @property
    def total(self) -> int:
        return len(self.claims)

    @property
    def grounded(self) -> int:
        return sum(1 for c in self.claims if c.supported)

    @property
    def score(self) -> float:
        return self.grounded / self.total if self.total else 0.0

    @property
    def hallucinations(self) -> list[Claim]:
        return [c for c in self.claims if not c.supported]


EXTRACT_SYSTEM = """Ты разбираешь текст на атомарные утверждения.

Атомарное утверждение - одна проверяемая мысль. "Возврат возможен в течение
14 дней, деньги приходят за 5 рабочих дней" - это ДВА утверждения, не одно.

Не включай: приветствия, вежливые обороты, предложения помощи, оговорки
вида "уточните у поддержки". Они ничего не утверждают о мире и проверке
не подлежат.

Верни JSON: {"claims": ["утверждение 1", "утверждение 2"]}"""


VERIFY_SYSTEM = """Ты проверяешь, следует ли утверждение из предоставленных источников.

Правила строгие:
- supported = true ТОЛЬКО если утверждение прямо следует из текста источника.
- Правдоподобность не считается опорой. Утверждение может быть верным
  в реальности, но если его нет в источниках - supported = false.
  Мы проверяем не истинность, а ОПОРУ: откуда агент это взял.
- Домысел, обобщение и "логично предположить" - это false.
- Если утверждение следует из источника частично или с искажением деталей
  (числа, сроки, условия) - false.

Верни JSON:
{"supported": true|false, "source_id": "id или null", "reason": "коротко"}"""


def extract_claims(llm: LLM, answer: str) -> list[str]:
    raw = llm.complete(EXTRACT_SYSTEM, answer)
    try:
        return json.loads(raw).get("claims", [])
    except json.JSONDecodeError:
        # Модель не отдала валидный JSON. Молча вернуть пусто нельзя:
        # пустой список утверждений даст groundedness = 0/0 и кейс тихо
        # выпадет из метрики. Лучше явный шум в логах, чем красивая ложь в отчёте.
        raise ValueError(f"Извлечение утверждений вернуло не-JSON: {raw[:200]}")


def verify_claim(llm: LLM, claim: str, docs: Sequence[Doc]) -> Claim:
    sources = "\n\n".join(f"[{d.id}]\n{d.text}" for d in docs)
    user = f"УТВЕРЖДЕНИЕ:\n{claim}\n\nИСТОЧНИКИ:\n{sources}"
    raw = llm.complete(VERIFY_SYSTEM, user)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Не смогли проверить - считаем НЕподтверждённым.
        # Дефолт всегда в сторону строгости: непроверенное утверждение
        # не должно улучшать метрику. Иначе харнесс будет врать в нашу пользу,
        # а это ровно та болезнь, которую он лечит.
        return Claim(claim, supported=False, source_id=None, reason="верификатор не ответил")

    return Claim(
        text=claim,
        supported=bool(data.get("supported")),
        source_id=data.get("source_id"),
        reason=data.get("reason", ""),
    )


def check(llm: LLM, answer: str, docs: Sequence[Doc]) -> GroundednessResult:
    """Полная проверка ответа на опору в источниках."""
    if not docs:
        # RAG не нашёл ничего, но агент всё равно ответил - каждое
        # утверждение по определению висит в воздухе.
        claims = extract_claims(llm, answer)
        return GroundednessResult(
            [Claim(c, False, None, "источники не найдены") for c in claims]
        )

    claims = extract_claims(llm, answer)
    return GroundednessResult([verify_claim(llm, c, docs) for c in claims])
