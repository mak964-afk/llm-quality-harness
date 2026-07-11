"""Интерфейс подключения вашего агента к харнессу.

Харнесс намеренно НИЧЕГО не знает о вашей системе - ни про фреймворк,
ни про базу знаний, ни про модель. Он знает ровно два метода.

Это принципиально. Инструмент оценки, привязанный к конкретному стеку,
умирает вместе с ним, а стек в этой области меняется каждые полгода.
Абстракция здесь - не академическая красота, а срок жизни инструмента.

Чтобы подключить свою систему, реализуйте Agent:

    class MyAgent:
        def answer(self, question: str) -> Answer:
            docs = my_vector_db.search(question, top_k=5)
            text = my_llm.generate(question, docs)
            return Answer(
                text=text,
                docs=[Doc(id=d.path, text=d.content) for d in docs],
                citations=extract_citations(text),
                cost_usd=...,
            )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .groundedness import Doc

__all__ = ["Doc", "Answer", "Agent", "DemoAgent"]


@dataclass
class Answer:
    """Ответ агента вместе со всем, что нужно для его проверки."""

    text: str
    docs: list[Doc] = field(default_factory=list)
    """Документы, которые RAG реально отдал модели в контекст.

    Это НЕ вся база знаний и НЕ то, что "должно было" найтись.
    Именно то, что агент видел в момент ответа. Иначе groundedness
    проверяет не ту систему, которая работает в проде.
    """

    citations: list[str] = field(default_factory=list)
    """ID источников, на которые агент сослался в тексте ответа.

    Проверяется на валидность: ссылка на документ, которого нет в docs,
    - выдуманная ссылка. Такое встречается чаще, чем кажется.
    """

    cost_usd: float = 0.0


class Agent(Protocol):
    def answer(self, question: str) -> Answer: ...


class DemoAgent:
    """Заглушка для демо-прогона без ключей и без вашей инфраструктуры.

    Нужна ровно для одного: чтобы `python -m src.harness` запустился
    сразу после клонирования и человек увидел, как выглядит отчёт.
    В реальной оценке замените на свой Agent.
    """

    _KB = {
        "policy/returns.md#sroki": "Возврат товара надлежащего качества возможен в течение 14 дней с момента получения.",
        "policy/returns.md#money": "Возврат денежных средств производится в течение 5 рабочих дней после приёмки товара.",
        "policy/delivery.md#srok": "Стандартная доставка по Москве занимает 1-2 рабочих дня.",
    }

    def answer(self, question: str) -> Answer:
        q = question.lower()

        if "возврат" in q and ("дн" in q or "срок" in q):
            doc_id = "policy/returns.md#sroki"
            return Answer(
                text="На возврат товара надлежащего качества даётся 14 дней с момента получения [policy/returns.md#sroki].",
                docs=[Doc(doc_id, self._KB[doc_id])],
                citations=[doc_id],
                cost_usd=0.0012,
            )

        if "деньг" in q or "вернут" in q:
            doc_id = "policy/returns.md#money"
            return Answer(
                text="Деньги возвращаются в течение 5 рабочих дней после приёмки товара [policy/returns.md#money].",
                docs=[Doc(doc_id, self._KB[doc_id])],
                citations=[doc_id],
                cost_usd=0.0011,
            )

        # Вопроса нет в базе знаний. Правильное поведение - признать это,
        # а не сочинять правдоподобный ответ. Негативные кейсы проверяют
        # именно эту ветку, и в реальных агентах она ломается чаще всего.
        return Answer(
            text="В доступных документах нет информации по этому вопросу.",
            docs=[],
            citations=[],
            cost_usd=0.0004,
        )
