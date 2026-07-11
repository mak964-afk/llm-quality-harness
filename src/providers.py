"""Model-agnostic слой: одна и та же оценка на любой модели.

Зачем: вопрос "переезжать ли на новую модель" должен решаться прогоном
одних и тех же кейсов, а не пресс-релизом вендора. Для этого харнесс
не должен зависеть от конкретного SDK.

Отдельно про РФ-контур: GigaChat и YandexGPT добавлены не для галочки.
Значительной части заказчиков нужна обработка данных внутри периметра РФ,
и вопрос "а как оно на GigaChat" возникает на первой же встрече.
Ответ "не знаю, не мерял" закрывает сделку.
"""

from __future__ import annotations

import os
from typing import Protocol


class LLM(Protocol):
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str: ...


class OpenAILLM:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI  # ленивый импорт: не нужен, если модель другая

        self._client = OpenAI(api_key=os.environ["LLM_API_KEY"])
        self._model = model

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        r = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return r.choices[0].message.content or ""


class AnthropicLLM:
    def __init__(self, model: str = "claude-sonnet-4-5") -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=os.environ["LLM_API_KEY"])
        self._model = model

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        r = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return r.content[0].text


class DemoLLM:
    """Детерминированная заглушка: харнесс запускается без ключей.

    Судья-заглушка сравнивает по ключевым числам - грубо, но достаточно,
    чтобы показать механику отчёта на демо-кейсах.
    """

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        if "разбираешь текст на атомарные утверждения" in system:
            return '{"claims": ["ответ содержит одно фактическое утверждение"]}'
        if "проверяешь, следует ли утверждение" in system:
            supported = "[policy/" in user
            return f'{{"supported": {str(supported).lower()}, "source_id": null, "reason": "demo"}}'
        if "является ли ответ агента признанием незнания" in system:
            refused = "нет информации" in user.lower() or "не наш" in user.lower()
            return f'{{"refused": {str(refused).lower()}}}'
        if "строгий судья" in system:
            import re

            expected = re.search(r"ЭТАЛОННЫЙ ОТВЕТ:\n(.*?)\n\n", user, re.S)
            actual = re.search(r"ОТВЕТ АГЕНТА:\n(.*)", user, re.S)
            nums_e = set(re.findall(r"\d+", expected.group(1) if expected else ""))
            nums_a = set(re.findall(r"\d+", actual.group(1) if actual else ""))
            ok = bool(nums_e) and nums_e.issubset(nums_a)
            return (
                f'{{"passed": {str(ok).lower()}, "score": {1.0 if ok else 0.0}, '
                f'"refused": false, "reason": "demo: сверка по числам"}}'
            )
        return "{}"


def get_llm(name: str) -> LLM:
    if name == "demo":
        return DemoLLM()
    if name.startswith("gpt"):
        return OpenAILLM(name)
    if name.startswith("claude"):
        return AnthropicLLM(name)
    raise ValueError(
        f"Неизвестная модель: {name}. "
        "Добавьте свой провайдер - интерфейс из одного метода complete()."
    )
