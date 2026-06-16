"""
JSON-schema описания инструментов для OpenAI Tool Use API.

На семинаре мы пишем их руками, чтобы увидеть каждое поле. В бою такой
файл генерируется автоматически — из pydantic-моделей (или из type hints
через `openai.pydantic_function_tool`). Руками пишут только тогда, когда
надо точно контролировать описание для LLM.
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_fx_rate",
            "description": (
                "Официальный курс валюты к рублю (сколько рублей за 1 единицу) "
                "на указанную дату по данным ЦБ РФ. Нельзя придумывать курс — "
                "всегда зови этот инструмент, если вопрос про USD/EUR/CNY/прочие."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "currency": {
                        "type": "string",
                        "description": "ISO-код валюты: USD, EUR, CNY, GBP, JPY, TRY и т.д.",
                    },
                    "on_date": {
                        "type": ["string", "null"],
                        "description": "Дата YYYY-MM-DD. Если не задана — берётся сегодняшняя.",
                    },
                },
                "required": ["currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_key_rate",
            "description": (
                "Ключевая ставка Банка России на дату, в процентах годовых. "
                "Для текущей ставки обращается к cbr.ru; для исторической — "
                "к локальному архиву изменений ставки."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "on_date": {
                        "type": ["string", "null"],
                        "description": "Дата YYYY-MM-DD. None = сегодня.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_inflation",
            "description": (
                "Индекс потребительских цен Росстата, % г/г, на конец указанного месяца. "
                "Используется для оценки инфляции и реальной доходности."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer", "description": "Год, например 2024"},
                    "month": {
                        "type": "integer",
                        "description": "Месяц 1..12 (1 = январь)",
                        "minimum": 1,
                        "maximum": 12,
                    },
                },
                "required": ["year", "month"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Безопасный математический калькулятор. Понимает +, -, *, /, ^, "
                "sqrt, ln, log, exp, скобки. Использовать для любых вычислений "
                "над числами, полученными от других инструментов — руками не считать."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "Математическое выражение, например '(21 - 9.5)' или "
                            "'log(2) / log(1 + 0.17)' (во сколько лет удвоится вклад "
                            "при ставке 17%)."
                        ),
                    },
                },
                "required": ["expression"],
            },
        },
    },
]
