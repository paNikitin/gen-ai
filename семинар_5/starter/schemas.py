"""
JSON-схемы инструментов для API вызова инструментов (OpenAI-совместимого).

Эту запись модель читает, чтобы решить, какой инструмент звать и с какими
аргументами. Чем точнее описание — тем реже агент ошибается.

На семинаре дописываем эти схемы руками (в бою их генерируют из Pydantic
и аннотаций типов, но сначала полезно понять, что туда попадает).
"""

TOOL_SCHEMAS = [
    # ----- пример схемы (готовый, для ориентира) -----
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
                            "'log(2) / log(1 + 0.17)'."
                        ),
                    },
                },
                "required": ["expression"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_fx_rate",
            "description": (
                "Официальный курс валюты к рублю на дату по данным ЦБ РФ. "
                "Зови, если вопрос про курс USD/EUR/CNY/прочих — не придумывай курс."
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
                        "description": "Дата YYYY-MM-DD. Если не задана — сегодня.",
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
                "Ключевая ставка Банка России на дату, % годовых. Для текущей — "
                "с cbr.ru, для исторической — из локального архива изменений ставки."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "on_date": {
                        "type": ["string", "null"],
                        "description": "Дата YYYY-MM-DD. Если не задана — сегодня.",
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
                "Индекс потребительских цен Росстата, % г/г, на конец месяца. "
                "Для инфляции и реальной доходности."
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
            "name": "get_unemployment",
            "description": (
                "Уровень безработицы (МОТ) Росстата, % от рабочей силы, на конец "
                "месяца. Для «индекса нищеты» (инфляция + безработица)."
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
]
