"""
Четыре инструмента макро-агента.

Стратегия «живое + fallback»
----------------------------
ЦБ иногда лежит, у студента иногда нет интернета, семинар не должен падать.
Поэтому каждый сетевой инструмент сначала пробует реальный эндпоинт
cbr.ru; если тот молчит дольше 6 секунд или отдаёт мусор — берём ближайшее
значение из локального CSV в ./data/.

В ответе tool всегда есть поле `source`:
  "cbr_live"     — данные прямо с ЦБ
  "fallback_csv" — взяты из локального архива (плюс причина в поле reason)

Это важно для агента: он видит, откуда число, и может честно оговориться
в итоговом ответе.
"""
from __future__ import annotations

import csv
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date as _date, datetime
from pathlib import Path

import sympy

DATA_DIR = Path(__file__).resolve().parent / "data"

CBR_FX_URL = "https://www.cbr.ru/scripts/XML_daily.asp"
CBR_KEYIND_URL = "https://www.cbr.ru/key-indicators/"

_TIMEOUT_SEC = 6
_UA = "Mozilla/5.0 (seminar-5-agent)"


# ---------------------------------------------------------------------------
# маленькие хелперы
# ---------------------------------------------------------------------------

def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_SEC) as r:
        return r.read()


def _parse_date(s: str | None) -> _date:
    if s is None:
        return _date.today()
    if isinstance(s, _date):
        return s
    return datetime.strptime(s, "%Y-%m-%d").date()


# ===========================================================================
# 1. Курс валюты ЦБ
# ===========================================================================

def get_fx_rate(currency: str = "USD", on_date: str | None = None) -> dict:
    """
    Официальный курс валюты к рублю (сколько рублей за 1 единицу валюты).

    Args:
        currency: ISO-код валюты (USD, EUR, CNY, GBP, ...).
        on_date:  дата в формате YYYY-MM-DD. None → сегодня.

    Returns:
        {"currency": "USD", "date": "2026-04-22", "rate": 82.5, "source": "cbr_live"}
    """
    d = _parse_date(on_date)
    currency = currency.upper()

    try:
        q = urllib.parse.urlencode({"date_req": d.strftime("%d/%m/%Y")})
        xml_bytes = _http_get(f"{CBR_FX_URL}?{q}")
        xml_text = xml_bytes.decode("windows-1251", errors="replace")
        root = ET.fromstring(xml_text)
        for val in root.findall("Valute"):
            if val.findtext("CharCode") == currency:
                nominal = int(val.findtext("Nominal") or 1)
                raw = (val.findtext("Value") or "").replace(",", ".")
                rate = float(raw) / nominal
                return {
                    "currency": currency,
                    "date": d.isoformat(),
                    "rate": round(rate, 4),
                    "source": "cbr_live",
                }
        # валюту не нашли в ответе
        return _fx_fallback(currency, d, reason=f"currency {currency} not in CBR response")
    except (urllib.error.URLError, TimeoutError, ET.ParseError, ValueError) as e:
        return _fx_fallback(currency, d, reason=f"live_failed: {type(e).__name__}: {e}")


def _fx_fallback(currency: str, d: _date, *, reason: str) -> dict:
    path = DATA_DIR / "fx_benchmark.csv"
    best = None
    best_delta = None
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["currency"] != currency:
                continue
            row_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
            delta = abs((row_date - d).days)
            if best is None or delta < best_delta:
                best = row
                best_delta = delta
    if best is None:
        return {"error": f"нет fallback-данных для {currency}"}
    return {
        "currency": currency,
        "date": best["date"],
        "rate": float(best["rate"]),
        "source": "fallback_csv",
        "reason": reason,
    }


# ===========================================================================
# 2. Ключевая ставка ЦБ
# ===========================================================================

_KEY_RATE_RE = re.compile(
    r"Ключевая\s*ставка[^<]*?</\w+>[^<]*?<[^>]*>\s*([\d]{1,2}[.,][\d]{1,2})\s*%",
    re.S | re.I,
)
# запасной паттерн — просто ищем число рядом со словом «ключевая ставка»
_KEY_RATE_FALLBACK_RE = re.compile(
    r"Ключевая\s*ставка.{0,200}?(\d{1,2}[.,]\d{1,2})\s*%",
    re.S | re.I,
)


def get_key_rate(on_date: str | None = None) -> dict:
    """
    Ключевая ставка ЦБ РФ, действовавшая на указанную дату.

    Args:
        on_date: YYYY-MM-DD. None → текущая ставка.

    Returns:
        {"rate": 21.0, "date": "2026-04-22", "valid_from": "2024-10-28", "source": "cbr_live"}
    """
    d = _parse_date(on_date)

    # Для «сегодня» пытаемся соскрабить главную — там всегда актуальное число.
    if on_date is None or d == _date.today():
        try:
            html = _http_get(CBR_KEYIND_URL).decode("utf-8", errors="ignore")
            m = _KEY_RATE_RE.search(html) or _KEY_RATE_FALLBACK_RE.search(html)
            if m:
                val = float(m.group(1).replace(",", "."))
                return {
                    "rate": val,
                    "date": d.isoformat(),
                    "source": "cbr_live",
                }
        except (urllib.error.URLError, TimeoutError, ValueError):
            pass  # провалимся в CSV ниже

    # Исторический путь — CSV с датами изменений.
    path = DATA_DIR / "key_rate_history.csv"
    chosen = None
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rd = datetime.strptime(row["valid_from"], "%Y-%m-%d").date()
            if rd <= d:
                chosen = row
            else:
                break
    if chosen is None:
        return {"error": f"нет исторической ставки на {d}"}
    return {
        "rate": float(chosen["rate"]),
        "date": d.isoformat(),
        "valid_from": chosen["valid_from"],
        "source": "fallback_csv",
    }


# ===========================================================================
# 3. Инфляция (ИПЦ г/г, Росстат)
# ===========================================================================

def get_inflation(year: int, month: int) -> dict:
    """
    Индекс потребительских цен, Росстат, % г/г, на конец указанного месяца.

    Args:
        year:  год (например 2024)
        month: месяц 1..12

    Returns:
        {"year": 2024, "month": 3, "cpi_yoy": 7.72, "source": "rosstat_csv"}
    """
    year = int(year)
    month = int(month)
    if not (1 <= month <= 12):
        return {"error": f"month={month} вне 1..12"}
    path = DATA_DIR / "cpi_ru_monthly.csv"
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["year"]) == year and int(row["month"]) == month:
                return {
                    "year": year,
                    "month": month,
                    "cpi_yoy": float(row["cpi_yoy"]),
                    "source": "rosstat_csv",
                }
    return {"error": f"нет данных ИПЦ на {year}-{month:02d}"}


# ===========================================================================
# 4. Калькулятор
# ===========================================================================

def calculate(expression: str) -> dict:
    """
    Безопасный калькулятор через sympy. Понимает арифметику, степени,
    sqrt, ln, log, exp. Не выполняет код — парсер.

    Args:
        expression: математическое выражение, например "(21 - 9.5) * 1.0" или
                    "log(2) / log(1 + 0.17/12)" (за сколько месяцев удвоится вклад).
    """
    if not isinstance(expression, str) or not expression.strip():
        return {"error": "пустое выражение"}
    # Отсечём подозрительные символы — имена могут содержать только буквы/цифры/_
    # (нужны для log, ln, sqrt, pi, exp). Запрещаем точку-доступ и скобки-вызовы
    # к чему угодно, кроме разрешённого множества.
    allowed = set("0123456789.+-*/(),% ^")
    letters = re.findall(r"[A-Za-zА-Яа-я_]+", expression)
    blacklist = set(letters) - {
        "log", "ln", "sqrt", "exp", "pi", "E", "e",
        "sin", "cos", "tan", "abs",
    }
    if blacklist:
        return {"error": f"недопустимые идентификаторы: {sorted(blacklist)}"}
    bad = set(expression) - allowed - set("".join(letters))
    if bad:
        return {"error": f"недопустимые символы: {sorted(bad)}"}
    try:
        val = float(sympy.sympify(expression.replace("^", "**")))
        return {"expression": expression, "result": round(val, 6)}
    except Exception as e:
        return {"expression": expression, "error": f"{type(e).__name__}: {e}"}
