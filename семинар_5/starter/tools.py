"""
Пять инструментов макро-агента.

Твоя задача на семинаре — заполнить TODO в каждой функции.
Стратегия: сначала попробуй живой адрес ЦБ, при ошибке — запасной путь:
ближайшее значение из CSV в ./data/.

ОБЯЗАТЕЛЬНО: в каждом ответе должно быть поле `source` — либо "cbr_live",
либо "fallback_csv". Агент использует это, чтобы честно оговариваться
в итоговом ответе.
"""

from __future__ import annotations

import csv
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date as _date
from datetime import datetime
from pathlib import Path

import sympy  # noqa: F401  (пригодится в calculate)

DATA_DIR = Path(__file__).resolve().parent / "data"

CBR_FX_URL = "https://www.cbr.ru/scripts/XML_daily.asp"
CBR_KEYIND_URL = "https://www.cbr.ru/key-indicators/"

_TIMEOUT_SEC = 6
_UA = "Mozilla/5.0 (seminar-5-agent)"


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
        currency: ISO-код (USD, EUR, CNY, GBP, ...).
        on_date:  YYYY-MM-DD. None → сегодня.

    Returns:
        {"currency": "USD", "date": "2026-04-22", "rate": 82.5, "source": "cbr_live"}

    Подсказки:
    - адрес ЦБ: GET https://www.cbr.ru/scripts/XML_daily.asp?date_req=DD/MM/YYYY
    - Ответ — XML в кодировке windows-1251.
      Ищем тег <Valute> с <CharCode>==currency, берём <Value> (делим на <Nominal>).
    - При любой ошибке (таймаут, 403, нет валюты в ответе) — вызови _fx_fallback.
    - НЕ забудь поле "source" в возвращаемом dict.
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
        # валюту в ответе не нашли
        return _fx_fallback(currency, d, reason=f"Валюты {currency} нет в ответе ЦБ.")

    except (urllib.error.URLError, TimeoutError, ET.ParseError, ValueError) as e:
        return _fx_fallback(
            currency,
            d,
            reason=f"Сбой живого запроса по {currency}. {type(e).__name__}: {e}",
        )


def _fx_fallback(currency: str, d: _date, *, reason: str) -> dict:
    """Ближайшая по дате запись из fx_benchmark.csv."""
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
        return {"error": f"нет запасных данных для {currency}"}
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

_KEY_RATE_FALLBACK_RE = re.compile(
    r"Ключевая\s*ставка.{0,200}?(\d{1,2}[.,]\d{1,2})\s*%",
    re.S | re.I,
)


def get_key_rate(on_date: str | None = None) -> dict:
    """
    Ключевая ставка Банка России, действующая на указанную дату, % годовых.

    Returns:
        {"rate": 16.0, "date": "2026-04-22", "valid_from": "2026-03-20", "source": "cbr_live"}

    Подсказки:
    - Для текущей даты (on_date is None) — считай страницу https://www.cbr.ru/key-indicators/
      регуляркой вытащи число рядом со словами "Ключевая ставка" и знаком "%".
    - Для исторической даты — бери из data/key_rate_history.csv запись с
      максимальной valid_from ≤ on_date. CSV уже отсортирован по возрастанию.
    - Если живой запрос не получился — провались в CSV и поставь source="fallback_csv".
    """
    d = _parse_date(on_date)

    if on_date is None or d == _date.today():
        try:
            html = _http_get(CBR_KEYIND_URL).decode("utf-8", errors="ignore")
            m = _KEY_RATE_RE.search(html) or _KEY_RATE_FALLBACK_RE.search(html)
            if m:
                val = float(m.group(1).replace(",", "."))
                return {"rate": val, "date": d.isoformat(), "source": "cbr_live"}
        except (urllib.error.URLError, TimeoutError, ValueError):
            pass

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
    Индекс потребительских цен Росстата, % г/г, на конец месяца.

    Args:
        year: int,
        month: int

    Returns:
        {"year": 2024, "month": 3, "cpi_yoy": 7.72, "source": "rosstat_csv"}

    Подсказки:
    - Никакого живого API у Росстата нет; читай data/cpi_ru_monthly.csv.
    - Проверь month in 1..12; если нет данных — верни {"error": ...}.
    """
    year = int(year)
    month = int(month)
    if not (1 <= month <= 12):
        return {"error": f"month= {month} вне промежутка 1..12"}

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
# 3b. Безработица (Росстат, % от рабочей силы)
# ===========================================================================


def get_unemployment(year: int, month: int) -> dict:
    """
    Уровень безработицы (методология МОТ) Росстата, % от рабочей силы,
    на конец месяца.

    Returns:
        {"year": 2024, "month": 3, "unemployment": 2.7, "source": "rosstat_csv"}

    Подсказки:
    - Источник — data/unemployment_ru_monthly.csv (колонки year, month, unemployment).
    - Логика 1-в-1 как у get_inflation: проверь month in 1..12, найди строку,
      нет данных — верни {"error": ...}.
    """
    year = int(year)
    month = int(month)
    if not (1 <= month <= 12):
        return {"error": f"month= {month} вне промежутка 1..12"}

    path = DATA_DIR / "unemployment_ru_monthly.csv"
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["year"]) == year and int(row["month"]) == month:
                return {
                    "year": year,
                    "month": month,
                    "unemployment": float(row["unemployment"]),
                    "source": "rosstat_csv",
                }
    return {"error": f"нет данных по безработице на {year}-{month:02d}"}


# ===========================================================================
# 4. Калькулятор
# ===========================================================================


def calculate(expression: str) -> dict:
    """
    Безопасный математический калькулятор. Понимает +, -, *, /, ^, sqrt, ln,
    log, exp, скобки.

    Returns:
        {"expression": "(21 - 9.5)", "result": 11.5}
        {"expression": ..., "error": "..."}

    Подсказки:
    - sympy.sympify(expr) даёт Expression; float(...) достаёт число.
    - ОПАСНО: sympify может вычислять произвольные функции. Запрети любые
      идентификаторы, кроме белого списка: {log, ln, sqrt, exp, pi, e, sin, cos, tan, abs}.
    - Верни ошибку как {"expression": ..., "error": "..."}, не кидай исключение —
      агент должен увидеть ошибку и попробовать ещё раз.
    """
    if not isinstance(expression, str) or not expression.strip():
        return {"error": "пустое выражение"}

    try:
        val = float(sympy.sympify(expression.replace("^", "**")))
        return {"expression": expression, "result": round(val, 6)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ===========================================================================
# 5. Сравнение метрик между двумя периодами
# ===========================================================================


def _parse_period(period: str) -> tuple[str, int, int]:
    """YYYY-MM или YYYY-MM-DD -> (iso_date, year, month)."""
    period = period.strip()
    if len(period) == 7 and period[4] == "-":
        year, month = int(period[:4]), int(period[5:7])
        from calendar import monthrange

        last_day = monthrange(year, month)[1]
        return f"{year}-{month:02d}-{last_day:02d}", year, month
    d = _parse_date(period)
    return d.isoformat(), d.year, d.month


def _metric_value(metric: str, period: str) -> tuple[dict, float | None, str]:
    """Получить сырое obs и числовое value для метрики на период."""
    iso_date, year, month = _parse_period(period)
    m = metric.lower()

    if m == "key_rate":
        obs = get_key_rate(on_date=iso_date)
        return obs, obs.get("rate"), obs.get("source", "")
    if m == "fx_usd":
        obs = get_fx_rate(currency="USD", on_date=iso_date)
        return obs, obs.get("rate"), obs.get("source", "")
    if m == "fx_eur":
        obs = get_fx_rate(currency="EUR", on_date=iso_date)
        return obs, obs.get("rate"), obs.get("source", "")
    if m == "fx_cny":
        obs = get_fx_rate(currency="CNY", on_date=iso_date)
        return obs, obs.get("rate"), obs.get("source", "")
    if m == "cpi":
        obs = get_inflation(year, month)
        return obs, obs.get("cpi_yoy"), obs.get("source", "")
    if m == "unemployment":
        obs = get_unemployment(year, month)
        return obs, obs.get("unemployment"), obs.get("source", "")

    return {"error": f"неизвестная метрика: {metric}"}, None, ""


def compare_periods(metric: str, period_a: str, period_b: str) -> dict:
    """
    Сравнить значение метрики в двух периодах.
    metric: key_rate | fx_USD | fx_EUR | fx_CNY | cpi | unemployment
    period: YYYY-MM или YYYY-MM-DD
    """
    allowed = {"key_rate", "fx_usd", "fx_eur", "fx_cny", "cpi", "unemployment"}
    if metric.lower() not in allowed:
        return {"error": f"metric должен быть одним из {sorted(allowed)}"}

    obs_a, val_a, src_a = _metric_value(metric, period_a)
    if "error" in obs_a or val_a is None:
        return {"error": f"период A ({period_a}): {obs_a.get('error', 'нет значения')}"}

    obs_b, val_b, src_b = _metric_value(metric, period_b)
    if "error" in obs_b or val_b is None:
        return {"error": f"период B ({period_b}): {obs_b.get('error', 'нет значения')}"}

    a_date = obs_a.get("date") or f"{obs_a.get('year', '')}-{obs_a.get('month', ''):02d}"
    b_date = obs_b.get("date") or f"{obs_b.get('year', '')}-{obs_b.get('month', ''):02d}"
    source = src_a if src_a == src_b else f"{src_a}+{src_b}"

    return {
        "metric": metric,
        "a": {"date": a_date, "value": val_a},
        "b": {"date": b_date, "value": val_b},
        "delta": round(val_b - val_a, 6),
        "ratio": round(val_b / val_a, 6) if val_a else None,
        "source": source,
    }
