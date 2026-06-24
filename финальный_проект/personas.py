"""
Синтетические персоны (homo silicus): мини фокус-группа по NovaPay.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_client import get_model, make_client
from prompts import PERSONA_SYSTEM
from schema import FocusGroup

BASE = Path(__file__).resolve().parent


def run_focus_group(topic: str = "мобильное приложение NovaPay", out: Path = None) -> FocusGroup:
    out = out or BASE / "output" / "focus_group.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    client = make_client()
    fg: FocusGroup = client.chat.completions.create(
        model=get_model(),
        response_model=FocusGroup,
        max_retries=2,
        temperature=0.7,
        messages=[
            {"role": "system", "content": PERSONA_SYSTEM},
            {"role": "user", "content": f"Тема обсуждения: {topic}. Верни 5 персон с разными сегментами."},
        ],
    )
    out.write_text(fg.model_dump_json(indent=2), encoding="utf-8")
    print(f"Фокус-группа: {len(fg.opinions)} персон -> {out}")
    return fg
