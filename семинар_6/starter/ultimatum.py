"""
Блок 7 — «Ультиматум» между двумя LLM-агентами.

Агенты не декомпозируют задачу, а ВЗАИМОДЕЙСТВУЮТ. Предлагающий делит 100 ₽,
Отвечающий принимает или отвергает (при отказе оба получают ноль). Прогоняем N
раундов, считаем долю принятия по размеру предложения.

На семинаре нужно закрыть 2 TODO — оживить обоих игроков.

Запуск:
    python ultimatum.py            # 10 раундов
    python ultimatum.py 20         # 20 раундов
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import get_model, make_client


class ProposerMove(BaseModel):
    share_to_responder: int = Field(
        ..., ge=0, le=100, description="Сколько из 100 ₽ предложить второму игроку."
    )
    reasoning: str = Field(default="", description="Одна фраза, почему столько.")


class ResponderMove(BaseModel):
    accept: bool = Field(..., description="Принять предложенный делёж или отвергнуть.")
    reasoning: str = Field(default="", description="Одна фраза, почему.")


PROPOSER_SYSTEM = """\
Ты — Предлагающий в игре «Ультиматум». У тебя 100 ₽. Ты предлагаешь, сколько
отдать второму игроку (Отвечающему); остальное берёшь себе. Если он откажется —
оба получите НОЛЬ. Цель — забрать как можно больше, но так, чтобы предложение
приняли. Верни целое число от 0 до 100 — сколько отдаёшь Отвечающему."""

RESPONDER_SYSTEM = """\
Ты — Отвечающий в игре «Ультиматум». Тебе предлагают часть от 100 ₽.
Прими — получишь предложенное; откажись — оба получите НОЛЬ. Ты живой человек,
для которого важна справедливость: явно несправедливые предложения отвергаешь
из принципа, даже теряя деньги; справедливые — принимаешь."""


def propose(round_num: int, history: list[dict]) -> ProposerMove:
    client = make_client()
    past = (
        "\n".join(
            f"  раунд {h['round']}: предложил {h['share']} → {'принято' if h['accept'] else 'отказ'}"
            for h in history
        )
        or "  (первый раунд)"
    )

    # TODO (блок 7.1): вызови make_client().chat.completions.create(...) с
    #   response_model=ProposerMove, system=PROPOSER_SYSTEM, user=текст с раундом
    #   и историей `past`, temperature=0.7. Верни результат.
    #   Сейчас заглушка — всегда поровну:
    return client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": PROPOSER_SYSTEM},
            {
                "role": "user",
                "content": f"Раунд {round_num}. История:\n{past}\n Сколько отдашь Отвечающему?",
            },
        ],
        response_model=ProposerMove,
        temperature=0.7,
        max_retries=2,
    )


def respond(share: int) -> ResponderMove:
    # TODO (блок 7.2): вызови make_client().chat.completions.create(...) с
    #   response_model=ResponderMove, system=RESPONDER_SYSTEM, user=«предлагают {share}
    #   из 100, принять?», temperature=0.7. Верни результат.
    #   Сейчас заглушка — соглашается на всё (нет нормы справедливости):
    client = make_client()
    return client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": RESPONDER_SYSTEM},
            {
                "role": "user",
                "content": f"Тебе предлагают {share} ₽ из 100. (себе оставляют {100 - share}). Принять?",
            },
        ],
        response_model=ResponderMove,
        temperature=0.7,
        max_retries=2,
    )


def play(n_rounds: int = 10, verbose: bool = True) -> list[dict]:
    """Прогнать N раундов: Предлагающий ↔ Отвечающий. Вернуть список раундов."""
    history: list[dict] = []
    for r in range(1, n_rounds + 1):
        offer = propose(r, history)
        decision = respond(offer.share_to_responder)
        row = {"round": r, "share": offer.share_to_responder, "accept": decision.accept}
        history.append(row)
        if verbose:
            mark = "принято" if decision.accept else "ОТКАЗ"
            print(f"  раунд {r:>2}: предложил {offer.share_to_responder:>3} ₽ → {mark}")
    return history


def summary(rounds: list[dict]) -> None:
    """Доля принятия по корзинам предложения — та же ось, что в семинаре 7."""
    buckets: dict[str, list[bool]] = defaultdict(list)
    for h in rounds:
        bucket = f"{(h['share'] // 10) * 10}–{(h['share'] // 10) * 10 + 9}"
        buckets[bucket].append(h["accept"])
    print("\n  предложено, ₽ | раундов | принято")
    for b in sorted(buckets):
        v = buckets[b]
        print(f"  {b:>8} | {len(v):>7} | {sum(v) / len(v):.0%}")
    total = sum(h["accept"] for h in rounds)
    print(f"  Всего принято: {total}/{len(rounds)}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"«Ультиматум» LLM↔LLM: {n} раундов\n")
    rounds = play(n)
    summary(rounds)
