"""Build the prompt of one R5 round for the agent (LLM, read-only sandbox, answer = JSON).

The agent never sees data or labels: only the grammar, the current feature set with gain importance
and the journal of earlier rounds (each proposal with its measured gain on val half A).

  python scripts/r5_prompt.py --round 1 > logs/experiment/r5_round1_prompt.md
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tbrec.config import ART  # noqa: E402
from tbrec import features  # noqa: E402

N_PROPOSALS = 10

TEMPLATE = """# Задача: предложи {n} новых признаков для ранжирования

Ты — исследователь рекомендательных систем. Ничего не запускай и не читай файлы; ответ — только JSON.

## Задача ранжирования
Yandex Music, «Моя волна»: сервис сам ставит треки (алгоритмические показы). Для каждого показа
пользователя за день предсказываем Listen+ (дослушал ≥ 50% трека) и ранжируем показы внутри
пользователь-дня. Признаки считаются по истории строго до начала дня (снимок), события самого дня не
видны. Метрика — GAUC внутри пользователь-дня (константные для пользователя признаки сами по себе не
помогают, работают только во взаимодействии с признаками трека/пары). 55% показов — уже слушанные треки.

## Грамматика признаков
{doc}

Составные признаки из двух имён грамматики: `ratio(a,b)` = a / (b + 1e-6), `diff(a,b)` = a - b
(например доля прослушиваний пользователя, ушедшая этому артисту: `ratio(UA__listen__cnt__h30d,U__listen__cnt__h30d)`;
тренд: `diff(UI__lp__rate__h7d,UI__lp__rate__h90d)`). Сетка весов: затухание h6h…h365d, окна w6h…w365d, all;
допустимы и другие значения вида h<n>h|d, w<n>h|d.

## Текущий набор (автоматический отбор R1 по сетке затухания), важность по gain
{current}

## Уже проверено (журнал прошлых раундов: Δ GAUC на val-A против текущего набора, порог принятия {thr})
{journal}

## Что нужно
{n} предложений, которых нет в текущем наборе и журнале. Думай о механизмах, которых набору не хватает
(насыщение и усталость от трека/артиста, тренды, доли, отношение к органике, свежесть, разнообразие).
Ответ — только JSON-массив без пояснений вокруг:
[{{"name": "<имя>", "why": "<одна фраза: какой механизм>"}}, ...]
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    sel = json.loads((ART / "r1_decay_selected.json").read_text())
    imp = dict((n, g) for n, g in sel["importance_top50"])
    state = json.loads((ART / "r5_state.json").read_text()) if (ART / "r5_state.json").exists() else {"accepted": [], "journal": [], "rounds": []}
    names = sel["names"] + state["accepted"]
    current = "\n".join(f"- {n}" + (f" (gain {imp[n]:.0f})" if n in imp else "") for n in names)
    jr = [e for e in state["journal"]]
    journal = "\n".join(f"- r{e['round']} {e['name']}: " + (f"Δ {e['delta']:+.4f}{' ПРИНЯТ' if e.get('accepted') else ''}" if "delta" in e else f"отклонён ({e['error']})")
                        for e in jr) or "- (пусто)"
    thr = f"+{state['rounds'][-1]['threshold']:.4f}" if state.get("rounds") else "+0.0005"
    print(TEMPLATE.format(n=N_PROPOSALS, doc="\n\n".join(features.__doc__.split("\n\n")[1:3]), current=current, journal=journal, thr=thr))


if __name__ == "__main__":
    main()
