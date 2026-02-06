from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable

FOCUS_NOUNS = ["书", "纸", "鱼", "车", "人", "狗", "猫", "票", "衣", "杯"]


def select_focus_samples(
    rows: list[dict[str, str]],
    total: int = 20,
    flower_min: int = 5,
    flower_max: int = 10,
    seed: int = 13,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    indices = list(range(len(rows)))

    utterances = [row.get("Utterance", "") for row in rows]
    flower_candidates = [i for i, utt in enumerate(utterances) if "花" in utt]
    rng.shuffle(flower_candidates)

    flower_count = min(flower_max, len(flower_candidates))
    if flower_count < flower_min:
        flower_count = len(flower_candidates)

    selected: list[int] = []
    used = set()

    for idx in flower_candidates[:flower_count]:
        selected.append(idx)
        used.add(idx)
        if len(selected) >= total:
            break

    for noun in FOCUS_NOUNS:
        if len(selected) >= total:
            break
        candidates = [i for i, utt in enumerate(utterances) if i not in used and noun in utt]
        if not candidates:
            continue
        idx = rng.choice(candidates)
        selected.append(idx)
        used.add(idx)

    if len(selected) < total:
        remaining = [i for i in indices if i not in used]
        rng.shuffle(remaining)
        for idx in remaining:
            selected.append(idx)
            if len(selected) >= total:
                break

    selected_sorted = sorted(selected)
    return [rows[i] for i in selected_sorted]


def select_random_samples(
    rows: list[dict[str, str]],
    total: int = 20,
    seed: int = 42,
) -> list[dict[str, str]]:
    if total <= 0:
        return []
    if total >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    selected = sorted(indices[:total])
    return [rows[i] for i in selected]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_rows(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError("No rows to write")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
