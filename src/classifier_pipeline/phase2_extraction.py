from __future__ import annotations

import csv
import random
from typing import Iterable, Optional, Sequence

import pymysql

from classifier_pipeline.childes_db import connect_childes_db

OUTPUT_HEADERS = [
    "File Name",
    "Collection_Type",
    "Speaker_Code",
    "Speaker_Role",
    "Age",
    "Utterance",
    "%gra",
    "Determiner/Numbers",
    "Classifier",
    "Classifier type",
    "Over use of Ge...",
]

REJECTED_HEADERS = [
    "File Name",
    "Collection_Type",
    "Speaker_Code",
    "Speaker_Role",
    "Age",
    "Utterance",
    "%gra",
    "Determiner/Numbers",
    "Determiner_POS",
    "Classifier",
]

FULL_CLASSIFIERS = [
    "个",
    "条",
    "只",
    "本",
    "张",
    "位",
    "头",
    "件",
    "年",
    "次",
    "天",
    "元",
    "下",
    "块",
    "岁",
    "名",
    "句",
    "家",
    "片",
    "份",
    "分",
    "部",
    "场",
    "把",
    "根",
    "颗",
    "辆",
    "种",
    "笔",
    "群",
    "对",
    "架",
    "组",
    "碗",
]

DEFAULT_CLASSIFIERS = FULL_CLASSIFIERS
DEFAULT_INCLUDE_LANGS = ("zho",)
DEFAULT_EXCLUDE_LANGS = ("yue", "nan")
DEFAULT_COLLECTIONS = ("Chinese",)


def is_number_or_determiner(part_of_speech: Optional[str]) -> bool:
    if not part_of_speech:
        return False
    if part_of_speech.startswith("num"):
        return True
    return part_of_speech in {"det", "pro:dem"}


def build_mandarin_language_clause(
    column: str,
    include: Sequence[str] = DEFAULT_INCLUDE_LANGS,
    exclude: Sequence[str] = DEFAULT_EXCLUDE_LANGS,
) -> tuple[str, list[str]]:
    params: list[str] = []
    include_clause = " OR ".join(f"{column} LIKE %s" for _ in include)
    params.extend([f"%{lang}%" for lang in include])

    exclude_clause = " AND ".join(f"{column} NOT LIKE %s" for _ in exclude)
    params.extend([f"%{lang}%" for lang in exclude])

    if include_clause and exclude_clause:
        clause = f"({include_clause}) AND {exclude_clause}"
    elif include_clause:
        clause = f"({include_clause})"
    elif exclude_clause:
        clause = exclude_clause
    else:
        clause = "1=1"

    return clause, params


def build_collection_clause(column: str, collections: Sequence[str]) -> tuple[str, list[str]]:
    if not collections:
        return "1=1", []
    placeholders = ", ".join(["%s"] * len(collections))
    return f"{column} IN ({placeholders})", list(collections)


def build_output_row(record: dict[str, object]) -> dict[str, object]:
    return {
        "File Name": record.get("file_name"),
        "Collection_Type": record.get("collection_type"),
        "Speaker_Code": record.get("speaker_code"),
        "Speaker_Role": record.get("speaker_role"),
        "Age": record.get("age"),
        "Utterance": record.get("utterance"),
        "%gra": record.get("gra"),
        "Determiner/Numbers": record.get("determiner"),
        "Classifier": record.get("classifier"),
        "Classifier type": "",
        "Over use of Ge...": "",
    }


def build_rejected_row(record: dict[str, object]) -> dict[str, object]:
    return {
        "File Name": record.get("file_name"),
        "Collection_Type": record.get("collection_type"),
        "Speaker_Code": record.get("speaker_code"),
        "Speaker_Role": record.get("speaker_role"),
        "Age": record.get("age"),
        "Utterance": record.get("utterance"),
        "%gra": record.get("gra"),
        "Determiner/Numbers": record.get("determiner"),
        "Determiner_POS": record.get("determiner_pos"),
        "Classifier": record.get("classifier"),
    }


def write_phase2_csv(
    output_path: str,
    classifiers: Iterable[str] = DEFAULT_CLASSIFIERS,
    include_langs: Sequence[str] = DEFAULT_INCLUDE_LANGS,
    exclude_langs: Sequence[str] = DEFAULT_EXCLUDE_LANGS,
    include_collections: Sequence[str] = DEFAULT_COLLECTIONS,
    rejected_output_path: Optional[str] = None,
    rejected_sample_size: int = 50,
    rejected_seed: int = 13,
    db_name: Optional[str] = None,
) -> int:
    language_clause, language_params = build_mandarin_language_clause(
        "u.language", include_langs, exclude_langs
    )
    collection_clause, collection_params = build_collection_clause(
        "t.collection_name", include_collections
    )

    classifier_list = list(classifiers)
    if not classifier_list:
        raise ValueError("At least one classifier must be provided")

    placeholders = ", ".join(["%s"] * len(classifier_list))

    where_clause = f"{language_clause} AND {collection_clause}"
    params = language_params + collection_params + classifier_list

    query = f"""
        SELECT
            t.filename AS file_name,
            t.collection_name AS collection_type,
            u.speaker_code AS speaker_code,
            u.speaker_role AS speaker_role,
            u.target_child_age AS age,
            u.gloss AS utterance,
            u.part_of_speech AS gra,
            p.gloss AS determiner,
            p.part_of_speech AS determiner_pos,
            c.gloss AS classifier
        FROM token c
        JOIN token p
            ON p.utterance_id = c.utterance_id
           AND p.token_order = c.token_order - 1
        JOIN utterance u
            ON u.id = c.utterance_id
        JOIN transcript t
            ON t.id = c.transcript_id
        WHERE {where_clause}
          AND c.gloss IN ({placeholders})
        ORDER BY t.filename, u.utterance_order, c.token_order
    """

    rng = random.Random(rejected_seed)
    rejected_samples: list[dict[str, object]] = []
    rejected_seen = 0

    rows_written = 0
    with connect_childes_db(db_name) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)

            with open(output_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=OUTPUT_HEADERS)
                writer.writeheader()

                for (
                    file_name,
                    collection_type,
                    speaker_code,
                    speaker_role,
                    age,
                    utterance,
                    gra,
                    determiner,
                    determiner_pos,
                    classifier,
                ) in cur:
                    record = {
                        "file_name": file_name,
                        "collection_type": collection_type,
                        "speaker_code": speaker_code,
                        "speaker_role": speaker_role,
                        "age": age,
                        "utterance": utterance,
                        "gra": gra,
                        "determiner": determiner,
                        "determiner_pos": determiner_pos,
                        "classifier": classifier,
                    }

                    if not is_number_or_determiner(determiner_pos):
                        if rejected_output_path and rejected_sample_size > 0:
                            rejected_seen += 1
                            if len(rejected_samples) < rejected_sample_size:
                                rejected_samples.append(build_rejected_row(record))
                            else:
                                index = rng.randint(0, rejected_seen - 1)
                                if index < rejected_sample_size:
                                    rejected_samples[index] = build_rejected_row(record)
                        continue

                    writer.writerow(build_output_row(record))
                    rows_written += 1

    if rejected_output_path and rejected_samples:
        with open(rejected_output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REJECTED_HEADERS)
            writer.writeheader()
            writer.writerows(rejected_samples)

    return rows_written
