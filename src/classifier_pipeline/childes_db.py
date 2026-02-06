from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional
import json

import pymysql
import requests

CHILDES_DB_INFO_URL = "https://langcog.github.io/childes-db-website/childes-db.json"


@dataclass(frozen=True)
class ChildesDbInfo:
    host: str
    user: str
    password: str
    current: str


def fetch_childes_db_info(url: str = CHILDES_DB_INFO_URL) -> ChildesDbInfo:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return ChildesDbInfo(
        host=payload["host"],
        user=payload["user"],
        password=payload["password"],
        current=payload["current"],
    )


def connect_childes_db(db_name: Optional[str] = None) -> pymysql.connections.Connection:
    info = fetch_childes_db_info()
    database = db_name or info.current
    return pymysql.connect(
        host=info.host,
        user=info.user,
        password=info.password,
        database=database,
        port=3306,
        charset="utf8mb4",
    )


def build_language_filter_clause(column: str, languages: Iterable[str]) -> tuple[str, list[str]]:
    patterns = [f"%{lang}%" for lang in languages]
    conditions = " OR ".join(f"{column} LIKE %s" for _ in patterns)
    return f"({conditions})", patterns


def apply_grouped_counts(
    rows_by_corpus: dict[str, dict[str, object]],
    counts: Iterable[tuple[str, str, int]],
    key: str,
) -> None:
    for corpus, token, count in counts:
        corpus_row = rows_by_corpus.setdefault(corpus, {"corpus": corpus})
        bucket = corpus_row.setdefault(key, {})
        bucket[token] = int(count)


def fetch_transcript_metadata(
    conn: pymysql.connections.Connection,
    languages: Iterable[str],
) -> list[dict[str, object]]:
    clause, params = build_language_filter_clause("language", languages)
    query = f"""
        SELECT
            corpus_name,
            collection_name,
            GROUP_CONCAT(DISTINCT language ORDER BY language SEPARATOR '; ') AS languages,
            COUNT(*) AS n_transcripts,
            SUM(target_child_age IS NOT NULL) AS n_transcripts_with_target_child_age,
            MIN(target_child_age) AS target_child_age_min,
            MAX(target_child_age) AS target_child_age_max
        FROM transcript
        WHERE {clause}
        GROUP BY corpus_name, collection_name
        ORDER BY corpus_name
    """
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    results = []
    for row in rows:
        results.append(
            {
                "corpus": row[0],
                "collection": row[1],
                "languages": row[2],
                "n_transcripts": int(row[3]),
                "n_transcripts_with_target_child_age": int(row[4]) if row[4] is not None else 0,
                "target_child_age_min": float(row[5]) if row[5] is not None else None,
                "target_child_age_max": float(row[6]) if row[6] is not None else None,
            }
        )
    return results


def fetch_utterance_counts(
    conn: pymysql.connections.Connection,
    languages: Iterable[str],
) -> dict[str, int]:
    clause, params = build_language_filter_clause("language", languages)
    query = f"""
        SELECT corpus_name, COUNT(*)
        FROM utterance
        WHERE {clause}
        GROUP BY corpus_name
    """
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    return {row[0]: int(row[1]) for row in rows}


def fetch_speaker_counts(
    conn: pymysql.connections.Connection,
    languages: Iterable[str],
) -> list[tuple[str, str, str, int]]:
    clause, params = build_language_filter_clause("language", languages)
    query = f"""
        SELECT corpus_name, speaker_code, speaker_role, COUNT(*)
        FROM utterance
        WHERE {clause}
        GROUP BY corpus_name, speaker_code, speaker_role
    """
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    return [(row[0], row[1], row[2], int(row[3])) for row in rows]


def fetch_classifier_counts(
    conn: pymysql.connections.Connection,
    languages: Iterable[str],
    classifiers: Iterable[str],
    target_child_only: bool = False,
) -> list[tuple[str, str, int]]:
    clause, params = build_language_filter_clause("language", languages)
    classifier_list = list(classifiers)
    if not classifier_list:
        return []
    placeholders = ", ".join(["%s"] * len(classifier_list))

    child_filter = " AND speaker_role = 'Target_Child'" if target_child_only else ""

    query = f"""
        SELECT corpus_name, gloss, COUNT(*)
        FROM token
        WHERE {clause}
          AND gloss IN ({placeholders})
          {child_filter}
        GROUP BY corpus_name, gloss
    """

    with conn.cursor() as cur:
        cur.execute(query, params + classifier_list)
        rows = cur.fetchall()
    return [(row[0], row[1], int(row[2])) for row in rows]


def serialize_row(row: dict[str, object]) -> dict[str, object]:
    serialized = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            serialized[key] = json.dumps(value, ensure_ascii=False)
        else:
            serialized[key] = value
    return serialized
