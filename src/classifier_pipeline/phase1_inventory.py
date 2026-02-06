from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Iterable, Optional
from urllib.parse import urljoin
import csv
import json
import os
import time

import requests
from bs4 import BeautifulSoup
import pylangacq

from classifier_pipeline.childes_db import (
    apply_grouped_counts,
    connect_childes_db,
    fetch_childes_db_info,
    fetch_classifier_counts,
    fetch_speaker_counts,
    fetch_transcript_metadata,
    fetch_utterance_counts,
)

CHINESE_INDEX_URL = "https://talkbank.org/childes/access/Chinese/"
DEFAULT_LANGUAGE_FILTER = ("zho", "yue", "nan", "cmn")


@dataclass(frozen=True)
class CorpusEntry:
    section: str
    name: str
    page_url: Optional[str]
    age_range: Optional[str]
    n_files: Optional[str]
    media: Optional[str]
    comments: Optional[str]


def parse_chinese_corpora_index(html: str, base_url: str) -> list[CorpusEntry]:
    soup = BeautifulSoup(html, "lxml")

    table = None
    for candidate in soup.find_all("table"):
        if candidate.find(string=lambda s: s and "Corpus" in s):
            table = candidate
            break

    if table is None:
        return []

    entries: list[CorpusEntry] = []
    current_section: Optional[str] = None

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if not cells:
            continue

        if len(cells) == 1:
            section_text = cells[0].get_text(" ", strip=True)
            if section_text:
                current_section = section_text
            continue

        if current_section is None:
            continue

        cell_text = [cell.get_text(" ", strip=True) for cell in cells]
        name = cell_text[0] if cell_text else ""
        age_range = cell_text[1] if len(cell_text) > 1 else None
        n_files = cell_text[2] if len(cell_text) > 2 else None
        media = cell_text[3] if len(cell_text) > 3 else None
        comments = cell_text[4] if len(cell_text) > 4 else None

        anchor = cells[0].find("a")
        page_url = None
        if anchor and anchor.get("href"):
            page_url = urljoin(base_url, anchor["href"])

        entries.append(
            CorpusEntry(
                section=current_section,
                name=name,
                page_url=page_url,
                age_range=age_range,
                n_files=n_files,
                media=media,
                comments=comments,
            )
        )

    return entries


def extract_zip_url_from_corpus_page(html: str, base_url: Optional[str] = None) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")
    for anchor in soup.find_all("a"):
        href = anchor.get("href")
        if not href:
            continue
        if ".zip" in href.lower():
            return urljoin(base_url, href) if base_url else href
    return None


def fetch_chinese_corpora_index(url: str = CHINESE_INDEX_URL) -> list[CorpusEntry]:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return parse_chinese_corpora_index(response.text, base_url=url)


def fetch_zip_url_for_corpus(page_url: str) -> Optional[str]:
    response = requests.get(page_url, timeout=60)
    response.raise_for_status()
    return extract_zip_url_from_corpus_page(response.text, base_url=page_url)


def count_classifier_tokens(
    reader: pylangacq.Reader,
    classifiers: Iterable[str],
    participants: Optional[set[str]] = None,
) -> dict[str, int]:
    counts = {token: 0 for token in classifiers}
    target = set(classifiers)

    for chat_file in reader._files:
        for utterance in chat_file.utterances:
            if participants and utterance.participant not in participants:
                continue
            for token in utterance.tokens:
                word = token.word
                if word in target:
                    counts[word] += 1

    return counts


def _collect_speaker_code_counts(headers: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for header in headers:
        participants = header.get("Participants", {})
        for code in participants:
            counts[code] += 1
    return dict(counts)


def _collect_age_stats(reader: pylangacq.Reader, participant: str = "CHI") -> dict[str, Optional[float]]:
    ages = [age for age in reader.ages(participant=participant, months=True) if age is not None]
    if not ages:
        return {
            "age_months_min": None,
            "age_months_max": None,
            "age_months_n": 0,
        }

    return {
        "age_months_min": min(ages),
        "age_months_max": max(ages),
        "age_months_n": len(ages),
    }


def _count_utterances(reader: pylangacq.Reader) -> int:
    return sum(len(chat_file.utterances) for chat_file in reader._files)


def collect_corpus_stats(
    reader: pylangacq.Reader,
    classifiers: Iterable[str],
) -> dict[str, object]:
    headers = reader.headers()
    speaker_code_counts = _collect_speaker_code_counts(headers)
    age_stats = _collect_age_stats(reader, participant="CHI")

    stats = {
        "n_transcripts": reader.n_files(),
        "n_utterances": _count_utterances(reader),
        "speaker_codes": ";".join(sorted(speaker_code_counts)),
        "speaker_code_counts": speaker_code_counts,
    }
    stats.update(age_stats)
    stats["classifier_counts_all"] = count_classifier_tokens(reader, classifiers)
    stats["classifier_counts_chi"] = count_classifier_tokens(reader, classifiers, participants={"CHI"})
    return stats


def run_phase1_inventory(
    output_dir: str,
    classifiers: Iterable[str],
    sections: Optional[set[str]] = None,
) -> list[dict[str, object]]:
    os.makedirs(output_dir, exist_ok=True)

    entries = fetch_chinese_corpora_index()
    if sections:
        entries = [entry for entry in entries if entry.section in sections]

    rows: list[dict[str, object]] = []

    for entry in entries:
        row: dict[str, object] = {
            "section": entry.section,
            "corpus": entry.name,
            "corpus_page_url": entry.page_url,
            "age_range": entry.age_range,
            "n_index_files": entry.n_files,
            "media": entry.media,
            "comments": entry.comments,
            "zip_url": None,
            "status": "pending",
        }

        if not entry.page_url:
            row["status"] = "no_page_url"
            rows.append(row)
            continue

        try:
            zip_url = fetch_zip_url_for_corpus(entry.page_url)
        except requests.RequestException as exc:
            row["status"] = f"page_error:{type(exc).__name__}"
            rows.append(row)
            continue

        if not zip_url:
            row["status"] = "no_zip_url"
            rows.append(row)
            continue

        row["zip_url"] = zip_url

        try:
            reader = pylangacq.Reader.from_zip(zip_url, parallel=False)
        except Exception as exc:  # pragma: no cover - exercised in integration only
            row["status"] = f"zip_error:{type(exc).__name__}"
            rows.append(row)
            continue

        stats = collect_corpus_stats(reader, classifiers)
        row.update(stats)
        row["status"] = "ok"
        rows.append(row)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_csv(rows, os.path.join(output_dir, "phase1_corpus_stats.csv"))
    _write_json(rows, os.path.join(output_dir, "phase1_corpus_stats.json"))
    _write_summary_markdown(
        rows,
        os.path.join(output_dir, "phase1_summary.md"),
        classifiers,
        timestamp=timestamp,
    )

    return rows


def run_phase1_inventory_db(
    output_dir: str,
    classifiers: Iterable[str],
    languages: Optional[Iterable[str]] = None,
    db_name: Optional[str] = None,
) -> list[dict[str, object]]:
    os.makedirs(output_dir, exist_ok=True)

    language_filter = list(languages) if languages else list(DEFAULT_LANGUAGE_FILTER)
    db_info = fetch_childes_db_info()
    database = db_name or db_info.current

    with connect_childes_db(database) as conn:
        transcript_rows = fetch_transcript_metadata(conn, language_filter)
        utterance_counts = fetch_utterance_counts(conn, language_filter)
        speaker_counts = fetch_speaker_counts(conn, language_filter)
        classifier_counts_all = fetch_classifier_counts(conn, language_filter, classifiers)
        classifier_counts_chi = fetch_classifier_counts(
            conn,
            language_filter,
            classifiers,
            target_child_only=True,
        )

    rows_by_corpus: dict[str, dict[str, object]] = {}
    for row in transcript_rows:
        corpus = row["corpus"]
        row["section"] = row.get("collection")
        row["status"] = "ok"
        rows_by_corpus[corpus] = row

    for corpus, count in utterance_counts.items():
        corpus_row = rows_by_corpus.setdefault(corpus, {"corpus": corpus, "status": "ok"})
        corpus_row["n_utterances"] = count

    for corpus, speaker_code, speaker_role, count in speaker_counts:
        corpus_row = rows_by_corpus.setdefault(corpus, {"corpus": corpus, "status": "ok"})
        code_counts = corpus_row.setdefault("speaker_code_counts", {})
        role_counts = corpus_row.setdefault("speaker_role_counts", {})
        if speaker_code:
            code_counts[speaker_code] = code_counts.get(speaker_code, 0) + count
        if speaker_role:
            role_counts[speaker_role] = role_counts.get(speaker_role, 0) + count

    for corpus_row in rows_by_corpus.values():
        if "speaker_code_counts" in corpus_row:
            corpus_row["speaker_codes"] = ";".join(sorted(corpus_row["speaker_code_counts"].keys()))
        if "speaker_role_counts" in corpus_row:
            corpus_row["speaker_roles"] = ";".join(sorted(corpus_row["speaker_role_counts"].keys()))

    apply_grouped_counts(rows_by_corpus, classifier_counts_all, key="classifier_counts_all")
    apply_grouped_counts(rows_by_corpus, classifier_counts_chi, key="classifier_counts_chi")

    rows = list(rows_by_corpus.values())

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_csv(rows, os.path.join(output_dir, "phase1_corpus_stats.csv"))
    _write_json(rows, os.path.join(output_dir, "phase1_corpus_stats.json"))
    _write_summary_markdown(
        rows,
        os.path.join(output_dir, "phase1_summary.md"),
        classifiers,
        timestamp=timestamp,
        metadata={
            "source": "childes-db",
            "db_version": database,
            "language_filter": language_filter,
        },
    )

    return rows


def _write_csv(rows: list[dict[str, object]], path: str) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = {
                key: (json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value)
                for key, value in row.items()
            }
            writer.writerow(serialized)


def _write_json(rows: list[dict[str, object]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def _write_summary_markdown(
    rows: list[dict[str, object]],
    path: str,
    classifiers: Iterable[str],
    timestamp: str,
    metadata: Optional[dict[str, object]] = None,
) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    total_transcripts = sum(row.get("n_transcripts", 0) for row in ok_rows)
    total_utterances = sum(row.get("n_utterances", 0) for row in ok_rows)

    classifier_totals: Counter[str] = Counter()
    chi_totals: Counter[str] = Counter()
    for row in ok_rows:
        classifier_totals.update(row.get("classifier_counts_all", {}))
        chi_totals.update(row.get("classifier_counts_chi", {}))

    section_counts: Counter[str] = Counter(row.get("section", "unknown") for row in rows)
    section_ok_counts: Counter[str] = Counter(row.get("section", "unknown") for row in ok_rows)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Phase 1 Summary\n\n")
        handle.write(f"Generated: {timestamp}\n\n")
        if metadata:
            handle.write("Metadata:\n")
            for key, value in metadata.items():
                handle.write(f"- {key}: {value}\n")
            handle.write("\n")
        handle.write(f"Total corpora listed: {len(rows)}\n")
        handle.write(f"Total corpora processed (ok): {len(ok_rows)}\n\n")
        handle.write("Section counts (listed / processed):\n")
        for section, count in section_counts.items():
            handle.write(f"- {section}: {count} listed, {section_ok_counts.get(section, 0)} processed\n")

        handle.write("\nAggregate counts across processed corpora:\n")
        handle.write(f"- Total transcripts: {total_transcripts}\n")
        handle.write(f"- Total utterances: {total_utterances}\n")
        handle.write("\nClassifier token totals (all speakers):\n")
        for token in classifiers:
            handle.write(f"- {token}: {classifier_totals.get(token, 0)}\n")
        handle.write("\nClassifier token totals (CHI only):\n")
        for token in classifiers:
            handle.write(f"- {token}: {chi_totals.get(token, 0)}\n")
