from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable, Optional

import requests

from classifier_pipeline.phase2_extraction import OUTPUT_HEADERS as PHASE2_HEADERS
from classifier_pipeline.phase2_extraction import compute_determiner_type
from classifier_pipeline.prompts import build_messages

OUTPUT_HEADERS = PHASE2_HEADERS + [
    "age_years",
    "age_available",
    "identified_noun",
    "conventional_classifier",
    "conventional_classifier_zh",
    "classifier_type",
    "overuse_of_ge",
    "rationale",
]

OPENROUTER_ALLOWED_MODELS = {
    "moonshotai/kimi-k2.5",
    "deepseek/deepseek-v3.2-speciale",
    "openai/gpt-5.2-codex",
}

OPENROUTER_REASONING_MODELS = {
    "moonshotai/kimi-k2.5",
    "deepseek/deepseek-v3.2-speciale",
    "openai/gpt-5.2-codex",
}


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_retry_delay(
    headers: dict[str, str],
    attempt: int,
    base_seconds: int,
    max_seconds: int,
) -> int:
    retry_after = headers.get("Retry-After") if headers else None
    if retry_after:
        try:
            return min(int(float(retry_after)), max_seconds)
        except ValueError:
            pass
    return min(base_seconds * (2**attempt), max_seconds)


def compute_throttle_delay(headers: dict[str, str]) -> float:
    limit = headers.get("X-RateLimit-Limit") if headers else None
    if not limit:
        return 0.0
    try:
        per_minute = float(limit)
    except ValueError:
        return 0.0
    if per_minute <= 0:
        return 0.0
    return round(60.0 / per_minute, 2)


def get_temperature(provider: str, model: str) -> float:
    if provider == "openrouter":
        return 0.3
    return 0.3


def get_reasoning_payload(provider: str, model: str) -> Optional[dict[str, str]]:
    if provider == "openrouter" and model in OPENROUTER_REASONING_MODELS:
        return {"effort": "medium"}
    return None


def ensure_model_allowed(model: str, provider: str) -> None:
    if provider == "openrouter":
        if model not in OPENROUTER_ALLOWED_MODELS:
            allowed = ", ".join(sorted(OPENROUTER_ALLOWED_MODELS))
            raise ValueError(f"Model not allowed for openrouter: {model}. Allowed: {allowed}")
        return
    raise ValueError(f"Unknown provider: {provider}")


def parse_json_response(text: str) -> dict[str, object]:
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0), strict=False)


def normalize_overuse_value(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _compute_age_fields(row: dict[str, str]) -> dict[str, str]:
    row = dict(row)
    age_raw = (row.get("Age") or "").strip()
    if not age_raw:
        row["age_available"] = False
        row["age_years"] = ""
    else:
        try:
            age_days = float(age_raw)
        except ValueError:
            row["age_available"] = False
            row["age_years"] = ""
        else:
            row["age_available"] = True
            row["age_years"] = round(age_days / 365.25, 1)
    if not row.get("determiner_type"):
        row["determiner_type"] = compute_determiner_type(row.get("Determiner/Numbers", ""))
    return row


def call_chat_completion(
    provider: str,
    api_key: Optional[str],
    model: str,
    base_url: str,
    messages: list[dict[str, str]],
    max_retries: int = 5,
    base_retry_seconds: int = 5,
) -> str:
    ensure_model_allowed(model, provider)

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": get_temperature(provider, model),
    }
    if provider == "openrouter":
        payload["response_format"] = {"type": "json_object"}
    reasoning = get_reasoning_payload(provider, model)
    if reasoning:
        payload["reasoning"] = reasoning

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if provider == "openrouter":
        site_url = os.environ.get("OPENROUTER_SITE_URL") or os.environ.get("OPEN_ROUTER_SITE_URL")
        app_title = os.environ.get("OPENROUTER_APP_NAME") or os.environ.get("OPEN_ROUTER_APP_NAME")
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_title:
            headers["X-Title"] = app_title

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            if response.status_code == 429:
                if attempt >= max_retries:
                    response.raise_for_status()
                delay = get_retry_delay(response.headers, attempt, base_retry_seconds, 60)
                time.sleep(delay)
                continue
            response.raise_for_status()
            data = response.json()
            throttle_delay = compute_throttle_delay(response.headers)
            if throttle_delay > 0:
                time.sleep(throttle_delay)
            return data["choices"][0]["message"]["content"]
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise
            headers = exc.response.headers if exc.response is not None else {}
            delay = get_retry_delay(headers, attempt, base_retry_seconds, 60)
            time.sleep(delay)
    raise RuntimeError("Failed to call LLM API after retries")


def _build_messages(row: dict[str, str]) -> list[dict[str, str]]:
    utterance = row.get("Utterance", "")
    classifier_token = row.get("Classifier", "")
    determiner = row.get("Determiner/Numbers", "")
    pos_tags = row.get("%gra", "")

    return build_messages(
        utterance=utterance,
        classifier_token=classifier_token,
        determiner_or_number=determiner,
        pos_tags=pos_tags,
    )


def _apply_response(row: dict[str, str], parsed: dict[str, object]) -> dict[str, str]:
    row = _compute_age_fields(row)
    row["identified_noun"] = parsed.get("identified_noun", "")
    row["conventional_classifier"] = parsed.get("conventional_classifier", "")
    row["conventional_classifier_zh"] = parsed.get("conventional_classifier_zh", "")
    classifier_type = parsed.get("classifier_type", "")
    row["classifier_type"] = classifier_type
    row["Classifier type"] = classifier_type
    overuse = normalize_overuse_value(parsed.get("overuse_of_ge"))
    overuse_value = overuse if overuse is not None else parsed.get("overuse_of_ge", "")
    row["overuse_of_ge"] = overuse_value
    row["Over use of Ge..."] = overuse_value
    row["rationale"] = parsed.get("rationale", "")
    return row


def _read_rows(input_path: Path, limit: Optional[int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _write_rows(output_path: Path, rows: Iterable[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=OUTPUT_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in OUTPUT_HEADERS})


def _load_provider_config() -> tuple[str, str, str]:
    provider = os.environ.get("LLM_PROVIDER", "openrouter").lower()
    if provider != "openrouter":
        raise RuntimeError("Only openrouter is supported")
    api_key = os.environ.get("OPEN_ROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPEN_ROUTER_API_KEY not found in environment or .env")
    model = os.environ.get("OPENROUTER_MODEL", "moonshotai/kimi-k2.5")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    ensure_model_allowed(model, provider)
    return api_key, model, base_url


def _request_payload_for_row(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
) -> dict[str, object]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": get_temperature(provider, model),
    }
    if provider == "openrouter":
        payload["response_format"] = {"type": "json_object"}
    reasoning = get_reasoning_payload(provider, model)
    if reasoning:
        payload["reasoning"] = reasoning
    return payload


def _request_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    site_url = os.environ.get("OPENROUTER_SITE_URL") or os.environ.get("OPEN_ROUTER_SITE_URL")
    app_title = os.environ.get("OPENROUTER_APP_NAME") or os.environ.get("OPEN_ROUTER_APP_NAME")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_title:
        headers["X-Title"] = app_title
    return headers


def _send_request(
    url: str,
    headers: dict[str, str],
    payload: dict[str, object],
    max_retries: int,
    base_retry_seconds: int,
) -> str:
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            if response.status_code == 429:
                if attempt >= max_retries:
                    response.raise_for_status()
                delay = get_retry_delay(response.headers, attempt, base_retry_seconds, 60)
                time.sleep(delay)
                continue
            response.raise_for_status()
            data = response.json()
            throttle_delay = compute_throttle_delay(response.headers)
            if throttle_delay > 0:
                time.sleep(throttle_delay)
            return data["choices"][0]["message"]["content"]
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise
            headers = exc.response.headers if exc.response is not None else {}
            delay = get_retry_delay(headers, attempt, base_retry_seconds, 60)
            time.sleep(delay)
    raise RuntimeError("Failed to call LLM API after retries")


def _prepare_request(
    provider: str,
    api_key: str,
    model: str,
    base_url: str,
    row: dict[str, str],
    max_retries: int,
    base_retry_seconds: int,
) -> tuple[str, dict[str, str], dict[str, object], int, int]:
    messages = _build_messages(row)
    payload = _request_payload_for_row(provider, model, messages)
    headers = _request_headers(api_key)
    url = f"{base_url.rstrip('/')}/chat/completions"
    return url, headers, payload, max_retries, base_retry_seconds


def _sync_process_row(
    provider: str,
    api_key: str,
    model: str,
    base_url: str,
    row: dict[str, str],
    max_retries: int,
    base_retry_seconds: int,
) -> dict[str, str]:
    url, headers, payload, max_retries, base_retry_seconds = _prepare_request(
        provider,
        api_key,
        model,
        base_url,
        row,
        max_retries,
        base_retry_seconds,
    )
    parse_attempts = max(2, int(os.environ.get("OPENROUTER_PARSE_RETRIES", "3")))
    last_error: Optional[Exception] = None
    for _ in range(parse_attempts):
        raw = _send_request(url, headers, payload, max_retries, base_retry_seconds)
        try:
            parsed = parse_json_response(raw)
            return _apply_response(row, parsed)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise RuntimeError("Unable to parse model response")


def run_with_semaphore(
    items: list[dict[str, str]],
    worker,
    max_concurrent: int,
) -> list[dict[str, str]]:
    async def _run() -> list[dict[str, str]]:
        semaphore = asyncio.Semaphore(max_concurrent)
        results: list[Optional[dict[str, str]]] = [None] * len(items)

        async def _run_one(index: int, item: dict[str, str]) -> None:
            async with semaphore:
                result = await worker(item)
                results[index] = result

        await asyncio.gather(*[_run_one(i, item) for i, item in enumerate(items)])
        return [row for row in results if row is not None]

    return asyncio.run(_run())


def _max_concurrent_from_env(default: int) -> int:
    value = os.environ.get("MAX_CONCURRENT")
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(1, parsed)


def run_pilot(
    input_path: Path,
    output_path: Path,
    limit: int = 20,
    env_path: Optional[Path] = None,
    max_concurrent: Optional[int] = None,
) -> int:
    env_path = env_path or Path(".env")
    load_env(env_path)

    api_key, model, base_url = _load_provider_config()
    provider = "openrouter"

    max_retries = int(os.environ.get("OPENROUTER_MAX_RETRIES", "5"))
    base_retry_seconds = int(os.environ.get("OPENROUTER_RETRY_BASE_SECONDS", "5"))

    rows = _read_rows(input_path, limit)

    async def _worker(row: dict[str, str]) -> dict[str, str]:
        return await asyncio.to_thread(
            _sync_process_row,
            provider,
            api_key,
            model,
            base_url,
            row,
            max_retries,
            base_retry_seconds,
        )

    max_concurrent = max_concurrent or _max_concurrent_from_env(10)
    processed = run_with_semaphore(rows, _worker, max_concurrent)
    _write_rows(output_path, processed)
    return len(processed)
