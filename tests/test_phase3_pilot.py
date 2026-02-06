import os
from pathlib import Path

import pytest

from classifier_pipeline.phase3_pilot import (
    OUTPUT_HEADERS,
    _compute_age_fields,
    _apply_response,
    _build_messages,
    _request_payload_for_row,
    compute_throttle_delay,
    ensure_model_allowed,
    get_temperature,
    get_retry_delay,
    load_env,
    parse_json_response,
    normalize_overuse_value,
)
from classifier_pipeline.prompts import build_messages, SYSTEM_INSTRUCTION


def test_load_env_sets_key(tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text("OPEN_ROUTER_API_KEY=test-key\n", encoding="utf-8")

    os.environ.pop("OPEN_ROUTER_API_KEY", None)
    load_env(env_path)

    assert os.environ["OPEN_ROUTER_API_KEY"] == "test-key"


def test_build_messages_includes_context():
    messages = build_messages(
        utterance="这 个 苹果",
        classifier_token="个",
        determiner_or_number="这",
        pos_tags="det cl n",
    )

    assert messages[0]["role"] == "system"
    assert SYSTEM_INSTRUCTION in messages[0]["content"]
    assert "Utterance: 这 个 苹果" in messages[1]["content"]
    assert "POS Structure: det cl n" in messages[1]["content"]
    assert "Target Classifier: 个" in messages[1]["content"]
    assert "Preceding Word: 这" in messages[1]["content"]


def test_parse_json_response_extracts_payload():
    payload = parse_json_response("""```json\n{\"identified_noun\": \"ren\", \"conventional_classifier\": \"ge\", \"conventional_classifier_zh\": \"个\", \"classifier_type\": \"General\", \"overuse_of_ge\": false, \"rationale\": \"example\"}\n```""")

    assert payload["identified_noun"] == "ren"
    assert payload["overuse_of_ge"] is False


def test_output_headers_include_new_fields():
    for field in [
        "identified_noun",
        "conventional_classifier",
        "conventional_classifier_zh",
        "classifier_type",
        "overuse_of_ge",
        "rationale",
        "age_years",
        "age_available",
        "utterance_id",
        "utterance_order",
        "classifier_token_order",
        "transcript_id",
        "determiner_type",
        "specific_semantic_class",
        "flag_for_review",
        "flag_reason",
    ]:
        assert field in OUTPUT_HEADERS


def test_output_headers_place_flags_before_rationale():
    assert OUTPUT_HEADERS.index("flag_for_review") < OUTPUT_HEADERS.index("rationale")
    assert OUTPUT_HEADERS.index("flag_reason") < OUTPUT_HEADERS.index("rationale")


def test_retry_delay_uses_retry_after():
    delay = get_retry_delay({"Retry-After": "7"}, attempt=1, base_seconds=5, max_seconds=60)

    assert delay == 7


def test_retry_delay_exponential():
    delay = get_retry_delay({}, attempt=2, base_seconds=5, max_seconds=60)

    assert delay == 20


def test_ensure_model_allowed_accepts_openrouter():
    ensure_model_allowed("moonshotai/kimi-k2.5", provider="openrouter")
    ensure_model_allowed("deepseek/deepseek-v3.2-speciale", provider="openrouter")
    ensure_model_allowed("openai/gpt-5.2-codex", provider="openrouter")


def test_ensure_model_allowed_rejects_other():
    with pytest.raises(ValueError):
        ensure_model_allowed("kimi-k2.5", provider="openrouter")


def test_compute_throttle_delay_from_limit():
    delay = compute_throttle_delay({"X-RateLimit-Limit": "20"})

    assert delay == 3.0


def test_get_temperature_kimi_k2_5():
    assert get_temperature("openrouter", "moonshotai/kimi-k2.5") == 0.3
    assert get_temperature("openrouter", "deepseek/deepseek-v3.2-speciale") == 0.3
    assert get_temperature("openrouter", "openai/gpt-5.2-codex") == 0.3


def test_request_payload_adds_reasoning_effort_for_openrouter():
    payload = _request_payload_for_row(
        provider="openrouter",
        model="openai/gpt-5.2-codex",
        messages=[{"role": "user", "content": "x"}],
    )

    assert payload["reasoning"] == {"effort": "medium"}


def test_normalize_overuse_value():
    assert normalize_overuse_value("true") is True
    assert normalize_overuse_value("False") is False
    assert normalize_overuse_value(True) is True


def test_compute_age_fields_from_age_days():
    row = _compute_age_fields({"Age": "1156.625"})

    assert row["age_available"] is True
    assert row["age_years"] == 3.2


def test_compute_age_fields_with_missing_age():
    row = _compute_age_fields({"Age": ""})

    assert row["age_available"] is False
    assert row["age_years"] == ""


def test_compute_age_fields_backfills_determiner_type():
    row = _compute_age_fields({"Age": "365.25", "Determiner/Numbers": "这"})

    assert row["determiner_type"] == "demonstrative"


def test_compute_age_fields_preserves_existing_determiner_type():
    row = _compute_age_fields({"Age": "365.25", "Determiner/Numbers": "这", "determiner_type": "demonstrative"})

    assert row["determiner_type"] == "demonstrative"


def test_compute_age_fields_backfills_specific_semantic_class():
    row = _compute_age_fields({"Age": "365.25", "Classifier": "只"})

    assert row["specific_semantic_class"] == "animacy"


def test_build_messages_backfills_specific_semantic_class():
    messages = _build_messages(
        {"Utterance": "一 只 猫", "Classifier": "只", "Determiner/Numbers": "一", "%gra": "num cl n"}
    )

    assert "Classifier Semantic Class: animacy" in messages[1]["content"]


def test_apply_response_extracts_conventional_classifier_zh():
    row = {"Age": "365.25", "Classifier type": "", "Over use of Ge...": ""}
    parsed = {
        "identified_noun": "书",
        "conventional_classifier": "ben",
        "conventional_classifier_zh": "本",
        "classifier_type": "General",
        "overuse_of_ge": True,
        "rationale": "example",
    }

    out = _apply_response(row, parsed)

    assert out["conventional_classifier_zh"] == "本"
    assert out["age_available"] is True
    assert out["age_years"] == 1.0


def test_apply_response_extracts_flag_fields():
    row = {"Age": "365.25", "Classifier type": "", "Over use of Ge...": ""}
    parsed = {
        "identified_noun": "杯子",
        "conventional_classifier": "ge",
        "conventional_classifier_zh": "个",
        "classifier_type": "General",
        "overuse_of_ge": False,
        "rationale": "Colloquially accepted.",
        "flag_for_review": True,
        "flag_reason": "colloquial_tolerance",
    }

    out = _apply_response(row, parsed)

    assert out["flag_for_review"] is True
    assert out["flag_reason"] == "colloquial_tolerance"


def test_apply_response_defaults_flag_to_false():
    row = {"Age": "365.25", "Classifier type": "", "Over use of Ge...": ""}
    parsed = {
        "identified_noun": "人",
        "conventional_classifier": "ge",
        "conventional_classifier_zh": "个",
        "classifier_type": "General",
        "overuse_of_ge": False,
        "rationale": "Standard classifier.",
    }

    out = _apply_response(row, parsed)

    assert out["flag_for_review"] is False
    assert out["flag_reason"] == ""
