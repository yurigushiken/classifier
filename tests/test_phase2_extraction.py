from classifier_pipeline.phase2_extraction import (
    FULL_CLASSIFIERS,
    OUTPUT_HEADERS,
    REJECTED_HEADERS,
    build_collection_clause,
    build_mandarin_language_clause,
    build_output_row,
    build_rejected_row,
    compute_determiner_type,
    is_number_or_determiner,
)


def test_is_number_or_determiner():
    assert is_number_or_determiner("num")
    assert is_number_or_determiner("num:card")
    assert is_number_or_determiner("det")
    assert is_number_or_determiner("pro:dem")
    assert not is_number_or_determiner("adj")
    assert not is_number_or_determiner(None)


def test_build_mandarin_language_clause():
    clause, params = build_mandarin_language_clause("language", include=("zho",), exclude=("yue", "nan"))

    assert clause == "(language LIKE %s) AND language NOT LIKE %s AND language NOT LIKE %s"
    assert params == ["%zho%", "%yue%", "%nan%"]


def test_build_collection_clause():
    clause, params = build_collection_clause("collection_name", ["Chinese", "Biling"])

    assert clause == "collection_name IN (%s, %s)"
    assert params == ["Chinese", "Biling"]


def test_full_classifiers_include_reference_list():
    for token in ["位", "头", "件", "把", "颗", "辆", "种", "群", "碗"]:
        assert token in FULL_CLASSIFIERS


def test_build_output_row():
    record = {
        "file_name": "Corpus/File.cha",
        "collection_type": "Chinese",
        "speaker_code": "CHI",
        "speaker_role": "Target_Child",
        "age": 30.5,
        "utterance": "这 个 苹果",
        "gra": "det clf n",
        "determiner": "这",
        "classifier": "个",
        "utterance_id": 12345,
        "utterance_order": 5,
        "classifier_token_order": 2,
        "transcript_id": 678,
    }

    row = build_output_row(record)

    assert list(row.keys()) == OUTPUT_HEADERS
    assert row["File Name"] == "Corpus/File.cha"
    assert row["Collection_Type"] == "Chinese"
    assert row["Speaker_Code"] == "CHI"
    assert row["Speaker_Role"] == "Target_Child"
    assert row["Age"] == 30.5
    assert row["Utterance"] == "这 个 苹果"
    assert row["%gra"] == "det clf n"
    assert row["Determiner/Numbers"] == "这"
    assert row["Classifier"] == "个"
    assert row["utterance_id"] == 12345
    assert row["utterance_order"] == 5
    assert row["classifier_token_order"] == 2
    assert row["transcript_id"] == 678
    assert row["determiner_type"] == "demonstrative"
    assert row["Classifier type"] == ""
    assert row["Over use of Ge..."] == ""


def test_build_output_row_defaults_missing_sql_columns():
    """When record lacks SQL metadata columns, they default to empty strings."""
    record = {
        "file_name": "Corpus/File.cha",
        "collection_type": "Chinese",
        "speaker_code": "CHI",
        "speaker_role": "Target_Child",
        "age": 30.5,
        "utterance": "一 个 书",
        "gra": "num clf n",
        "determiner": "一",
        "classifier": "个",
    }

    row = build_output_row(record)

    assert row["utterance_id"] == ""
    assert row["utterance_order"] == ""
    assert row["classifier_token_order"] == ""
    assert row["transcript_id"] == ""
    assert row["determiner_type"] == "numeral"


def test_build_rejected_row_includes_new_columns():
    record = {
        "file_name": "Corpus/File.cha",
        "collection_type": "Chinese",
        "speaker_code": "CHI",
        "speaker_role": "Target_Child",
        "age": 30.5,
        "utterance": "这 个 苹果",
        "gra": "det clf n",
        "determiner": "这",
        "determiner_pos": "det",
        "classifier": "个",
        "utterance_id": 999,
        "utterance_order": 3,
        "classifier_token_order": 1,
        "transcript_id": 55,
    }

    row = build_rejected_row(record)

    assert list(row.keys()) == REJECTED_HEADERS
    assert row["utterance_id"] == 999
    assert row["utterance_order"] == 3
    assert row["classifier_token_order"] == 1
    assert row["transcript_id"] == 55


def test_output_headers_include_new_columns():
    for col in ["utterance_id", "utterance_order", "classifier_token_order",
                "transcript_id", "determiner_type"]:
        assert col in OUTPUT_HEADERS


def test_compute_determiner_type_demonstratives():
    assert compute_determiner_type("这") == "demonstrative"
    assert compute_determiner_type("那") == "demonstrative"


def test_compute_determiner_type_numerals():
    assert compute_determiner_type("一") == "numeral"
    assert compute_determiner_type("两") == "numeral"
    assert compute_determiner_type("三") == "numeral"
    assert compute_determiner_type("十") == "numeral"


def test_compute_determiner_type_interrogative():
    assert compute_determiner_type("几") == "interrogative"


def test_compute_determiner_type_ordinals():
    assert compute_determiner_type("第一") == "ordinal"
    assert compute_determiner_type("第三") == "ordinal"


def test_compute_determiner_type_quantifiers():
    assert compute_determiner_type("每") == "quantifier"


def test_compute_determiner_type_empty():
    assert compute_determiner_type("") == "unknown"
