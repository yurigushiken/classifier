from classifier_pipeline.phase2_extraction import (
    FULL_CLASSIFIERS,
    OUTPUT_HEADERS,
    build_collection_clause,
    build_mandarin_language_clause,
    build_output_row,
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
    assert row["Classifier type"] == ""
    assert row["Over use of Ge..."] == ""
