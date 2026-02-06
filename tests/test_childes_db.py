from classifier_pipeline.childes_db import apply_grouped_counts, build_language_filter_clause


def test_build_language_filter_clause():
    clause, params = build_language_filter_clause("language", ["zho", "yue"])

    assert clause == "(language LIKE %s OR language LIKE %s)"
    assert params == ["%zho%", "%yue%"]


def test_apply_grouped_counts_inserts_by_corpus():
    rows = {"Tong": {"corpus": "Tong"}}
    counts = [("Tong", "个", 3), ("Tong", "只", 2)]

    apply_grouped_counts(rows, counts, key="classifier_counts_all")

    assert rows["Tong"]["classifier_counts_all"] == {"个": 3, "只": 2}
