from classifier_pipeline.phase3_sampling import select_focus_samples, select_random_samples


def test_select_focus_samples_prefers_flowers():
    rows = [
        {"Utterance": "一 个 花"},
        {"Utterance": "一 个 书"},
        {"Utterance": "一 个 花"},
        {"Utterance": "一 个 车"},
        {"Utterance": "一 个 花"},
        {"Utterance": "一 个 人"},
        {"Utterance": "一 个 花"},
        {"Utterance": "一 个 鱼"},
        {"Utterance": "一 个 花"},
        {"Utterance": "一 个 狗"},
    ]

    sample = select_focus_samples(rows, total=6, flower_min=3, flower_max=4, seed=1)
    flower_count = sum(1 for row in sample if "花" in row.get("Utterance", ""))

    assert len(sample) == 6
    assert flower_count >= 3


def test_select_random_samples_is_reproducible():
    rows = [{"Utterance": f"row-{idx}"} for idx in range(30)]

    sample_a = select_random_samples(rows, total=20, seed=42)
    sample_b = select_random_samples(rows, total=20, seed=42)

    assert len(sample_a) == 20
    assert sample_a == sample_b
