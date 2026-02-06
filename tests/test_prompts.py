from classifier_pipeline.prompts import build_messages, SYSTEM_INSTRUCTION


def test_system_instruction_contains_examples_and_schema():
    assert "Examples" in SYSTEM_INSTRUCTION
    assert "identified_noun" in SYSTEM_INSTRUCTION
    assert "overuse_of_ge" in SYSTEM_INSTRUCTION


def test_build_messages_includes_pos_tags():
    messages = build_messages(
        utterance="一 个 苹果",
        classifier_token="个",
        determiner_or_number="一",
        pos_tags="num cl n",
    )

    assert "POS Structure: num cl n" in messages[1]["content"]
