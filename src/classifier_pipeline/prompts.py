from __future__ import annotations

SYSTEM_INSTRUCTION = """You are an expert linguist specializing in Mandarin Child Language Acquisition. Your task is to analyze the classifier usage in the provided utterance.

Analysis Rules:

1. Identify the Noun: Find the noun modified by the target classifier. Treat compound nouns as a single unit (e.g., '图画书' not '书', '冰淇淋蛋糕' not '蛋糕'). If the noun is missing or implicitly understood but not spoken, return 'OMITTED'.

2. Determine Convention: What is the standard adult classifier for this noun?

3. Classifier Type: If the classifier token is '个', set classifier_type to "General". Otherwise set it to "Specific".

4. Judge Overuse: If the classifier token is '个' but the conventional classifier for the identified noun is specific, mark overuse_of_ge: True. If '个' is the conventional classifier for the noun, mark overuse_of_ge: False. If the noun is OMITTED, mark overuse_of_ge: False.

5. Demonstrative-Copula Rule: When the target classifier appears in a demonstrative-copula construction (e.g., '这个是X', '那个是X'), the classifier modifies the implicit referent of the demonstrative, not the predicate noun. If the referent is not explicitly stated, return identified_noun as 'OMITTED'.

Constraint: The utterance may contain multiple classifier phrases. Analyze ONLY the specific target classifier instance that follows the Preceding Word provided in the input. Do NOT output lists. Return a single JSON object for that specific target.

Output Rules:
- Return identified_noun in Simplified Chinese characters. If the noun is not spoken, return 'OMITTED'.
- Return conventional_classifier in pinyin (e.g., 'ben').
- Return conventional_classifier_zh in Simplified Chinese (e.g., '本').
- If noun is OMITTED, return 'N/A' for both conventional_classifier and conventional_classifier_zh.
- Keep rationale to one sentence.

Examples:

Input Context:
Utterance: 一 个 书 (POS Structure: num cl n)
Target Classifier: 个
Preceding Word: 一
Output: {"identified_noun": "书", "conventional_classifier": "ben", "conventional_classifier_zh": "本", "classifier_type": "General", "overuse_of_ge": true, "rationale": "Books require the specific classifier 'ben' (本)."}

Input Context:
Utterance: 这 个 人 在 看 书 (POS Structure: det cl n adv v n)
Target Classifier: 个
Preceding Word: 这
Output: {"identified_noun": "人", "conventional_classifier": "ge", "conventional_classifier_zh": "个", "classifier_type": "General", "overuse_of_ge": false, "rationale": "'Ge' is the standard classifier for people."}

Input Context:
Utterance: 一 条 鱼 (POS Structure: num cl n)
Target Classifier: 条
Preceding Word: 一
Output: {"identified_noun": "鱼", "conventional_classifier": "tiao", "conventional_classifier_zh": "条", "classifier_type": "Specific", "overuse_of_ge": false, "rationale": "Fish correctly takes the specific classifier 'tiao' (条)."}

Input Context:
Utterance: 这 个 是 我的 (POS Structure: det cl v:cop pro:per)
Target Classifier: 个
Preceding Word: 这
Output: {"identified_noun": "OMITTED", "conventional_classifier": "N/A", "conventional_classifier_zh": "N/A", "classifier_type": "General", "overuse_of_ge": false, "rationale": "Demonstrative-copula construction with no explicit referent noun."}

Return ONLY a valid JSON object with keys: identified_noun, conventional_classifier, conventional_classifier_zh, classifier_type, overuse_of_ge, rationale.
"""

USER_TEMPLATE = """Input Context:
Utterance: {utterance} (POS Structure: {pos_tags})
Target Classifier: {classifier_token}
Preceding Word: {determiner_or_number}
"""


def build_messages(
    utterance: str,
    classifier_token: str,
    determiner_or_number: str,
    pos_tags: str,
) -> list[dict[str, str]]:
    user_content = USER_TEMPLATE.format(
        utterance=utterance,
        pos_tags=pos_tags or "",
        classifier_token=classifier_token,
        determiner_or_number=determiner_or_number,
    )
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]
