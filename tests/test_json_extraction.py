import re
import json

import pytest

from app.services.parsing import extract_json_object


def _clean_and_extract(raw: str) -> dict:
    clean_output = re.sub(r"<\|.*?\|>", "", raw)
    clean_output = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", clean_output)
    clean_output = clean_output.strip()
    json_str = extract_json_object(clean_output)
    return json.loads(json_str)


VALID_PAYLOAD = '{"impacted_stocks":[{"reasoning":"r","direction":"BULLISH","confidence":0.8}]}'


@pytest.mark.parametrize(
    "raw",
    [
        VALID_PAYLOAD,
        f'<|start_header_id|>assistant<|end_header_id|>\n{VALID_PAYLOAD}<|eot_id|>',
        f'```json\n{VALID_PAYLOAD}\n```',
        f'Sure, here is the analysis:\n{VALID_PAYLOAD}\nHope this helps!',
        # Trailing commentary with its own stray braces - defeated the
        # original greedy regex (\{[\s\S]*\}), which matched through to
        # the LAST '}' in the whole string instead of the first balanced one.
        f'{VALID_PAYLOAD}\nNote: this is a test {{not json}}',
    ],
)
def test_extracts_valid_json_despite_model_noise(raw):
    result = _clean_and_extract(raw)
    assert result["impacted_stocks"][0]["direction"] == "BULLISH"


def test_handles_braces_inside_string_values():
    raw = '{"impacted_stocks":[{"reasoning":"uses r{eason}ing braces","direction":"BEARISH","confidence":0.4}]}'
    result = _clean_and_extract(raw)
    assert result["impacted_stocks"][0]["reasoning"] == "uses r{eason}ing braces"


def test_returns_original_text_when_no_brace_present():
    assert extract_json_object("no json here") == "no json here"
