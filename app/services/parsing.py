def extract_json_object(text: str) -> str:
    # Balances braces (ignoring ones inside string literals) instead of a
    # greedy regex, so trailing commentary from the model with its own
    # stray braces can't extend the match past the real JSON object.
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return text[start:]
