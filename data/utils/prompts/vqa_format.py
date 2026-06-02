RE_FORMAT = """
End your answer with a separate line formatted exactly as:

Answer: X
where X ∈ {A, B, C, D, ...}.
"""

JSON_FORMAT = """
You need to respond with the answer in JSON format:

```json
{
    "analysis": "The analysis of the image and question",
    "answer": "A"
}
```
"""

LLM_FORMAT = """
Your answer must be clear and accurate.
"""

DIRECT_FORMAT = """
Note: You only need to respond with A, B, C, D, ... without providing any additional information.
"""

FORMAT_PROMPTS = {
    "re": RE_FORMAT,
    "json": JSON_FORMAT,
    "llm": LLM_FORMAT,
    "direct": DIRECT_FORMAT
}