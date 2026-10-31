"""
Focus: cut down what the LLM has to read.

Reading is the slow part of a run (minutes, on a laptop), and its cost grows with
the amount of text in the prompt. But a boundary value or a date cutoff can only
live in a sentence that contains a number or a date. So before the LLM sees
anything, plain code:

  1. drops whole paragraphs that contain no number or date at all, and
  2. inside the others, keeps only the sentences with a number or date, plus the
     one sentence before each (it often names the thing being limited).

This test is deliberately about numbers, not about words like "maximum" or
"shall not exceed": the project must find implicit wording, so no wording list is
used to decide what the model gets to read.

Known limits: a value written without any digit or number word ("a year",
"half of income") is not seen. Use --no-focus on the command line to send full
paragraphs instead. The quote the model returns is still checked against the full
original paragraph later (grounding), so trimming cannot make a wrong value pass.
"""
from __future__ import annotations

from lending_agent.agent.text_values import numbers_in_text
from lending_agent.ingestion.chunker import split_sentences


def focus_text(text: str, context_before: int = 1) -> str:
    """Return only the number-bearing sentences of `text` (plus context), or "" if it has none."""
    sentences = split_sentences(text)
    keep: set[int] = set()
    for i, sentence in enumerate(sentences):
        if numbers_in_text(sentence):
            keep.update(range(max(0, i - context_before), i + 1))
    if not keep:
        return ""

    pieces: list[str] = []
    previous = None
    for i in sorted(keep):
        if previous is not None and i != previous + 1:
            pieces.append("...")  # tells the model that text was skipped here
        pieces.append(sentences[i])
        previous = i
    return " ".join(pieces)
