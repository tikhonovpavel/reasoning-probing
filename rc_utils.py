import json
import re
from typing import List, Optional


# Shared constant used for Qwen3 forced-solution prompting
QWEN3_SPECIAL_STOPPING_PROMPT = (
    "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>"
)


def split_reasoning_chain(reasoning_chain: str) -> List[str]:
    """Splits a reasoning chain into chunks based on trigger words at paragraph starts.

    Logic mirrors the viewer implementation: paragraphs are separated by blank lines;
    a new chunk starts when a paragraph begins with one of the trigger words.
    """
    if not reasoning_chain:
        return []
    triggers = ['Wait', 'Alternatively', 'Hmm', 'Perhaps', 'Maybe', 'But', 'However']
    trigger_pattern = re.compile(r'^\s*(?:' + '|'.join(triggers) + r')\b', re.IGNORECASE)
    paragraphs = re.split(r'\n\s*\n', reasoning_chain.strip())
    if not paragraphs or not paragraphs[0]:
        return []
    final_chunks: List[str] = []
    current_chunk_paragraphs: List[str] = [paragraphs[0]]
    for paragraph in paragraphs[1:]:
        paragraph_stripped = paragraph.strip()
        if not paragraph_stripped:
            continue
        if trigger_pattern.match(paragraph_stripped):
            final_chunks.append("\n\n".join(current_chunk_paragraphs).strip())
            current_chunk_paragraphs = [paragraph]
        else:
            current_chunk_paragraphs.append(paragraph)
    if current_chunk_paragraphs:
        final_chunks.append("\n\n".join(current_chunk_paragraphs).strip())
    return [chunk for chunk in final_chunks if chunk]


def extract_think_content(full_prompt_text: str) -> Optional[str]:
    """Return inner text strictly inside <think>...</think> if present; else None.

    Tokenizer-independent; keeps behavior consistent across modules.
    """
    if not full_prompt_text:
        return None
    open_tag = "<think>"
    close_tag = "</think>"
    open_pos = full_prompt_text.find(open_tag)
    if open_pos == -1:
        return None
    close_pos = full_prompt_text.find(close_tag, open_pos + len(open_tag))
    if close_pos == -1:
        return None
    inner = full_prompt_text[open_pos + len(open_tag):close_pos]
    return inner.strip()


def parse_choices_text(choices_text: str) -> str:
    if not choices_text:
        return ""
    parsed = None
    try:
        parsed = json.loads(choices_text)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        lines = []
        for key in sorted(parsed.keys()):
            lines.append(f"{key}) {parsed[key]}")
        return "\n".join(lines)
    if isinstance(parsed, list):
        lines = []
        for idx, value in enumerate(parsed):
            letter = chr(ord('A') + idx)
            lines.append(f"{letter}) {value}")
        return "\n".join(lines)
    return choices_text.strip()


def compose_user_prompt(question_text: str, choices_text: str) -> str:
    choices_block = parse_choices_text(choices_text)
    if choices_block:
        return f"{question_text}\n\nChoices:\n{choices_block}"
    return question_text


