import re
import sys
from transformers import TextStreamer
from functools import partial


class WordHighlightingStreamer(TextStreamer):
    """Streamer for highlighting a given word, which may be split into tokens."""
    def __init__(self, tokenizer, highlight_word: str, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.highlight_word = highlight_word.lower()
        self.buffer = ""
        self.highlight_start = "\033[1;91m"  # Bold red
        self.highlight_end = "\033[0m"
        self.split_regex = re.compile(r"(\s+)")

    def on_finalized_decode(self, token_string: str, **kwargs):
        self.buffer += token_string
        parts = self.split_regex.split(self.buffer)
        
        if len(parts) > 1:
            if parts[-1] and not self.split_regex.match(parts[-1]):
                to_process = parts[:-1]
                self.buffer = parts[-1]
            else:
                to_process = parts
                self.buffer = ""
        else:
            return

        for part in to_process:
            if part.strip().lower() == self.highlight_word:
                sys.stdout.write(f"{self.highlight_start}{part}{self.highlight_end}")
            else:
                sys.stdout.write(part)
        sys.stdout.flush()

    def end(self):
        if self.buffer:
            if self.buffer.strip().lower() == self.highlight_word:
                sys.stdout.write(f"{self.highlight_start}{self.buffer}{self.highlight_end}")
            else:
                sys.stdout.write(self.buffer)
            self.buffer = ""
        sys.stdout.flush()

WordHighlightingStreamerFactory = lambda tokenizer, highlight_word, skip_prompt=False: partial(WordHighlightingStreamer, tokenizer, highlight_word, skip_prompt=skip_prompt)