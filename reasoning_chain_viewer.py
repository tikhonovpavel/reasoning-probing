import gradio as gr
import datasets
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import pandas as pd
import logging
import sqlite3
import json
import argparse


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
MAX_CHUNKS = 20
MODEL_NAME = "Qwen/QwQ-32B"
OVERRIDE_MODEL_NAME: str | None = None
INITIAL_SOURCE: str = "dataset"
SQLITE_DB_PATH: str = "reasoning_traces.sqlite"

QWEN3_SPECIAL_STOPPING_PROMPT = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>"

# --- Model Loading ---
MODEL_LOADED = False
model = None
tokenizer = None
model_error_message = ""

def ensure_model_for_source(source: str):
    global MODEL_NAME, MODEL_LOADED, model, tokenizer
    desired = OVERRIDE_MODEL_NAME if OVERRIDE_MODEL_NAME else ("Qwen/QwQ-32B" if source != "sqlite" else "Qwen/Qwen3-32B")
    if MODEL_LOADED and MODEL_NAME == desired and model is not None and tokenizer is not None:
        return
    # unload previous
    try:
        if model is not None:
            del model
            torch.cuda.empty_cache()
    except Exception:
        pass
    MODEL_NAME = desired
    logger.info(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    MODEL_LOADED = True
    logger.info("Model loaded successfully.")

# Parse CLI args early to decide initial source/model before loading anything heavy
try:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--initial-source", choices=["dataset", "sqlite"], default="sqlite")
    parser.add_argument("--model-name", type=str, default='Qwen/Qwen3-0.6B')
    parser.add_argument("--sqlite-db", type=str, default=SQLITE_DB_PATH)
    args, _unknown = parser.parse_known_args()
    INITIAL_SOURCE = args.initial_source
    OVERRIDE_MODEL_NAME = args.model_name
    SQLITE_DB_PATH = args.sqlite_db
except Exception:
    INITIAL_SOURCE = "dataset"
    OVERRIDE_MODEL_NAME = None
    SQLITE_DB_PATH = "/disk/4tb/tikhonov/projects/reasoning-probing/reasoning_traces.sqlite"

# Initial load based on selected source or explicit model override
ensure_model_for_source(INITIAL_SOURCE)


# --- Dataset Loading ---
DATASET_LOADED = False
dataset = None
dataset_error_message = ""
try:
    dataset = datasets.load_dataset('PrimeIntellect/NuminaMath-QwQ-CoT-5M', cache_dir='/mnt/nfs_share/tikhonov/hf_cache')
    DATASET_LOADED = True
except Exception as e:
    dataset_error_message = f"Failed to load dataset: {e}"


# --- Core Logic ---

def split_reasoning_chain(reasoning_chain: str) -> list[str]:
    """Splits a reasoning chain into chunks based on trigger words."""
    triggers = ['Wait', 'Alternatively', 'Hmm', 'Perhaps', 'Maybe', 'But', 'However']
    trigger_pattern = re.compile(r'^\s*(?:' + '|'.join(triggers) + r')\b', re.IGNORECASE)
    paragraphs = re.split(r'\n\s*\n', reasoning_chain.strip())
    if not paragraphs or not paragraphs[0]: return []
    final_chunks = []
    current_chunk_paragraphs = [paragraphs[0]]
    for paragraph in paragraphs[1:]:
        paragraph_stripped = paragraph.strip()
        if not paragraph_stripped: continue
        if trigger_pattern.match(paragraph_stripped):
            final_chunks.append("\n\n".join(current_chunk_paragraphs).strip())
            current_chunk_paragraphs = [paragraph]
        else:
            current_chunk_paragraphs.append(paragraph)
    if current_chunk_paragraphs:
        final_chunks.append("\n\n".join(current_chunk_paragraphs).strip())
    return [chunk for chunk in final_chunks if chunk]

def count_sentences_simple(text: str) -> int:
    """A simple sentence counter based on splitting by '. '."""
    if not text:
        return 0
    normalized_text = text.replace('.\n', '. ').strip()
    sentences = [s for s in normalized_text.split('. ') if s]
    return len(sentences)

def extract_think_content(full_prompt_text: str) -> str | None:
    """Return inner text strictly inside <think>...</think> if present; else None.

    Uses tokenizer offset_mapping to ensure we align to the original templated text
    without relying on any hardcoded special tokens.
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

    # Tokenize to obtain offsets (we don't search in partially tokenized text)
    inputs = tokenizer(full_prompt_text, return_offsets_mapping=True, return_tensors="pt")
    _ = inputs.offset_mapping  # not used directly for slicing; ensures consistent tokenization context

    inner = full_prompt_text[open_pos + len(open_tag):close_pos]
    return inner.strip()

def parse_choices_text(choices_text: str) -> str:
    if not choices_text:
        return ""
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

def load_sqlite_rows(db_path: str) -> list[dict]:
    rows: list[dict] = []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, model_path, question_id, question_text, choices, correct_answer_letter,
                   full_prompt_text, full_prompt_token_ids, extracted_answer, is_correct
            FROM reasoning_traces_qpqa
            WHERE model_path = 'Qwen/Qwen3-32B'
            ORDER BY id ASC
            """
        )
        for r in cur.fetchall():
            full_prompt_text = r["full_prompt_text"] or ""
            think_content = extract_think_content(full_prompt_text)
            if not think_content:
                continue
            item = {
                "prompt": compose_user_prompt(r["question_text"] or "", r["choices"] or ""),
                "response": think_content,
                "ground_truth": f"Correct: {r['correct_answer_letter']}",
                "meta": {
                    "id": r["id"],
                    "model_path": r["model_path"],
                    "question_id": r["question_id"],
                    "is_correct": r["is_correct"],
                },
            }
            rows.append(item)
    except Exception as e:
        logger.error(f"Failed to load SQLite rows: {e}")
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return rows

def calculate_perplexity_and_token_probs(prompt, context_chunks, target_chunk, system_prompt: str | None = None):
    """
    Calculates perplexity and returns token-level probabilities for the target chunk.

    Returns:
        perplexity (float), token_infos (list[dict]) where each dict contains:
            - token: str               decoded token text
            - prob: float              probability of the token
            - start: int               start char (inclusive) in target_chunk
            - end: int                 end char (exclusive) in target_chunk
    """
    if not MODEL_LOADED or not target_chunk:
        return 0.0, []

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    if context_chunks:
        messages.extend([{"role": "assistant", "content": chunk} for chunk in context_chunks])
    
    full_messages = messages + [{"role": "assistant", "content": target_chunk}]
    templated_string = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    target_start_char = templated_string.rfind(target_chunk)
    target_end_char = target_start_char + len(target_chunk)

    inputs = tokenizer(
        templated_string, 
        return_tensors="pt", 
        return_offsets_mapping=True
    )
    
    input_ids = inputs.input_ids
    offset_mapping = inputs.offset_mapping[0]

    labels = input_ids.clone()
    token_in_target_found = False
    for i, (start, end) in enumerate(offset_mapping):
        if (start >= target_start_char and end <= target_end_char and start != end):
            token_in_target_found = True
        else:
            labels[0, i] = -100

    if not token_in_target_found:
        return float('inf'), []

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
    
    if loss is None or torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0:
        return float('inf'), []
        
    perplexity = torch.exp(loss).item()

    # --- Token probability calculation ---
    target_indices = (labels[0] != -100).nonzero(as_tuple=True)[0]
    if len(target_indices) == 0:
        return perplexity, []

    pred_indices = target_indices - 1
    if pred_indices[0] < 0:
        pred_indices = pred_indices[1:]
        target_indices = target_indices[1:]
        if len(target_indices) == 0: return perplexity, []

    logits_for_targets = logits[0, pred_indices]
    probs = torch.nn.functional.softmax(logits_for_targets, dim=-1)
    
    # Probabilities of actual tokens
    target_token_ids = labels[0, target_indices]
    actual_token_probs = probs.gather(1, target_token_ids.unsqueeze(1)).squeeze().tolist()
    
    # Argmax token probabilities
    top_token_probs, top_token_ids = torch.max(probs, dim=-1)
    top_token_probs = top_token_probs.tolist()
    decoded_top_tokens = [tokenizer.decode(token_id) for token_id in top_token_ids]
    
    if not isinstance(actual_token_probs, list):
        actual_token_probs = [actual_token_probs]
    if not isinstance(top_token_probs, list):
        top_token_probs = [top_token_probs]


    decoded_tokens = [tokenizer.decode(token_id) for token_id in target_token_ids]

    token_infos = []
    # Map token offsets into target_chunk coordinate space
    zipped_data = zip(
        target_indices.tolist(), 
        decoded_tokens, 
        actual_token_probs, 
        decoded_top_tokens,
        top_token_probs
    )
    for idx_in_seq, token_text, prob, top_token, top_prob in zipped_data:
        start_char, end_char = offset_mapping[idx_in_seq]
        # Convert from tensors to ints if needed
        start_char = int(start_char)
        end_char = int(end_char)
        if start_char >= target_start_char and end_char <= target_end_char and start_char != end_char:
            token_infos.append({
                "token": token_text,
                "prob": float(prob),
                "start": start_char - target_start_char,
                "end": end_char - target_start_char,
                "top_token": top_token,
                "top_token_prob": float(top_prob),
            })

    return perplexity, token_infos

def generate_solution_from_think(prompt_text: str, think_content: str, system_prompt: str | None = None) -> str | None:
    """If model is Qwen3, generate a solution given the thinking so far."""
    if "Qwen3" not in MODEL_NAME or not MODEL_LOADED:
        return None
    
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_text})
        
        header = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # The think_content is the reasoning chain up to a certain chunk
        templated = header + f"<think>\n{think_content}{QWEN3_SPECIAL_STOPPING_PROMPT}"
        
        inputs = tokenizer(templated, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False, # Be deterministic for the analysis
                temperature=1.0, # Not used if do_sample=False
                top_p=1.0,       # Not used if do_sample=False
            )
        prompt_len = inputs["input_ids"].shape[1]
        new_ids = output_ids[0, prompt_len:]
        continuation = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return continuation
    except Exception as e:
        logger.error(f"Solution generation failed: {e}")
        return None

def _is_escaped(text: str, pos: int) -> bool:
    """Return True if character at pos is escaped by an odd number of backslashes immediately preceding it."""
    backslashes = 0
    i = pos - 1
    while i >= 0 and text[i] == "\\":
        backslashes += 1
        i -= 1
    return (backslashes % 2) == 1

def segment_text_with_latex(text: str) -> list[tuple[int, int, bool]]:
    """
    Split text into segments (start, end, is_math) where math segments are delimited by
    one of: $$...$$, \[...\], \(...\), $...$ (unescaped). No nesting support; segments are greedy.
    """
    n = len(text)
    segments: list[tuple[int, int, bool]] = []
    pos = 0
    last_plain = 0
    in_math = False
    math_start = 0
    current_delim = None  # one of "$$", "$", "\\[", "\\("

    while pos < n:
        if not in_math:
            if text.startswith("$$", pos):
                if last_plain < pos:
                    segments.append((last_plain, pos, False))
                in_math = True
                current_delim = "$$"
                math_start = pos
                pos += 2
                continue
            if text.startswith("\\[", pos):
                if last_plain < pos:
                    segments.append((last_plain, pos, False))
                in_math = True
                current_delim = "\\["
                math_start = pos
                pos += 2
                continue
            if text.startswith("\\(", pos):
                if last_plain < pos:
                    segments.append((last_plain, pos, False))
                in_math = True
                current_delim = "\\("
                math_start = pos
                pos += 2
                continue
            ch = text[pos]
            if ch == "$" and not _is_escaped(text, pos):
                # single dollar
                if last_plain < pos:
                    segments.append((last_plain, pos, False))
                in_math = True
                current_delim = "$"
                math_start = pos
                pos += 1
                continue
            pos += 1
        else:
            if current_delim == "$$":
                if text.startswith("$$", pos):
                    segments.append((math_start, pos + 2, True))
                    pos += 2
                    last_plain = pos
                    in_math = False
                    current_delim = None
                    continue
                pos += 1
                continue
            if current_delim == "\\[":
                if text.startswith("\\]", pos):
                    segments.append((math_start, pos + 2, True))
                    pos += 2
                    last_plain = pos
                    in_math = False
                    current_delim = None
                    continue
                pos += 1
                continue
            if current_delim == "\\(":
                if text.startswith("\\)", pos):
                    segments.append((math_start, pos + 2, True))
                    pos += 2
                    last_plain = pos
                    in_math = False
                    current_delim = None
                    continue
                pos += 1
                continue
            if current_delim == "$":
                if text[pos] == "$" and not _is_escaped(text, pos):
                    segments.append((math_start, pos + 1, True))
                    pos += 1
                    last_plain = pos
                    in_math = False
                    current_delim = None
                    continue
                pos += 1
                continue

    # Close trailing segment
    if in_math:
        segments.append((math_start, n, True))
        if n > n:  # no-op, structural symmetry
            pass
    if last_plain < n:
        segments.append((last_plain, n, False))

    # Ensure segments are ordered by start
    segments.sort(key=lambda t: t[0])
    return segments

def _escape_html(text: str) -> str:
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))

def _build_char_info_array(length: int, token_infos: list[dict] | None) -> list[dict | None]:
    """Build a per-character array containing token metadata for tooltip generation."""
    arr: list[dict | None] = [None] * length
    if not token_infos:
        return arr
    for info in token_infos:
        start = max(0, int(info.get("start", 0)))
        end = min(length, int(info.get("end", 0)))
        if start >= end:
            continue
        # This data is replicated for each character belonging to the token.
        char_data = {
            "token": info.get("token", ""),
            "prob": info.get("prob", 0.0),
            "top_token": info.get("top_token", ""),
            "top_token_prob": info.get("top_token_prob", 0.0),
        }
        for i in range(start, end):
            arr[i] = char_data
    return arr

def build_highlighted_html_from_token_infos(text: str, baseline_infos: list[dict], new_infos: list[dict]) -> str:
    """Construct HTML with per-character background and tooltips.
    
    - Uses token offsets (relative to the original `text`) derived from full-template tokenization.
    - Avoids injecting spans inside LaTeX/math segments to prevent MathJax breakage.
    """
    if not text:
        return ""

    length = len(text)
    baseline_char_infos = _build_char_info_array(length, baseline_infos)
    new_char_infos = _build_char_info_array(length, new_infos)

    diffs: list[float] = []
    for i in range(length):
        b_info = baseline_char_infos[i]
        n_info = new_char_infos[i]
        if b_info and n_info:
            diffs.append(abs(n_info["prob"] - b_info["prob"]))
        else:
            diffs.append(0.0)

    max_diff = max(diffs) if diffs else 0.0
    # If no difference, just return HTML-escaped original text with tooltips
    if max_diff <= 1e-6: # Use a small epsilon
        html_parts: list[str] = ["<div class=\"chunk-text\">"]
        last_info_str = None
        run_start = 0
        for i in range(length):
            b_info = baseline_char_infos[i]
            n_info = new_char_infos[i]
            current_info_str = ""
            if b_info and n_info:
                # Deliberately verbose for clarity
                t = b_info['token']
                bp_argmax_tok = b_info['top_token']
                bp_argmax_prob = b_info['top_token_prob']
                np_argmax_tok = n_info['top_token']
                np_argmax_prob = n_info['top_token_prob']
                bp_tok_prob = b_info['prob']
                np_tok_prob = n_info['prob']
                delta = np_tok_prob - bp_tok_prob
                
                current_info_str = (
                    f"Token: '{t}'\n"
                    f"P(token|baseline): {bp_tok_prob:.3f}\n"
                    f"P(token|new): {np_tok_prob:.3f}\n"
                    f"Δ(prob): {delta:+.3f}"
                    "___________\n\n"
                    f"Baseline argmax: '{bp_argmax_tok}' ({bp_argmax_prob:.3f})\n"
                    f"New argmax: '{np_argmax_tok}' ({np_argmax_prob:.3f})\n"
                )

            if current_info_str != last_info_str:
                if i > run_start:
                    segment = text[run_start:i]
                    if last_info_str:
                        escaped_tooltip = _escape_html(last_info_str).replace("\n", "&#10;")
                        html_parts.append(f"<span title=\"{escaped_tooltip}\">{_escape_html(segment)}</span>")
                    else:
                        html_parts.append(_escape_html(segment))
                run_start = i
                last_info_str = current_info_str

        # Append final segment
        if run_start < length:
            segment = text[run_start:length]
            if last_info_str:
                escaped_tooltip = _escape_html(last_info_str).replace("\n", "&#10;")
                html_parts.append(f"<span title=\"{escaped_tooltip}\">{_escape_html(segment)}</span>")
            else:
                html_parts.append(_escape_html(segment))
        
        html_parts.append("</div>")
        return "".join(html_parts)


    # Quantize to reduce span churn
    levels = 12
    def bucketize(v: float) -> int:
        if v <= 0:
            return 0
        return max(1, min(levels, int(round((v / max_diff) * levels))))

    # Style alpha mapping for buckets
    min_alpha = 0.12
    max_alpha = 0.85
    def alpha_for(bucket: int) -> float:
        if bucket <= 0:
            return 0.0
        return min_alpha + (max_alpha - min_alpha) * (bucket / levels)

    segments = segment_text_with_latex(text)
    html_parts: list[str] = ["<div class=\"chunk-text\">"]

    for seg_start, seg_end, is_math in segments:
        segment_text = text[seg_start:seg_end]
        if is_math:
            # Leave math untouched
            html_parts.append(segment_text)
            continue

        # Process non-math segments for highlighting and tooltips
        idx = seg_start
        while idx < seg_end:
            b_info = baseline_char_infos[idx]
            n_info = new_char_infos[idx]
            
            # Form tooltip string for current position
            tooltip_str = ""
            if b_info and n_info:
                t = b_info['token']
                bp_argmax_tok = b_info['top_token']
                bp_argmax_prob = b_info['top_token_prob']
                np_argmax_tok = n_info['top_token']
                np_argmax_prob = n_info['top_token_prob']
                bp_tok_prob = b_info['prob']
                np_tok_prob = n_info['prob']
                delta = np_tok_prob - bp_tok_prob
                
                tooltip_str = (
                    f"Token: '{t}'\n"
                    f"P(token|baseline): {bp_tok_prob:.3f}\n"
                    f"P(token|new): {np_tok_prob:.3f}\n"
                    f"Δ(prob): {delta:+.3f}"
                    "___________\n\n"
                    f"Baseline argmax: '{bp_argmax_tok}' ({bp_argmax_prob:.3f})\n"
                    f"New argmax: '{np_argmax_tok}' ({np_argmax_prob:.3f})\n"
                )

            current_bucket = bucketize(diffs[idx])
            
            run_start = idx
            idx += 1
            # Greedily extend run with same bucket and tooltip
            while (idx < seg_end and 
                   bucketize(diffs[idx]) == current_bucket and
                   baseline_char_infos[idx] == b_info): # Pointer comparison is fine for this dict
                idx += 1
            
            run_text = text[run_start:idx]
            escaped_run_text = _escape_html(run_text)
            escaped_tooltip = _escape_html(tooltip_str).replace("\n", "&#10;")

            if current_bucket == 0:
                if tooltip_str:
                    html_parts.append(f"<span title=\"{escaped_tooltip}\">{escaped_run_text}</span>")
                else:
                    html_parts.append(escaped_run_text)
            else:
                alpha = alpha_for(current_bucket)
                style = f"background-color: rgba(255, 80, 0, {alpha:.3f});"
                if tooltip_str:
                    html_parts.append(f"<span class=\"tok-diff\" style=\"{style}\" title=\"{escaped_tooltip}\">{escaped_run_text}</span>")
                else:
                    html_parts.append(f"<span class=\"tok-diff\" style=\"{style}\">{escaped_run_text}</span>")

    html_parts.append("</div>")
    return "".join(html_parts)

def build_style(min_width_px: int) -> str:
    """Return a style tag to control chunk min-width dynamically."""
    safe_width = max(200, int(min_width_px))
    return (
        f"<style>\n"
        f"#chunks-container {{ display: block !important; }}\n"
        f"#chunks-container > div {{ overflow-x: auto !important; padding-bottom: 15px; }}\n"
        f"#chunks-row {{ display: inline-flex !important; flex-wrap: nowrap !important; gap: 12px; padding: 5px; }}\n"
        f".reasoning-chunk {{ min-width: {safe_width}px !important; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #f9f9f9; padding: 10px; color: #333; }}\n"
        f".reasoning-chunk > div {{ overflow: visible !important; }}\n"
        f".reasoning-chunk .chunk-text {{ white-space: pre-wrap; }}\n"
        f".reasoning-chunk .tok-diff {{ border-radius: 2px; }}\n"
        f".reasoning-chunk .vega-embed .axis text {{ font-size: 5px !important; }}\n"
        f".reasoning-chunk .vega-embed .axis-title {{ font-size: 8px !important; }}\n"
        f".reasoning-chunk .vega-embed .legend .label {{ font-size: 8px !important; }}\n"
        f"</style>"
    )

# --- Gradio UI ---
with gr.Blocks(css="#chunks-container > div { overflow-x: auto !important; padding-bottom: 15px; }") as demo:
    gr.Markdown("# Reasoning Chain Perplexity Analyzer")
    style_injector = gr.HTML(value=build_style(380))
    
    # Pre-load SQLite rows from configured path
    default_sql_rows = load_sqlite_rows(SQLITE_DB_PATH)

    chunks_state = gr.State([])
    baseline_ppls_state = gr.State([])
    baseline_token_probs_state = gr.State([])
    prompt_state = gr.State("")
    # Source control states
    source_state = gr.State(INITIAL_SOURCE)
    sql_rows_state = gr.State(default_sql_rows)
    baseline_solutions_state = gr.State([])

    with gr.Row():
        source_selector = gr.Dropdown(choices=["dataset", "sqlite"], value=INITIAL_SOURCE, label="Source")
        index_input = gr.Number(value=0, step=1, label="Index", precision=0)
        recalc_button = gr.Button("Recalculate Perplexity", variant="primary")
    
    with gr.Row():
        baseline_system_prompt_input = gr.Textbox(
            label="Baseline System Prompt (for initial Forced Solution)", 
            lines=3, 
            placeholder="System prompt for the 'Baseline Forced Solution'. PPL calculation ignores this.",
            value="Answer only with a letter of a correct choice."
        )
        recalc_system_prompt_input = gr.Textbox(
            label="Recalculate System Prompt (for New PPL & Solution)", 
            lines=3, 
            placeholder="System prompt for 'New PPL' and 'New Forced Solution'.",
            value="Answer only with a letter of a correct choice."
        )

    use_baseline_prompt_for_ppl_cb = gr.Checkbox(
        label="Use Baseline System Prompt for Baseline PPL Calculation",
        value=True
    )

    source_info_md = gr.Markdown(value=(f"Using dataset. Available: {len(dataset['train']) if DATASET_LOADED else 0}" if INITIAL_SOURCE == "dataset" else f"Using SQLite. Loaded rows: {len(default_sql_rows)}"))

    latex_delimiters_config = [{"left": "$$", "right": "$$", "display": True}, {"left": "$", "right": "$", "display": False}, {"left": r"\(", "right": r"\)", "display": False}, {"left": r"\[", "right": r"\]", "display": True}]
    prompt_output = gr.Markdown(latex_delimiters=latex_delimiters_config)
    ground_truth_output = gr.Markdown(latex_delimiters=latex_delimiters_config)
    
    gr.Markdown("## Reasoning Chunks")
    with gr.Group(elem_id="chunks-container"):
        with gr.Row(elem_id="chunks-row"):
            chunk_cols, chunk_checkboxes, chunk_markdowns, chunk_plots = [], [], [], []
            for i in range(MAX_CHUNKS):
                with gr.Column(visible=False, scale=1, elem_classes="reasoning-chunk") as col:
                    cb = gr.Checkbox(label=f"Use Chunk {i+1}", value=True)
                    md = gr.Markdown(latex_delimiters=latex_delimiters_config)
                    plot = gr.LinePlot(interactive=False)
                    chunk_cols.append(col)
                    chunk_checkboxes.append(cb)
                    chunk_markdowns.append(md)
                    chunk_plots.append(plot)

    gr.Markdown("## Compose & Generate")
    composed_text_tb = gr.Textbox(label="Composed Reasoning (editable)", lines=12)
    generate_button = gr.Button("Generate", variant="primary")

    baseline_outputs = [style_injector, prompt_output, ground_truth_output, chunks_state, baseline_ppls_state, baseline_token_probs_state, baseline_solutions_state, prompt_state]
    baseline_outputs.extend(chunk_cols)
    baseline_outputs.extend(chunk_checkboxes)
    baseline_outputs.extend(chunk_markdowns)
    baseline_outputs.extend(chunk_plots)
    baseline_outputs.append(composed_text_tb)

    def get_baseline_final(index_str, current_source, sql_rows, baseline_system_prompt, use_baseline_prompt_for_ppl):
        # Ensure correct model/tokenizer for current source
        ensure_model_for_source(current_source)
        try:
            index = int(index_str)
            if current_source == "sqlite":
                if not sql_rows or index < 0 or index >= len(sql_rows):
                    raise IndexError()
                item = sql_rows[index]
            else:
                item = dataset['train'][index]
        except (ValueError, IndexError):
            # Simplified error handling
            updates = {
                prompt_output: gr.update(value=f"Invalid index: {index_str}", visible=True), 
                ground_truth_output: gr.update(visible=False)
            }
            # Clear all chunks
            for i in range(MAX_CHUNKS):
                updates[chunk_cols[i]] = gr.update(visible=False)
                updates[chunk_markdowns[i]] = gr.update(value="")
                updates[chunk_plots[i]] = gr.update(visible=False, value=None)
                updates[chunk_checkboxes[i]] = gr.update(visible=False, value=True, label=f"Use Chunk {i+1}")
            updates[composed_text_tb] = gr.update(value="")
            return updates

        prompt = item['prompt']
        response = item['response']
        ground_truth_value = item.get('ground_truth', '') if isinstance(item, dict) else item['ground_truth']
        chunks = split_reasoning_chain(response)
        
        if len(chunks) > MAX_CHUNKS:
            skip_message = f"## Skipped Example {index}\n\nReason: Too many chunks ({len(chunks)} > {MAX_CHUNKS})."
            updates = {
                prompt_output: gr.update(value=skip_message, visible=True), 
                ground_truth_output: gr.update(visible=False)
            }
            # Clear all chunks
            for i in range(MAX_CHUNKS):
                updates[chunk_cols[i]] = gr.update(visible=False)
                updates[chunk_markdowns[i]] = gr.update(value="")
                updates[chunk_plots[i]] = gr.update(visible=False, value=None)
                updates[chunk_checkboxes[i]] = gr.update(visible=False, value=True, label=f"Use Chunk {i+1}")
            updates[composed_text_tb] = gr.update(value="")
            return updates

        updates = {
            style_injector: gr.update(value=build_style(380)), # Hardcoded width for now
            prompt_output: gr.update(value=f"## Prompt\n\n{prompt}", visible=True),
            ground_truth_output: gr.update(value=f"## Ground Truth\n\n{ground_truth_value}", visible=True),
            chunks_state: chunks,
            prompt_state: prompt,
        }

        baseline_perplexities = []
        baseline_token_probs = []
        baseline_solutions = []
        
        system_prompt_for_baseline_ppl = baseline_system_prompt if use_baseline_prompt_for_ppl else None

        logger.info("Calculating baseline perplexities and token probabilities...")
        for i in tqdm(range(len(chunks)), desc="Baseline PPL & Probs"):
            chunk = chunks[i]
            # Baseline is calculated WITH or WITHOUT the system prompt based on the checkbox.
            ppl, token_info = calculate_perplexity_and_token_probs(prompt, chunks[:i], chunk, system_prompt=system_prompt_for_baseline_ppl)
            baseline_perplexities.append(ppl)
            baseline_token_probs.append(token_info)

            md_text = f"**Baseline PPL: {ppl:.2f}**"
            generated_solution_text = ""
            if "Qwen3" in MODEL_NAME:
                solution_context = "\n\n".join(chunks[:i+1])
                generated_solution = generate_solution_from_think(prompt, solution_context, baseline_system_prompt)
                generated_solution_text = generated_solution or ""
                if generated_solution:
                    escaped_solution = _escape_html(generated_solution).replace("\n", "<br>")
                    blockquote_html = f"<blockquote>{escaped_solution}</blockquote>"
                    md_text += f"\n\n**Baseline Forced Solution:**\n\n{blockquote_html}"
            
            baseline_solutions.append(generated_solution_text)

            md_text += f"\n\n---\n\n{chunk}"
            updates[chunk_markdowns[i]] = gr.update(value=md_text)
            
            num_sentences = count_sentences_simple(chunk)
            checkbox_label = f"Use Chunk {i+1} ({num_sentences} sentences)"
            updates[chunk_checkboxes[i]] = gr.update(visible=True, value=True, label=checkbox_label)

            updates[chunk_cols[i]] = gr.update(visible=True)
            
            if token_info:
                df = pd.DataFrame([(t["token"], t["prob"]) for t in token_info], columns=["Token", "Probability"])
                df["Index"] = list(range(1, len(df) + 1))
                updates[chunk_plots[i]] = gr.update(value=df, x="Index", y="Probability", color=None, title=f"Chunk {i+1} Baseline Probs", visible=True, y_lim=[0, 1], tooltip=["Token", "Index"])
            else:
                updates[chunk_plots[i]] = gr.update(visible=False, value=None)

        updates[baseline_ppls_state] = baseline_perplexities
        updates[baseline_token_probs_state] = baseline_token_probs
        updates[baseline_solutions_state] = baseline_solutions

        # Initialize composed text with all chunks enabled
        composed_initial = "\n\n".join(chunks)
        updates[composed_text_tb] = gr.update(value=composed_initial)

        for i in range(len(chunks), MAX_CHUNKS):
            updates[chunk_cols[i]] = gr.update(visible=False)
        
        logger.info("Done.")
        return updates

    def recalculate_perplexities(prompt_text, chunks, baseline_ppls, baseline_token_probs, baseline_solutions, recalc_system_prompt, *cb_values):
        if not chunks: return {}
        logger.info("Recalculating...")
        updates = {}
        for i in tqdm(range(len(chunks)), desc="Recalculating PPL & Probs"):
            chunk = chunks[i]
            baseline_ppl = baseline_ppls[i]
            baseline_info = baseline_token_probs[i]

            if not cb_values[i]:
                updates[chunk_markdowns[i]] = gr.update(value=f"**Baseline PPL: {baseline_ppl:.2f}**\n**(DISABLED)**\n\n---\n\n{chunk}")
                updates[chunk_plots[i]] = gr.update(visible=False, value=None)
                continue

            context_chunks = [chunks[j] for j in range(i) if cb_values[j]]
            new_ppl, new_token_info = calculate_perplexity_and_token_probs(prompt_text, context_chunks, chunk, recalc_system_prompt)
            
            # Build highlighted HTML using token span offsets to avoid tokenizing partial strings
            if baseline_info and new_token_info:
                highlighted_html = build_highlighted_html_from_token_infos(chunk, baseline_info, new_token_info)
            else:
                highlighted_html = _escape_html(chunk)
                highlighted_html = f"<div class=\"chunk-text\">{highlighted_html}</div>"

            md_text = f"**Baseline PPL: {baseline_ppl:.2f}**\n**New PPL: {new_ppl:.2f}**"

            # Display baseline solution retrieved from state
            if i < len(baseline_solutions) and baseline_solutions[i]:
                baseline_solution = baseline_solutions[i]
                escaped_solution = _escape_html(baseline_solution).replace("\n", "<br>")
                blockquote_html = f"<blockquote>{escaped_solution}</blockquote>"
                md_text += f"\n\n**Baseline Forced Solution:**\n\n{blockquote_html}"

            # Generate and display new solution
            if "Qwen3" in MODEL_NAME:
                current_and_context_chunks = context_chunks + [chunk]
                solution_context = "\n\n".join(current_and_context_chunks)
                generated_solution = generate_solution_from_think(prompt_text, solution_context, recalc_system_prompt)
                if generated_solution:
                    escaped_solution = _escape_html(generated_solution).replace("\n", "<br>")
                    blockquote_html = f"<blockquote>{escaped_solution}</blockquote>"
                    md_text += f"\n\n**New Forced Solution:**\n\n{blockquote_html}"

            md_text += f"\n\n---\n\n{highlighted_html}"
            updates[chunk_markdowns[i]] = gr.update(value=md_text)
            
            if baseline_info and new_token_info:
                # Build two series with explicit positional index to preserve order per series
                df_base = pd.DataFrame([(t["token"], t["prob"]) for t in baseline_info], columns=["Token", "Probability"])  
                df_base["Index"] = list(range(1, len(df_base) + 1))
                df_base["Series"] = "Baseline"

                df_new = pd.DataFrame([(t["token"], t["prob"]) for t in new_token_info], columns=["Token", "Probability"])  
                df_new["Index"] = list(range(1, len(df_new) + 1))
                df_new["Series"] = "New"

                df_long = pd.concat([df_base, df_new], ignore_index=True)

                updates[chunk_plots[i]] = gr.update(value=df_long, x="Index", y="Probability", visible=True, y_lim=[0, 1], color="Series", title=f"Chunk {i+1} Probability Comparison", tooltip=["Token", "Series", "Index"])
            elif baseline_info: # If new failed, show baseline
                df = pd.DataFrame([(t["token"], t["prob"]) for t in baseline_info], columns=["Token", "Probability"]) 
                df["Index"] = list(range(1, len(df) + 1))
                updates[chunk_plots[i]] = gr.update(value=df, x="Index", y="Probability", color=None, title=f"Chunk {i+1} Baseline Probs", visible=True, y_lim=[0, 1], tooltip=["Token", "Index"])
            else:
                updates[chunk_plots[i]] = gr.update(visible=False, value=None)

        logger.info("Done.")
        # Update composed text based on enabled chunks
        try:
            enabled_chunks = [chunks[i] for i in range(len(chunks)) if cb_values[i]]
        except Exception:
            enabled_chunks = chunks
        updates[composed_text_tb] = gr.update(value="\n\n".join(enabled_chunks))
        return updates

    # Gradio changed its API, now we can return dicts if outputs are named. Let's name them.
    recalc_outputs = {**{m: m for m in chunk_markdowns}, **{p: p for p in chunk_plots}}

    index_input.change(
        get_baseline_final,
        inputs=[index_input, source_state, sql_rows_state, baseline_system_prompt_input, use_baseline_prompt_for_ppl_cb],
        outputs=list(recalc_outputs.values()) + baseline_outputs
    ).then(
        None, # This is a placeholder for the component dict wiring
        inputs=None, outputs=None, js="() => { console.log('Update dictionary wiring is complex, returning full list.'); }"
    )

    def recalculate_perplexities_with_source(current_source, prompt_text, chunks, baseline_ppls, baseline_token_probs, baseline_solutions, recalc_system_prompt, *cb_values):
        ensure_model_for_source(current_source)
        return recalculate_perplexities(prompt_text, chunks, baseline_ppls, baseline_token_probs, baseline_solutions, recalc_system_prompt, *cb_values)

    recalc_button.click(
        recalculate_perplexities_with_source,
        inputs=[source_state, prompt_state, chunks_state, baseline_ppls_state, baseline_token_probs_state, baseline_solutions_state, recalc_system_prompt_input, *chunk_checkboxes],
        outputs=list(recalc_outputs.keys()) + [composed_text_tb]
    )

    # --- Generation of continuation ---
    def generate_continuation(current_source, prompt_text, composed_text, system_prompt: str | None = None):
        """Generate 50-token continuation for composed_text with hidden user prompt and <think> context."""
        ensure_model_for_source(current_source)
        if not prompt_text:
            return composed_text or ""
        base_text = composed_text or ""
        try:
            # Manually construct prompt for continuation based on tokenizer_test.ipynb.
            # Using apply_chat_template with a partial assistant message is unreliable.
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt_text})
            header = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
 
            if "Qwen3" in MODEL_NAME:
                templated = header + f"<think>\n{base_text}"
            else: # For QwQ models which do not use <think>
                templated = header + base_text
 
            inputs = tokenizer(
                templated,
                return_tensors="pt",
            )
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                )
            prompt_len = inputs["input_ids"].shape[1]
            new_ids = output_ids[0, prompt_len:]
            continuation = tokenizer.decode(new_ids, skip_special_tokens=True)
            return (base_text + continuation)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return base_text

    def on_generate(current_source, prompt_text, composed_text, system_prompt):
        new_text = generate_continuation(current_source, prompt_text, composed_text, system_prompt)
        return gr.update(value=new_text)

    generate_button.click(
        on_generate,
        inputs=[source_state, prompt_state, composed_text_tb, recalc_system_prompt_input],
        outputs=[composed_text_tb],
    )

    # Live recomposition when toggling chunk checkboxes
    def update_composed_from_cbs(chunks, *cb_values):
        if not chunks:
            return ""
        try:
            enabled_chunks = [chunks[i] for i in range(min(len(chunks), len(cb_values))) if cb_values[i]]
        except Exception:
            enabled_chunks = chunks
        return "\n\n".join(enabled_chunks)

    for _cb in chunk_checkboxes:
        _cb.change(
            update_composed_from_cbs,
            inputs=[chunks_state, *chunk_checkboxes],
            outputs=[composed_text_tb],
        )

    # Source controls callbacks
    def set_source(src, current_rows):
        ensure_model_for_source(src)
        if src == "sqlite":
            rows = current_rows if current_rows else load_sqlite_rows(SQLITE_DB_PATH)
            return src, gr.update(value=f"Using SQLite. Loaded rows: {len(rows)}"), rows
        else:
            return src, gr.update(value=f"Using dataset. Available: {len(dataset['train']) if DATASET_LOADED else 0}"), current_rows

    source_selector.change(
        set_source,
        inputs=[source_selector, sql_rows_state],
        outputs=[source_state, source_info_md, sql_rows_state]
    )

    demo.load(
        get_baseline_final,
        inputs=[index_input, source_state, sql_rows_state, baseline_system_prompt_input, use_baseline_prompt_for_ppl_cb],
        outputs=list(recalc_outputs.values()) + baseline_outputs
    )

if __name__ == "__main__":
    demo.launch()
