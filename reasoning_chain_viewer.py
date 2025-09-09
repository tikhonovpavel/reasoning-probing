import gradio as gr
import datasets
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import pandas as pd
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
MAX_CHUNKS = 30
MODEL_NAME = "Qwen/QwQ-32B"

# --- Model Loading ---
MODEL_LOADED = False
model = None
tokenizer = None
model_error_message = ""
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=device,
    torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32
)
MODEL_LOADED = True
logger.info("Model loaded successfully.")


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

def calculate_perplexity_and_token_probs(prompt, context_chunks, target_chunk):
    """
    Calculates perplexity and returns token-level probabilities for the target chunk.
    """
    if not MODEL_LOADED or not target_chunk:
        return 0.0, []

    messages = [{"role": "user", "content": prompt}]
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
    ).to(device)
    
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
    target_token_ids = labels[0, target_indices]
    actual_token_probs = probs.gather(1, target_token_ids.unsqueeze(1)).squeeze().tolist()
    
    if not isinstance(actual_token_probs, list):
        actual_token_probs = [actual_token_probs]

    decoded_tokens = [tokenizer.decode(token_id) for token_id in target_token_ids]
    token_info = list(zip(decoded_tokens, actual_token_probs))
    
    return perplexity, token_info

def build_style(min_width_px: int) -> str:
    """Return a style tag to control chunk min-width dynamically."""
    safe_width = max(200, int(min_width_px))
    return (
        f"<style>\n"
        f"#chunks-container {{ display: block !important; }}\n"
        f"#chunks-container > div {{ overflow-x: auto !important; padding-bottom: 15px; }}\n"
        f"#chunks-row {{ display: inline-flex !important; flex-wrap: nowrap !important; gap: 12px; padding: 5px; }}\n"
        f".reasoning-chunk {{ min-width: {safe_width}px !important; border: 1px solid #444; border-radius: 8px; background-color: #1c1c1c; padding: 10px; }}\n"
        f".reasoning-chunk > div {{ overflow: visible !important; }}\n"
        f".reasoning-chunk .vega-embed .axis text {{ font-size: 5px !important; }}\n"
        f".reasoning-chunk .vega-embed .axis-title {{ font-size: 8px !important; }}\n"
        f".reasoning-chunk .vega-embed .legend .label {{ font-size: 8px !important; }}\n"
        f"</style>"
    )

# --- Gradio UI ---
with gr.Blocks(css="#chunks-container > div { overflow-x: auto !important; padding-bottom: 15px; }") as demo:
    gr.Markdown("# Reasoning Chain Perplexity Analyzer")
    style_injector = gr.HTML(value=build_style(380))
    
    chunks_state = gr.State([])
    baseline_ppls_state = gr.State([])
    baseline_token_probs_state = gr.State([])
    prompt_state = gr.State("")

    with gr.Row():
        index_input = gr.Number(value=0, step=1, label=f"Dataset Index (0 to {len(dataset['train']) - 1})", precision=0)
        chunk_min_width = gr.Number(value=380, step=10, label="Chunk min width (px)", precision=0)
        recalc_button = gr.Button("Recalculate Perplexity", variant="primary")

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

    baseline_outputs = [style_injector, prompt_output, ground_truth_output, chunks_state, baseline_ppls_state, baseline_token_probs_state, prompt_state]
    baseline_outputs.extend(chunk_cols)
    baseline_outputs.extend(chunk_checkboxes)
    baseline_outputs.extend(chunk_markdowns)
    baseline_outputs.extend(chunk_plots)

    def get_baseline_final(index_str, min_chunk_width):
        try:
            index = int(index_str)
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
                updates[chunk_checkboxes[i]] = gr.update(visible=False, value=True)
            return updates

        prompt = item['prompt']
        response = item['response']
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
                updates[chunk_checkboxes[i]] = gr.update(visible=False, value=True)
            return updates

        updates = {
            style_injector: gr.update(value=build_style(int(min_chunk_width))),
            prompt_output: gr.update(value=f"## Prompt\n\n{prompt}", visible=True),
            ground_truth_output: gr.update(value=f"## Ground Truth\n\n{item['ground_truth']}", visible=True),
            chunks_state: chunks,
            prompt_state: prompt,
        }

        baseline_perplexities = []
        baseline_token_probs = []
        
        logger.info("Calculating baseline perplexities and token probabilities...")
        for i in tqdm(range(len(chunks)), desc="Baseline PPL & Probs"):
            chunk = chunks[i]
            ppl, token_info = calculate_perplexity_and_token_probs(prompt, chunks[:i], chunk)
            baseline_perplexities.append(ppl)
            baseline_token_probs.append(token_info)

            updates[chunk_markdowns[i]] = gr.update(value=f"**Baseline PPL: {ppl:.2f}**\n\n---\n\n{chunk}")
            updates[chunk_checkboxes[i]] = gr.update(visible=True, value=True)
            updates[chunk_cols[i]] = gr.update(visible=True)
            
            if token_info:
                df = pd.DataFrame(token_info, columns=["Token", "Probability"])
                # if i < 3:
                #     print(f"--- Plotting data for chunk {i+1} (baseline) ---\n{df.head(10).to_string()}\n-------------------------------------------------")
                updates[chunk_plots[i]] = gr.update(value=df, x="Token", y="Probability", color=None, title=f"Chunk {i+1} Baseline Probs", visible=True, y_lim=[0, 1], x_label_angle=-90)
            else:
                updates[chunk_plots[i]] = gr.update(visible=False, value=None)

        updates[baseline_ppls_state] = baseline_perplexities
        updates[baseline_token_probs_state] = baseline_token_probs

        for i in range(len(chunks), MAX_CHUNKS):
            updates[chunk_cols[i]] = gr.update(visible=False)
        
        logger.info("Done.")
        return updates

    def recalculate_perplexities(prompt_text, chunks, baseline_ppls, baseline_token_probs, *cb_values):
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
            new_ppl, new_token_info = calculate_perplexity_and_token_probs(prompt_text, context_chunks, chunk)
            
            updates[chunk_markdowns[i]] = gr.update(value=f"**Baseline PPL: {baseline_ppl:.2f}**\n**New PPL: {new_ppl:.2f}**\n\n---\n\n{chunk}")
            
            if baseline_info and new_token_info:
                df_base = pd.DataFrame(baseline_info, columns=["Token", "Baseline Probability"])
                df_new = pd.DataFrame(new_token_info, columns=["Token", "New Probability"])
                
                # Merge on index to avoid issues with duplicate tokens in the 'Token' column
                df = df_base.merge(df_new[['New Probability']], left_index=True, right_index=True, how='outer')
                df['Token'] = df['Token'].fillna('')
                df = df.fillna(0)

                # Convert to long-form for stable rendering of multiple series
                df_long = pd.melt(
                    df,
                    id_vars=["Token"],
                    value_vars=["Baseline Probability", "New Probability"],
                    var_name="Series",
                    value_name="Probability"
                )

                # if i < 3:
                    # print(f"--- Plotting data for chunk {i+1} (recalculate comparison) ---\n{df_long.head(10).to_string()}\n-------------------------------------------------")
                updates[chunk_plots[i]] = gr.update(value=df_long, x="Token", y="Probability", visible=True, y_lim=[0, 1], x_label_angle=-90, color="Series", title=f"Chunk {i+1} Probability Comparison")
            elif baseline_info: # If new failed, show baseline
                df = pd.DataFrame(baseline_info, columns=["Token", "Probability"])
                # if i < 3:
                    # print(f"--- Plotting data for chunk {i+1} (recalculate baseline only) ---\n{df.head(10).to_string()}\n---------------------------------------------------")
                updates[chunk_plots[i]] = gr.update(value=df, x="Token", y="Probability", color=None, title=f"Chunk {i+1} Baseline Probs", visible=True, y_lim=[0, 1], x_label_angle=-90)
            else:
                updates[chunk_plots[i]] = gr.update(visible=False, value=None)

        logger.info("Done.")
        return updates

    # Gradio changed its API, now we can return dicts if outputs are named. Let's name them.
    recalc_outputs = {**{m: m for m in chunk_markdowns}, **{p: p for p in chunk_plots}}

    index_input.change(
        get_baseline_final,
        inputs=[index_input, chunk_min_width],
        outputs=list(recalc_outputs.values()) + baseline_outputs  # A bit of a hack to get all components
    ).then(
        None, # This is a placeholder for the component dict wiring
        inputs=None, outputs=None, js="() => { console.log('Update dictionary wiring is complex, returning full list.'); }"
    )

    recalc_button.click(
        recalculate_perplexities,
        inputs=[prompt_state, chunks_state, baseline_ppls_state, baseline_token_probs_state, *chunk_checkboxes],
        outputs=list(recalc_outputs.keys())
    )

    demo.load(
        get_baseline_final,
        inputs=[index_input, chunk_min_width],
        outputs=list(recalc_outputs.values()) + baseline_outputs
    )

if __name__ == "__main__":
    demo.launch()
