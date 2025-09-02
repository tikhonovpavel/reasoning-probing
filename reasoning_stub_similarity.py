import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from gpqa_dataset import GPQADataset
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import random
import fire
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import itertools

def setup(model_name, seed, cache_dir):
    """Set up model, tokenizer, and dataset."""
    print("Setting up model, tokenizer, and dataset...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    config.output_hidden_states = True

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        config=config,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir
    ).to("cuda")
    model.eval()

    dataset = GPQADataset(split="train", seed=seed, cache_dir=cache_dir)
    
    print("Setup complete.")
    return model, tokenizer, dataset

def plot_results(results, output_dir, experiment_name, target_tokens):
    """Plots the results and saves them to the output directory."""
    
    # --- Heatmap: Similarity per Layer and Token ---
    plt.figure(figsize=(max(12, len(target_tokens)), 10))
    sns.heatmap(
        results["avg_sim_per_layer_token"],
        annot=True,
        fmt=".4f",
        cmap="viridis",
        xticklabels=target_tokens,
    )
    plt.xlabel("Tokens")
    plt.ylabel("Layers")
    plt.title(f"Cosine Similarity of Hidden States ({experiment_name})\nModel: {results['model_name']}")
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, f"{experiment_name}_similarity_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved heatmap to {heatmap_path}")

    # --- Line Plot: Average Similarity per Layer ---
    plt.figure(figsize=(10, 6))
    plt.plot(results["avg_sim_per_layer"], marker='o')
    plt.xlabel("Layer")
    plt.ylabel("Average Cosine Similarity")
    plt.title(f"Average Similarity Across Tokens per Layer ({experiment_name})\nModel: {results['model_name']}")
    plt.grid(True)
    plt.tight_layout()
    line_plot_path = os.path.join(output_dir, f"{experiment_name}_similarity_per_layer_line_plot.png")
    plt.savefig(line_plot_path)
    plt.close()
    print(f"Saved line plot to {line_plot_path}")


def find_target_indices(tokenizer, full_prompt, target_text):
    """Find the token indices corresponding to the target text within the full prompt."""
    
    # Find the character start and end positions of the target text in the full prompt
    target_start_char = full_prompt.find(target_text)
    if target_start_char == -1:
        # Fallback for when tokenization adds prefixes, making exact string match fail
        # This is a bit of a heuristic.
        encoded_target = tokenizer.encode(target_text, add_special_tokens=False)
        encoded_full = tokenizer.encode(full_prompt, add_special_tokens=False)
        
        # Naive subsequence search
        for i in range(len(encoded_full) - len(encoded_target) + 1):
            if encoded_full[i:i+len(encoded_target)] == encoded_target:
                return list(range(i, i + len(encoded_target))), torch.tensor([encoded_full], dtype=torch.long)
        return None, None

    target_end_char = target_start_char + len(target_text)

    # Tokenize the full prompt to get offset mappings
    inputs = tokenizer(full_prompt, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = inputs.offset_mapping[0]

    # Find tokens that fall within the target's character range
    target_token_indices = [
        i for i, (start, end) in enumerate(offset_mapping)
        if start >= target_start_char and end <= target_end_char and start < end
    ]
    
    return target_token_indices, inputs.input_ids

def get_hidden_states(model, tokenizer, sample, model_name, reasoning_stub, revealing_answer_prompt, target_text, token_limit=None):
    """Get the hidden states for a target text from a given sample."""
    
    user_prompt = sample["prompt"]
    answer_letter = sample["answer_letter"]
    
    if 'QwQ' in model_name:
        assistant_content = f"{reasoning_stub} {revealing_answer_prompt} {answer_letter}" # there's no <think></think> tokens in QwQ
    elif ('Qwen2.5' in model_name) or ('Qwen3' in model_name):
        assistant_content = f"<think>\n{reasoning_stub}\n<think>\n\n{revealing_answer_prompt} {answer_letter}"
    else:
        raise NotImplementedError()


    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_content},
    ]

    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    
    target_indices, input_ids = find_target_indices(tokenizer, full_prompt, target_text)
    
    if not target_indices:
        return None, None
        
    if token_limit:
        target_indices = target_indices[:token_limit]

    with torch.no_grad():
        outputs = model(input_ids.to(model.device))
    
    hidden_states = outputs.hidden_states
    
    # a tuple of (num_layers, batch_size, seq_len, hidden_size)
    # let's stack them into a tensor
    all_layers_hidden_states = torch.stack(hidden_states, dim=0)
    
    # (num_layers, seq_len, hidden_size)
    all_layers_hidden_states = all_layers_hidden_states.squeeze(1) 

    # get target hidden states
    target_hidden_states = all_layers_hidden_states[:, target_indices, :] # (num_layers, num_target_tokens, hidden_size)
    
    # get tokens for logging
    target_tokens = tokenizer.convert_ids_to_tokens(input_ids[0, target_indices])
    
    return target_hidden_states, target_tokens
    

def create_pairs_different_answers(dataset, num_pairs):
    """Create pairs of data samples with different answer letters."""
    print(f"Creating {num_pairs} pairs of questions with DIFFERENT answers...")
    
    samples_by_answer = defaultdict(list)
    for i in range(len(dataset)):
        sample = dataset[i]
        samples_by_answer[sample['answer_letter']].append(sample)

    # Shuffle samples within each group for randomness
    for letter in samples_by_answer:
        random.shuffle(samples_by_answer[letter])

    # Create a flat list of all samples to draw from
    all_samples = [item for sublist in samples_by_answer.values() for item in sublist]
    random.shuffle(all_samples)

    pairs = []
    used_indices = set()
    
    for i in range(len(all_samples)):
        if len(pairs) >= num_pairs:
            break
        if i in used_indices:
            continue
            
        sample1 = all_samples[i]
        
        for j in range(i + 1, len(all_samples)):
            if j in used_indices:
                continue
            
            sample2 = all_samples[j]
            
            if sample1['answer_letter'] != sample2['answer_letter']:
                pairs.append((sample1, sample2))
                used_indices.add(i)
                used_indices.add(j)
                break
    
    print(f"Created {len(pairs)} pairs.")
    return pairs

def create_data_pairs(dataset, num_pairs):
    """Create pairs of data samples with the same answer letter."""
    print(f"Creating {num_pairs} pairs of questions with the same answer...")
    
    samples_by_answer = defaultdict(list)
    for i in range(len(dataset)):
        sample = dataset[i]
        samples_by_answer[sample['answer_letter']].append(sample)

    pairs = []
    
    # Ensure we can create pairs
    possible_pairs = sum(len(samples) // 2 for samples in samples_by_answer.values())
    if possible_pairs < num_pairs:
        raise ValueError(f"Not enough samples to create {num_pairs} pairs. Maximum possible: {possible_pairs}")

    # Shuffle samples within each group for randomness
    for letter in samples_by_answer:
        random.shuffle(samples_by_answer[letter])

    # Create pairs
    while len(pairs) < num_pairs:
        for letter in sorted(samples_by_answer.keys()): # sorted for deterministic picking
            if len(samples_by_answer[letter]) >= 2:
                p1 = samples_by_answer[letter].pop()
                p2 = samples_by_answer[letter].pop()
                pairs.append((p1, p2))
                if len(pairs) == num_pairs:
                    break
    
    print(f"Created {len(pairs)} pairs.")
    return pairs

def process_and_save_results(
    experiment_name, results_per_layer, all_target_tokens,
    model_name, reasoning_stub, revealing_answer_prompt,
    n_pairs, seed, run_output_dir
):
    """Analyzes, prints, plots, and returns the results of an experiment."""
    if not results_per_layer or not results_per_layer[0]:
        print(f"\nNo results to process for experiment: {experiment_name}")
        return None

    num_layers = len(results_per_layer)
    num_tokens = len(all_target_tokens)
    
    print(f"\n--- Results for: {experiment_name} ---")
    print(f"Analysis based on {len(results_per_layer[0])} pairs.")
    print(f"Target Tokens: {all_target_tokens}")
    
    print("\nAverage Cosine Similarity per Layer and Token:")
    
    avg_sim_per_layer_token = np.zeros((num_layers, num_tokens))

    for layer_idx in range(num_layers):
        if results_per_layer[layer_idx]:
            # shape of stacked_sims: (num_pairs, num_tokens)
            stacked_sims = np.stack(results_per_layer[layer_idx])
            avg_sim_per_layer_token[layer_idx] = stacked_sims.mean(axis=0)

    # Pretty print the results table
    header = f"{'Layer':<7}" + "".join([f"{token:<10}" for token in all_target_tokens])
    print(header)
    print("-" * len(header))
    
    for i in range(num_layers):
        layer_str = f"{i:<7}"
        for val in avg_sim_per_layer_token[i]:
            layer_str += f"{val:<10.4f}"
        print(layer_str)

    print("\n--- Summary ---")
    avg_sim_per_layer = avg_sim_per_layer_token.mean(axis=1)
    for i, avg_sim in enumerate(avg_sim_per_layer):
        print(f"Layer {i:2d} average similarity: {avg_sim:.4f}")

    # --- Save raw results ---
    results_data = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "reasoning_stub": reasoning_stub,
        "revealing_answer_prompt": revealing_answer_prompt,
        "n_pairs": n_pairs,
        "seed": seed,
        "target_tokens": all_target_tokens,
        "avg_sim_per_layer_token": avg_sim_per_layer_token.tolist(),
        "avg_sim_per_layer": avg_sim_per_layer.tolist(),
        "raw_results_per_layer": [np.array(layer_res).tolist() for layer_res in results_per_layer]
    }
    
    # --- Plot results for this run ---
    plot_results(results_data, run_output_dir, experiment_name, all_target_tokens)
    
    return results_data


def main(
    model_name="Qwen/QwQ-32B",
    reasoning_stub="Okay, I have finished thinking.",
    revealing_answer_prompt="The final answer is",
    n_pairs=100,
    seed=42,
    output_dir="results",
    cache_dir: str = "/mnt/nfs_share/tikhonov/hf_cache"
):
    """Main function to run the experiment."""
    # --- Setup output directory ---
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, f"{os.path.basename(model_name)}_{run_timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Results will be saved to: {run_output_dir}")


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model, tokenizer, dataset = setup(model_name, seed, cache_dir)
    num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer

    all_experiment_results = []

    # --- Sanity Check: Assert tokenization consistency ---
    print("\nRunning tokenization sanity check...")
    base_prompt_text = f"{reasoning_stub} {revealing_answer_prompt}"
    prompts_to_check = [f"{base_prompt_text} {letter}" for letter in ["A", "B", "C", "D"]]
    tokenized_prompts = [tokenizer.encode(p) for p in prompts_to_check]

    base_length = len(tokenized_prompts[0])
    for i, tokens in enumerate(tokenized_prompts):
        assert len(tokens) == base_length, \
            f"Tokenization length mismatch! Prompt A has {base_length} tokens, but prompt {chr(65+i)} has {len(tokens)} tokens."
        # Check that the shared part is identical
        assert tokenized_prompts[0][:-1] == tokens[:-1], \
            f"Shared prompt part tokenization mismatch between prompt A and prompt {chr(65+i)}."
            
    print("Tokenization sanity check passed. Assistant prompts are consistent.")

    # =================================================================================
    # Experiment 1: Same Answer, Full Assistant Prompt
    # =================================================================================
    print("\n" + "="*80)
    print("Running Experiment 1: Comparing ASSISTANT PROMPTS for pairs with SAME answer.")
    print("="*80)
    
    same_answer_pairs = create_data_pairs(dataset, n_pairs)
    results_per_layer_exp1 = [[] for _ in range(num_layers)]
    all_target_tokens_exp1 = []

    for sample1, sample2 in tqdm(same_answer_pairs, desc="Experiment 1"):
        # Construct the full assistant prompt to be used as the target
        assistant_content = f"{reasoning_stub} {revealing_answer_prompt} {sample1['answer_letter']}"
        
        hidden_states1, target_tokens1 = get_hidden_states(
            model, tokenizer, sample1, model_name, reasoning_stub, revealing_answer_prompt, assistant_content
        )
        hidden_states2, target_tokens2 = get_hidden_states(
            model, tokenizer, sample2, model_name, reasoning_stub, revealing_answer_prompt, assistant_content
        )
        
        if target_tokens1 is None or target_tokens1 != target_tokens2:
            print("Skipping pair due to inconsistent tokenization.")
            continue

        if not all_target_tokens_exp1:
            all_target_tokens_exp1 = target_tokens1
            all_target_tokens_exp1[-1] = "[ANSWER_TOKEN]"

        for layer_idx in range(num_layers):
            sim = F.cosine_similarity(hidden_states1[layer_idx], hidden_states2[layer_idx], dim=-1)
            results_per_layer_exp1[layer_idx].append(sim.cpu().float().numpy())

    results_exp1 = process_and_save_results(
        "exp1_same_answer_assistant", results_per_layer_exp1, all_target_tokens_exp1,
        model_name, reasoning_stub, revealing_answer_prompt, len(same_answer_pairs), 
        seed, run_output_dir
    )
    if results_exp1:
        all_experiment_results.append(results_exp1)

    # =================================================================================
    # Experiment 2: Different Answer, Reasoning Part of Assistant Prompt
    # =================================================================================
    print("\n" + "="*80)
    print("Running Experiment 2: Comparing REASONING part of ASSISTANT PROMPTS for pairs with DIFFERENT answers.")
    print("="*80)

    diff_answer_pairs = create_pairs_different_answers(dataset, n_pairs)
    results_per_layer_exp2 = [[] for _ in range(num_layers)]
    all_target_tokens_exp2 = []

    for sample1, sample2 in tqdm(diff_answer_pairs, desc="Experiment 2"):
        assistant_content1 = f"{reasoning_stub} {revealing_answer_prompt} {sample1['answer_letter']}"
        assistant_content2 = f"{reasoning_stub} {revealing_answer_prompt} {sample2['answer_letter']}"

        hidden_states1, target_tokens1 = get_hidden_states(
            model, tokenizer, sample1, model_name, reasoning_stub, revealing_answer_prompt, assistant_content1
        )
        hidden_states2, target_tokens2 = get_hidden_states(
            model, tokenizer, sample2, model_name, reasoning_stub, revealing_answer_prompt, assistant_content2
        )

        if hidden_states1 is None or hidden_states2 is None:
            print("Skipping pair due to hidden state extraction failure.")
            continue
        
        # This check should pass because of our initial assert
        if hidden_states1.shape[1] != hidden_states2.shape[1]:
            print("Skipping pair due to tokenization length mismatch (should not happen).")
            continue

        # --- KEY CHANGE: Exclude the last token (the answer) from analysis ---
        hidden_states1_reasoning = hidden_states1[:, :-1, :]
        hidden_states2_reasoning = hidden_states2[:, :-1, :]
        target_tokens_reasoning = target_tokens1[:-1]

        if not all_target_tokens_exp2:
            all_target_tokens_exp2 = target_tokens_reasoning

        for layer_idx in range(num_layers):
            sim = F.cosine_similarity(hidden_states1_reasoning[layer_idx], hidden_states2_reasoning[layer_idx], dim=-1)
            results_per_layer_exp2[layer_idx].append(sim.cpu().float().numpy())

    results_exp2 = process_and_save_results(
        "exp2_different_answer_reasoning_only", results_per_layer_exp2, all_target_tokens_exp2,
        model_name, reasoning_stub, revealing_answer_prompt, len(diff_answer_pairs), 
        seed, run_output_dir
    )
    if results_exp2:
        all_experiment_results.append(results_exp2)

    # =================================================================================
    # Experiment 3: Same Answer, Question Text
    # =================================================================================
    print("\n" + "="*80)
    print("Running Experiment 3: Comparing first 15 QUESTION TOKENS for pairs with SAME answer.")
    print("="*80)
    
    results_per_layer_exp3 = [[] for _ in range(num_layers)]
    all_target_tokens_exp3 = []

    for sample1, sample2 in tqdm(same_answer_pairs, desc="Experiment 3"):
        hidden_states1, target_tokens1 = get_hidden_states(
            model, tokenizer, sample1, model_name, reasoning_stub, revealing_answer_prompt, 
            target_text=sample1['question'], token_limit=15
        )
        hidden_states2, target_tokens2 = get_hidden_states(
            model, tokenizer, sample2, model_name, reasoning_stub, revealing_answer_prompt, 
            target_text=sample2['question'], token_limit=15
        )
        
        # Questions are different, so tokens will be different. We need pairs with same tokenization length.
        if hidden_states1 is None or hidden_states2 is None or hidden_states1.shape[1] != hidden_states2.shape[1]:
            print("Skipping pair due to question tokenization length mismatch.")
            continue

        if not all_target_tokens_exp3:
            # For visualization, just use the tokens from the first valid pair
            all_target_tokens_exp3 = target_tokens1

        for layer_idx in range(num_layers):
            sim = F.cosine_similarity(hidden_states1[layer_idx], hidden_states2[layer_idx], dim=-1)
            results_per_layer_exp3[layer_idx].append(sim.cpu().float().numpy())

    results_exp3 = process_and_save_results(
        "exp3_same_answer_question", results_per_layer_exp3, all_target_tokens_exp3,
        model_name, reasoning_stub, revealing_answer_prompt, len(same_answer_pairs), 
        seed, run_output_dir
    )
    if results_exp3:
        all_experiment_results.append(results_exp3)
    
    # --- Save all results to a single JSON file ---
    results_json_path = os.path.join(run_output_dir, "results.json")
    with open(results_json_path, "w") as f:
        json.dump(all_experiment_results, f, indent=4)
    print(f"\nSaved all experiment results to {results_json_path}")


if __name__ == "__main__":
    fire.Fire(main)
