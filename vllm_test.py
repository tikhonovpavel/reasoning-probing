# vllm_test_min.py
# English comments per your rule.

from vllm import LLM, SamplingParams
import math, vllm

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("vLLM version:", vllm.__version__)

llm = LLM(
    model=MODEL,
    dtype="float16",            # use "bfloat16" on bf16 GPUs
    tensor_parallel_size=1,
)

prompt = "Explain the Pythagorean theorem in one sentence."
params = SamplingParams(
    max_tokens=8,
    temperature=0.0,
    logprobs=5,                 # request top-k logprobs
    prompt_logprobs=True,       # return logprobs for prompt tokens
)

outputs = llm.generate([prompt], params)
out = outputs[0]
gen = out.outputs[0]

print("Generated text:", (gen.text or "").strip())

def chosen_lp_from_topk(topk_dict, chosen_id=None, chosen_str=None):
    # topk_dict: Dict[key, Logprob], where key is token-id (int) or token string (str)
    if not topk_dict:
        return None
    any_key = next(iter(topk_dict.keys()))
    if isinstance(any_key, int) and chosen_id is not None and chosen_id in topk_dict:
        return topk_dict[chosen_id].logprob
    if isinstance(any_key, str) and chosen_str is not None and chosen_str in topk_dict:
        return topk_dict[chosen_str].logprob
    # Fallback: take the max logprob across candidates
    return max(obj.logprob for obj in topk_dict.values())

# Prompt logprobs → compute prompt-level PPL (for sanity check)
prompt_chosen_lps = []
if getattr(out, "prompt_logprobs", None):
    prompt_token_ids = getattr(out, "prompt_token_ids", None)
    prompt_tokens = getattr(out, "prompt_tokens", None)  # may exist depending on version
    for i, topk in enumerate(out.prompt_logprobs):
        chosen_id = prompt_token_ids[i] if prompt_token_ids is not None else None
        chosen_str = prompt_tokens[i] if prompt_tokens is not None else None
        lp = chosen_lp_from_topk(topk, chosen_id=chosen_id, chosen_str=chosen_str)
        if lp is not None:
            prompt_chosen_lps.append(lp)

if prompt_chosen_lps:
    avg_neg_logprob = -sum(prompt_chosen_lps) / len(prompt_chosen_lps)
    ppl = math.exp(avg_neg_logprob)
    print("Prompt PPL:", ppl)
else:
    print("No prompt logprobs returned (check prompt_logprobs=True and model support).")

# Generated token logprobs (optional)
gen_chosen_lps = []
if getattr(gen, "logprobs", None):
    gen_token_ids = getattr(gen, "token_ids", None)
    gen_tokens = getattr(gen, "tokens", None)  # may exist depending on version
    for j, topk in enumerate(gen.logprobs):
        chosen_id = gen_token_ids[j] if gen_token_ids is not None else None
        chosen_str = gen_tokens[j] if gen_tokens is not None else None
        lp = chosen_lp_from_topk(topk, chosen_id=chosen_id, chosen_str=chosen_str)
        if lp is not None:
            gen_chosen_lps.append(lp)
print("Generated steps with logprobs:", len(gen_chosen_lps))