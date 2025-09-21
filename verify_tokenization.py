import torch
from transformers import AutoTokenizer

def analyze_tokenizer(model_name: str):
    """
    Loads a tokenizer and analyzes its behavior for single letters,
    letters with preceding spaces, and whitespace characters.
    """
    print("-" * 80)
    print(f"Analyzing tokenizer for: {model_name}")
    print("-" * 80)

    try:
        # It's good practice to set trust_remote_code=True for many new models
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    # --- 1. Analyze Letter Tokenization ---
    print("\n--- 1. Letter Tokenization Analysis ---")
    letters = ["A", "B", "C", "D"]
    letter_to_token_ids = {}

    for letter in letters:
        print(f"\n--- Analyzing letter: '{letter}' ---")
        variants = {
            f"'{letter}'": letter,
            f"' {letter}' (with space)": f" {letter}",
        }

        letter_ids = []
        for desc, text in variants.items():
            encoded = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(encoded)
            is_single_token = len(encoded) == 1

            print(f"  - Text: {desc}")
            print(f"    - Encoded IDs: {encoded}")
            print(f"    - Decoded back: '{decoded}'")
            print(f"    - Is single token: {is_single_token}")

            if is_single_token:
                letter_ids.append(encoded[0])

        letter_to_token_ids[letter] = list(dict.fromkeys(letter_ids))

    print("\n--- Summary: `letter_to_token_ids` simulation ---")
    print("This dictionary shows which token IDs would be used by the main script.")
    print(f"Result for {model_name}: {letter_to_token_ids}")


    # --- 2. Analyze Whitespace Tokenization ---
    print("\n--- 2. Whitespace Tokenization Analysis ---")
    whitespace_chars = {
        "' ' (one space)": " ",
        "'  ' (two spaces)": "  ",
        "'\\n' (newline)": "\n",
        "'\\n\\n' (two newlines)": "\n\n",
        "'\\t' (tab)": "\t",
    }
    for desc, text in whitespace_chars.items():
        encoded = tokenizer.encode(text, add_special_tokens=False)
        # Note: some tokenizers produce empty strings for certain whitespace tokens when decoded individually.
        # We decode the whole sequence to be accurate.
        decoded = tokenizer.decode(encoded)
        print(f"\n  - Text: {desc}")
        print(f"    - Encoded IDs: {encoded}")
        print(f"    - Decoded back: '{decoded}'")
        for i, token_id in enumerate(encoded):
             print(f"      - Token {i} (ID: {token_id}): '{tokenizer.decode([token_id])}'")


    # --- 3. Conclusion ---
    print("\n--- 3. Conclusion ---")
    # Check if for any letter, more than one token ID was found (implying 'A' and ' A' are distinct single tokens)
    has_space_letter_tokens = any(len(ids) > 1 for ids in letter_to_token_ids.values())

    if has_space_letter_tokens:
        print("This tokenizer HAS dedicated single tokens for letters with a preceding space (e.g., ' A').")
        print("The method in the script is likely to work well, as it can capture the probability of both 'A' and ' A' directly.")
    else:
        print("This tokenizer does NOT have dedicated single tokens for letters with a preceding space.")
        print("A string like ' A' is likely tokenized into two separate tokens: [' ', 'A'].")
        print("The logic `P(letter) = P(direct) + P(whitespace) * P(letter | whitespace)` is CRITICAL here.")
        print("The script's assumption that it can find a single token for ' A' fails for this tokenizer.")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # The model used in the script, which is expected to work well.
    qwen_model_name = "Qwen/Qwen3-32B"

    # The model from the user's command, which is Llama-based and might behave differently.
    llama_model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

    print("Starting tokenization comparison script.")
    print("This will help understand if the `letter_probs_first_nonspace` method is robust across different models.\n")

    analyze_tokenizer(qwen_model_name)
    analyze_tokenizer(llama_model_name)

    print("Script finished.")
