import datasets
from tqdm.auto import tqdm
import re

dataset = datasets.load_dataset('PrimeIntellect/NuminaMath-QwQ-CoT-5M', cache_dir='/mnt/nfs_share/tikhonov/hf_cache')


def split_reasoning_chain(reasoning_chain: str) -> list[str]:
    triggers = ['Wait', 'Alternatively', 'Hmm', 'Perhaps', 'Maybe', 'But', 'However']

    trigger_pattern = re.compile(r'^\s*(?:' + '|'.join(triggers) + r')\b', re.IGNORECASE)

    paragraphs = re.split(r'\n\s*\n', reasoning_chain)

    if not paragraphs:
        return []

    final_chunks = []
    current_chunk_paragraphs = [paragraphs[0]]

    for paragraph in paragraphs[1:]:
        if not paragraph.strip():
            continue

        if trigger_pattern.match(paragraph.strip()):
            final_chunks.append("\n\n".join(current_chunk_paragraphs).strip())
            current_chunk_paragraphs = [paragraph]
        else:
            current_chunk_paragraphs.append(paragraph)

    if current_chunk_paragraphs:
        final_chunks.append("\n\n".join(current_chunk_paragraphs).strip())
    
    return [chunk for chunk in final_chunks if chunk]

for i in range(10):
    print('=' * 100)
    print(f"dataset['train'][{i}]['prompt']:")
    print(dataset['train'][i]['prompt'])
    print('=' * 100)
    print(f"dataset['train'][{i}]['ground_truth']:")
    print(dataset['train'][i]['ground_truth'])
    print('=' * 100)

    print(f"split_reasoning_chain(dataset['train'][{i}]['response']):")
    for chunk in split_reasoning_chain(dataset['train'][i]['response']):
        print(repr(chunk))
        print('=' * 20)
    print('=' * 100)
    print('\n\n\n')
