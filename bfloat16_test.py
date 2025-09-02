import torch
import torch.nn.functional as F

def test_bfloat16_cosine_similarity(dim=8192, iterations=1000000):
    """
    Tests for cosine similarity numerical instability with bfloat16.
    """
    print(f"Testing cosine similarity with torch.bfloat16.")
    print(f"Vector dimension: {dim}")
    print(f"Iterations: {iterations}")
    print("-" * 30)

    for i in range(iterations):
        # Generate two random vectors on the GPU
        vec1 = torch.randn(dim, device="cuda")
        vec2 = torch.randn(dim, device="cuda")

        # To maximize the chance of getting a value close to 1, 
        # let's make the second vector very similar to the first one
        # with a tiny bit of noise.
        noise = torch.randn(dim, device="cuda") * 0.0001
        vec2 = vec1 + noise

        # Convert to bfloat16
        vec1_bf16 = vec1.to(torch.bfloat16)
        vec2_bf16 = vec2.to(torch.bfloat16)

        # Calculate cosine similarity
        similarity = F.cosine_similarity(vec1_bf16.unsqueeze(0), vec2_bf16.unsqueeze(0), dim=-1).item()

        if similarity > 1.0 or similarity < -1.0:
            print(f"\n!!! Instability detected on iteration {i} !!!")
            print(f"Similarity: {similarity}")
            
            # For diagnostics, let's calculate the same with float32
            sim_f32 = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0), dim=-1).item()
            print(f"For comparison, float32 similarity is: {sim_f32}")
            
            return

        if (i + 1) % 100000 == 0:
            print(f"Completed {i + 1}/{iterations} iterations. No instability found so far.")

    print(f"\nTest complete. No numerical instability detected in {iterations} iterations.")

if __name__ == "__main__":
    test_bfloat16_cosine_similarity()
