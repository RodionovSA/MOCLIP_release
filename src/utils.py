# src/utils.py
# Utility functions for MOCLIP evaluation

import torch
        
def recall_at_k(queries: torch.Tensor, gallery: torch.Tensor, k: int = 1, chunk_size: int = 512) -> float:
    """
    Compute R@K for a batch of queries and a gallery.
    
    Args:
        queries: Tensor of shape (N, D) containing query features.
        gallery: Tensor of shape (N, D) containing gallery features.
        k: The "K" in R@K, i.e., how many top matches to consider.
        chunk_size: Number of queries to process at once to save memory.
    """
    if queries.size(1) != gallery.size(1):
        raise ValueError("Dimension mismatch: queries and gallery must have the same feature dimension.")
    if queries.size(0) != gallery.size(0):
        raise ValueError("Size mismatch: queries and gallery must have the same number of samples.")
    
    N = queries.size(0)
    correct = []
    for i in range(0, N, chunk_size):
        q = queries[i : i + chunk_size]        # chunk_size × D
        # (chunk_size × D) @ (D × N) → (chunk_size × N)
        sims = q @ gallery.t()                 
        # for each row, find top-k indices
        topk_idx = sims.topk(k, dim=1).indices  # (chunk_size × k)
        # compare to “ground-truth” index: we assume sample i matches gallery i
        true_idx = torch.arange(i, i + q.size(0),device=queries.device).unsqueeze(1)
        hits = (topk_idx == true_idx).any(dim=1)  # (chunk_size,)
        correct.append(hits)
    return torch.cat(correct).float().mean().item()

def zero_shot_prediction(target_emb: torch.Tensor, library_emb: torch.Tensor) -> torch.Tensor:
    """
    Given a target embedding and a library of embeddings, return the index of the closest match.
    If target embeddings are spectra embeddings and library embeddings are geometry embeddings - inverse design task, 
    if target embeddings are geometry embeddings and library embeddings are spectra embeddings - forward design task.
    
    Args:
        target_emb: Tensor of shape (M, D) containing the normalized target embeddings.
        library_emb: Tensor of shape (N, D) containing the normalized library embeddings.
    Returns:
        Tensor of shape (M,) containing the indices of the closest matches in the library for each target embedding.
    """
    if target_emb.dim() != 2:
        raise ValueError("target_emb must be a 2D tensor.")
    if library_emb.dim() != 2:
        raise ValueError("library_emb must be a 2D tensor.")
    if target_emb.size(1) != library_emb.size(1):
        raise ValueError("Dimension mismatch: target_emb and library_emb must have the same feature dimension.")
    
    similarity = target_emb @ library_emb.T  # (M, D) @ (D, N) → (M, N)
    closest_indices = similarity.argmax(dim=1)  # (M,)
    
    return closest_indices
    