from retrieval import wav_to_rep
from models import SpecAutoEncoder
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def fingerprinting(wav_file_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    model = SpecAutoEncoder()
    model_name = "SpecAutoEncoder"

    model.load_state_dict(torch.load("../models" + f"/{model_name}_best.pt"))
    model.to(device)
    fingerprints = []
    for wav_file_path in wav_file_paths:
        rep = wav_to_rep(wav_file_path, model, device)
        fingerprints.append(rep)

    return torch.cat(fingerprints, dim=0)

def load_database():
    features = np.load("../data/database.npy")
    indicies = np.load("../data/tensor_ids.npy")
    return features, indicies


def top_k(features, indicies, query, k=10):
    query = query.reshape(1, -1)
    features = features.reshape(features.shape[0], -1)

    # Normalize features and query to unit vectors
    features_normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
    query_normalized = query / np.linalg.norm(query)

    # Compute cosine similarity scores
    score = cosine_similarity(query_normalized, features_normalized)
    
    # Get top 2k indices based on scores
    top_indices = np.argsort(score[0])[-2*k:][::-1]

    # Corresponding scores and indices
    top_scores = score[0][top_indices]
    top_indices = indicies[top_indices]

    # Select unique indices, preserving the order of first occurrence
    unique_indices = np.unique(top_indices, return_index=True)[1]
    unique_indices.sort()

    # Get the top k unique indices and corresponding scores
    top_k_indices = top_indices[unique_indices[:k]]
    top_k_scores = top_scores[unique_indices[:k]]
    return top_k_scores, top_k_indices    


def lookup(query, k=10):
    """
    query: a list of fingerprint of the audios
    k: top k results

    """
    features, indicies = load_database()
    results = []
    for q in query:
        q= q.cpu().detach().numpy()
        score, index = top_k(features, indicies, q, k)
        results.append(index)
    return results

if __name__=='__main__':
    wav_file_paths = [r"E:\cs682\data\fma_small_wav\004080.wav"]
    query = fingerprinting(wav_file_paths)
    results = lookup(query)
    print(results)