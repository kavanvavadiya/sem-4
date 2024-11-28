import numpy as np
import time


def vectorized_dct(X: np.ndarray) -> np.ndarray:
    N1, N2 = X.shape
    k1, k2 = np.meshgrid(np.arange(N1), np.arange(N2), indexing='ij')
    cos1 = np.cos(np.pi * (k1 + 0.5) / N1)
    cos2 = np.cos(np.pi * (k2 + 0.5) / N2)
    DCT = 4 * np.sum(X * cos1 * cos2, axis=(0, 1)) / (N1 * N2)
    return DCT
    '''
    @params
        X : np.float64 array of size(m,n)
    return np.float64 array of size(m,n)
    '''
    # TODO
    # return None
    # END TODO


def relevance_one(D: np.ndarray, Q: np.ndarray) -> np.ndarray:
    similarities = []
    for vi in Q:
        max_sim = -np.inf
        for aj in D:
            sim = np.dot(vi, aj)
            if sim > max_sim:
                max_sim = sim
        similarities.append(max_sim)
    relevance = sum(similarities)
    return relevance
    '''
    @params
        D : n x w x k numpy float64 array 
            where each n x w slice represents a document
            with n vectors of length w 
        Q : m x w numpy float64 array
            which represents a query with m vectors of length w

    return np.ndarray of shape (k,) of docIDs sorted in descending order by relevance score
    '''
    # TODO
    # return None
    # END TODO


def relevance_two(D: np.ndarray, Q: np.ndarray) -> np.ndarray:
    '''
    @params
        D : n x w x k numpy float64 array 
            where each n x w slice represents a document
            with n vectors of length w 
        Q : m x w numpy float64 array
            which represents a query with m vectors of length w

    return np.ndarray of shape (k,) of docIDs sorted in descending order by relevance score
    '''
    # Compute pairwise scores between query vectors and document embeddings
    scores = np.zeros(D.shape[0])
    for q_vec in Q:
        max_score = np.sum(np.max(np.maximum(0, (q_vec - D)), axis=1))
        scores = np.vstack((scores, max_score))

    # Remove the initial zero row and sum the scores to obtain relevance score
    relevance_score = np.sum(scores[1:], axis=0)

    # Return the indices of documents sorted by relevance score
    return np.argsort(relevance_score)
    # TODO
    return None
    # END TODO
