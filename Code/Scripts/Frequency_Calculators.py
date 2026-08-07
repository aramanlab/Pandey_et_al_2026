import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import r2_score
from scipy.stats import pearsonr


### --- General helpers ------
def Alphabet_list(Compact_dataset):
    """Returns the set of unique alphabets in a given compact representation dataset"""
    Alphabets = np.unique(Compact_dataset)
    return Alphabets 


def One_hot_decoder(state, alphabets):
    """Input: -Takes a state in one-hot encoded representation
              - Take the array of unique alphabets
    - returns the compact representation of the state"""
    step = len(alphabets)
    Compact_state = []
    for i in range(0, len(state), step):
        start = i
        end = i + step
        slice_ = state[start:end]
        pos = np.argwhere(slice_ == 1).reshape(-1)[0]
        letter = alphabets[pos]
        Compact_state.append(letter)
    return np.array(Compact_state)


def One_hot_encoder(state, alphabets):
    """Input: -Takes a state in compact representation
              - Take the array of unique alphabets
    - returns the one hot encoded representation of the state"""
    step = len(alphabets)
    unfurled_state = []
    for i in state:
        vector_i = (alphabets == i) * 1
        unfurled_state.append(vector_i)
    return np.concatenate(unfurled_state)


def One_hot_encoder_Array(Compact_Array, alphabets):
    """One hot encoder of an array in compact representation"""
    OHE_Array = []
    for state in Compact_Array:
        ohe_state = One_hot_encoder(state, alphabets)
        OHE_Array.append(ohe_state)
    return np.array(OHE_Array)

def One_hot_decoder_Array(OHE_Array, alphabets):
    """One hot decoder of an array in one-hot encoded representation"""
    Compact_Array = []
    for state in OHE_Array:
        ohe_state = One_hot_decoder(state, alphabets)
        Compact_Array.append(ohe_state)
    return np.array(Compact_Array)

def Integer_encoder(state, alphabets):
    """Residue-space (non-one-hot) encoding: each sequence -> array of ints in [0, q-1].
    This is the representation the fast pairwise/triplet routines below expect.
    `alphabets` should be an array/list; index lookup uses a dict for O(1) mapping.
    - This is necessary because we can now exploit numpy binning functions on integers.
    - For words and residues, there is no such binning function. But integers are ordered."""
    letter_to_idx = {a: i for i, a in enumerate(alphabets)}
    return np.array([letter_to_idx[c] for c in state])


def Integer_encode_Array(Compact_Array, alphabets):
    """Takes compact representation array, returns integer encoded array"""
    return np.array([Integer_encoder(seq, alphabets) for seq in Compact_Array])


def Hamming(seq, Natuals_Array):
    """Returns Hamming distance between a sequence and the Natural Array (Training corpus)"""
    diff = np.sum(np.abs((Natuals_Array - seq)) * 0.5, axis=1)
    d_hamm = np.min(diff)
    return int(d_hamm)


###### --- Frequency calculation ----

def Singlet_Frequency(One_hot_encoded, Alphabet_size):
    """Singlet Frequency is just per-site per residue mean value.
    - we can use numpy mean over one-hot encoded representation"""
    Mean_array = np.mean(One_hot_encoded, axis=0)
    return Mean_array


def Pairwise_Frequency(One_hot_encoded, Alphabet_size):
    """Vectorized calculation of the full one-hot joint-frequency matrix is just
    X.T @ X / N -- a single matmul instead of a Python counter loop over (a,b) pairs.
    Returns the same (frequencies, Matrix) shape/order as the original.
    """
    num_seq, length_seq = One_hot_encoded.shape
    num_position = length_seq // Alphabet_size

    Matrix = (One_hot_encoded.T @ One_hot_encoded) / num_seq

    # zero out same-position blocks (original loop only ever did i < j)
    for i in range(num_position):
        s = slice(i * Alphabet_size, (i + 1) * Alphabet_size)
        Matrix[s, s] = 0

    frequencies = []
    for i in range(num_position):
        for j in range(i + 1, num_position):
            block = Matrix[i * Alphabet_size:(i + 1) * Alphabet_size,
                           j * Alphabet_size:(j + 1) * Alphabet_size]
            frequencies.append(block.flatten())
    frequencies = np.concatenate(frequencies) if frequencies else np.array([])

    return frequencies, Matrix


def Pairwise_Correlation(One_hot_encoded, Alphabet_size):
    """Vectorized Correlation calculation= F_ij - outer(f, f),
      computed for the whole matrix at once."""
    Single_freq = Singlet_Frequency(One_hot_encoded, Alphabet_size)
    _, F_ij = Pairwise_Frequency(One_hot_encoded, Alphabet_size)
    num_position = One_hot_encoded.shape[1] // Alphabet_size

    Matrix = F_ij - np.outer(Single_freq, Single_freq)
    for i in range(num_position):
        s = slice(i * Alphabet_size, (i + 1) * Alphabet_size)
        Matrix[s, s] = 0

    Correlations = []
    for i in range(num_position):
        for j in range(i + 1, num_position):
            block = Matrix[i * Alphabet_size:(i + 1) * Alphabet_size,
                           j * Alphabet_size:(j + 1) * Alphabet_size]
            Correlations.append(block.flatten())
    Correlations = np.concatenate(Correlations) if Correlations else np.array([])

    return Correlations, Matrix


def Triplet_Frequency(Int_encoded, Alphabet_size):
    """ Computes Triplet frequency given Integer encoded representation of sequence
    - Faster than operating using one-hot encoded representation 
    since we can use bincounts and einsum operators"""
    num_seq, num_position = Int_encoded.shape
    q = Alphabet_size

    if num_position < 3:
        """Sequences have less than 3 positions. Can't have triplet frequency"""
        return np.array([])

    frequencies = []
    for i in range(num_position):
        for j in range(i + 1, num_position):
            for k in range(j + 1, num_position):
                code = Int_encoded[:, i] * q * q + Int_encoded[:, j] * q + Int_encoded[:, k]
                counts = np.bincount(code, minlength=q ** 3).astype(float) / num_seq
                frequencies.append(counts)
    return np.concatenate(frequencies)


def Triplet_Correlation(Int_encoded, One_hot_encoded, Alphabet_size):
    """Computes Triplet correlation given Integer encoded representation of sequence
    - Faster than operating using one-hot encoded representation 
    since we can use bincounts and einsum operators"""

    q = Alphabet_size
    num_seq, num_position = Int_encoded.shape

    if num_position < 3:
        """Sequences have less than 3 positions. Can't have triplet correlations"""
        return np.array([])

    f = Singlet_Frequency(One_hot_encoded, Alphabet_size).reshape(num_position, q)
    _, PW_corr_matrix = Pairwise_Correlation(One_hot_encoded, Alphabet_size)

    frequencies = []
    for i in range(num_position):
        for j in range(i + 1, num_position):
            for k in range(j + 1, num_position):
                code = Int_encoded[:, i] * q * q + Int_encoded[:, j] * q + Int_encoded[:, k]
                F_ijk = (np.bincount(code, minlength=q ** 3).astype(float) / num_seq).reshape(q, q, q)

                fi, fj, fk = f[i], f[j], f[k]
                singlet_contrib = np.einsum('a,b,c->abc', fi, fj, fk)

                Cij = PW_corr_matrix[i * q:(i + 1) * q, j * q:(j + 1) * q]
                Cjk = PW_corr_matrix[j * q:(j + 1) * q, k * q:(k + 1) * q]
                Cik = PW_corr_matrix[i * q:(i + 1) * q, k * q:(k + 1) * q]

                pairwise_contrib = (
                    np.einsum('ab,c->abc', Cij, fk) +
                    np.einsum('bc,a->abc', Cjk, fi) +
                    np.einsum('ac,b->abc', Cik, fj))

                stat = F_ijk - singlet_contrib - pairwise_contrib
                frequencies.append(stat.flatten())

    return np.concatenate(frequencies)



def Triplet_Correlation_Slow(One_hot_encoded, Alphabet_size):

    """ Uses the one-hot encoded representation. Hence it needs to loop over positions and residues (SLOW!)

        Input - 
        One_hot_encoded : Takes in One hot encoded dataset
        Alphabet_size : Number of alphabets that could go in any position (internal deg of freedom)

        Output- 
        triplet correlations in the given dataset
        """
    
    Single_freq = Singlet_Frequency(One_hot_encoded, Alphabet_size)
    _, PW_corr_matrix = Pairwise_Correlation(One_hot_encoded, Alphabet_size)

    num_seq, length_seq = np.shape(One_hot_encoded)
    num_position = length_seq//Alphabet_size
    frequencies = []
    for i in range(0, num_position):
        for j in range(i+1, num_position):
            for k in range(j+1, num_position):
                freq_ijk = np.zeros((Alphabet_size, Alphabet_size, Alphabet_size))
                for s_i in range(i*Alphabet_size, (i+1) * Alphabet_size):
                    for s_j in range(j*Alphabet_size, (j+1) * Alphabet_size):
                        for s_k in range(k*Alphabet_size, (k+1) * Alphabet_size):
                            vec_i = One_hot_encoded[:,s_i]
                            vec_j = One_hot_encoded[:,s_j]
                            vec_k = One_hot_encoded[:,s_k]
                            Singlet_contrib = Single_freq[s_i] * Single_freq[s_j] * Single_freq[s_k]
                            Pairwise_contrib = PW_corr_matrix[s_i,s_j] * Single_freq[s_k] + PW_corr_matrix[s_j,s_k] * Single_freq[s_i] + PW_corr_matrix[s_i,s_k] * Single_freq[s_j]  
                            Stat = (np.sum(vec_i * vec_j * vec_k)/ num_seq) - Singlet_contrib - Pairwise_contrib

                            frequencies.append(Stat)

    return frequencies