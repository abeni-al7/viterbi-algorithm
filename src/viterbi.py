import numpy as np
from typing import List, Tuple


def viterbi(
    observations: np.ndarray,
    start_probs: np.ndarray,
    transition_probs: np.ndarray,
    emission_probs: np.ndarray,
) -> Tuple[List[int], float]:
    """
    Implements the Viterbi algorithm to find the most likely sequence of
    hidden states.

    Args:
        observations (np.ndarray): Array of observation indices of shape (T,).
        start_probs (np.ndarray): Initial state probabilities of shape (N,).
        transition_probs (np.ndarray): Transition probability matrix
                                        of shape (N, N). transition_probs[i, j]
                                        is P(state_j | state_i).
        emission_probs (np.ndarray): Emission probability matrix of
                                     shape (N, M). emission_probs[i, k] is
                                     P(obs_k | state_i).

    Returns:
        Tuple[List[int], float]:
            - The most likely sequence of state indices.
            - The probability of that sequence.

    Raises:
        ValueError: If input dimensions are inconsistent.
    """
    # Validate inputs
    T = observations.shape[0]
    N = start_probs.shape[0]

    if transition_probs.shape != (N, N):
        raise ValueError(
            f"Transition matrix shape {transition_probs.shape}",
            "must be ({N}, {N})",
        )

    # M is the number of possible observations, inferred from emission matrix
    M = emission_probs.shape[1]
    if emission_probs.shape[0] != N:
        raise ValueError(
            f"Emission matrix shape {emission_probs.shape} must be ({N}, {M})"
        )

    # Initialize tables
    # viterbi_table[t, s] stores the max probability of being in state s at
    # time t given the observations up to t and the most likely path ending
    # in s
    viterbi_table = np.zeros((T, N))

    # backpointer[t, s] stores the state at time t-1 that maximized the
    # probability for state s at time t
    backpointer = np.zeros((T, N), dtype=int)

    # 1. Initialization (t=0)
    first_obs = observations[0]
    viterbi_table[0, :] = start_probs * emission_probs[:, first_obs]

    # 2. Recursion
    for t in range(1, T):
        obs = observations[t]
        for s in range(N):
            # Calculate prob of transitioning to state s from each prior state
            # prev_s P(path to prev_s) * P(prev_s -> s) * P(obs | s)
            # Note: emission_probs[s, obs] is constant for the max over prev_s

            trans_probs = viterbi_table[t - 1, :] * transition_probs[:, s]
            max_prev_prob = np.max(trans_probs)

            viterbi_table[t, s] = max_prev_prob * emission_probs[s, obs]
            backpointer[t, s] = np.argmax(trans_probs)

    # 3. Termination
    # The probability of the most likely path
    best_path_prob = np.max(viterbi_table[T - 1, :])
    # The last state of the most likely path
    best_last_state = np.argmax(viterbi_table[T - 1, :])

    # 4. Path Backtracking
    best_path = [best_last_state]
    current_state = best_last_state

    # Loop backwards from T-1 down to 1
    for t in range(T - 1, 0, -1):
        prev_state = backpointer[t, current_state]
        best_path.insert(0, prev_state)
        current_state = prev_state

    return best_path, best_path_prob
