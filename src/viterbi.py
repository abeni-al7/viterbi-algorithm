import numpy as np
from typing import List, Dict, Tuple


def log_prob(prob: float) -> float:
    """
    Compute the logarithm of a probability safely.

    Args:
        prob (float): Probability value.

    Returns:
        float: Log probability. Returns -inf if prob is 0.
    """
    if prob == 0:
        return -np.inf
    return np.log(prob)


def initialize_tables(
    n_states: int,
    n_obs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize the Viterbi tables (delta and psi).

    Args:
        n_states (int): Number of hidden states.
        n_obs (int): Number of observations in the sequence.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - delta: Probability table of shape (n_states, n_obs).
            - psi: Backpointer table of shape (n_states, n_obs).
    """
    delta = np.full((n_states, n_obs), -np.inf)
    psi = np.zeros((n_states, n_obs), dtype=int)
    return delta, psi


def viterbi(
    obs: List[str],
    states: List[str],
    start_p: Dict[str, float],
    trans_p: Dict[str, Dict[str, float]],
    emit_p: Dict[str, Dict[str, float]],
) -> Tuple[List[str], float]:
    """
    Viterbi Algorithm for HMM decoding.

    Args:
        obs (List[str]): Sequence of observations.
        states (List[str]): List of all possible states.
        start_p (Dict[str, float]): Initial state probabilities.
        trans_p (Dict[str, Dict[str, float]]): Transition probabilities.
        emit_p (Dict[str, Dict[str, float]]): Emission probabilities.

    Returns:
        Tuple[List[str], float]:
            - Best path (sequence of states).
            - Log probability of the best path.
    """

    # Convert parameters to log-space NumPy arrays
    N = len(states)
    T = len(obs)

    start_log_p = np.array([log_prob(start_p.get(s, 0.0)) for s in states])

    trans_log_p = np.full((N, N), -np.inf)
    for i, s_from in enumerate(states):
        for j, s_to in enumerate(states):
            # Use get with default 0.0 to handle missing transitions safely
            prob = trans_p.get(s_from, {}).get(s_to, 0.0)
            trans_log_p[i, j] = log_prob(prob)

    emit_log_p = np.full((N, T), -np.inf)
    for i, s in enumerate(states):
        for t, o in enumerate(obs):
            # Use get with default 0.0
            prob = emit_p.get(s, {}).get(o, 0.0)
            emit_log_p[i, t] = log_prob(prob)

    # Initialize tables
    delta, psi = initialize_tables(N, T)

    # Initialization step (t=0)
    # delta[s, 0] = start_p[s] * emit_p[s][obs[0]]
    # In log space: log(start_p[s]) + log(emit_p[s][obs[0]])
    delta[:, 0] = start_log_p + emit_log_p[:, 0]

    # Recursion step
    for t in range(1, T):
        for s in range(N):
            # Calculate prob of reaching state s at time t from all possible
            # prev states
            # delta[prev_s, t-1] * trans_p[prev_s, s] * emit_p[s][obs[t]]
            # Log space:
            # delta[prev_s, t-1] + log(trans_p[prev_s, s]) + log(emit_p[s][obs[t]])

            # We want to find max over prev_s
            # trans_log_p[:, s] is the column for transitioning TO s from all
            # prev states
            probs = delta[:, t - 1] + trans_log_p[:, s]
            best_prev_s = np.argmax(probs)

            delta[s, t] = probs[best_prev_s] + emit_log_p[s, t]
            psi[s, t] = best_prev_s

    # Termination
    best_last_state = np.argmax(delta[:, T - 1])
    best_path_prob = delta[best_last_state, T - 1]

    # Path reconstruction (Backtracking)
    best_path_indices = [best_last_state]
    for t in range(T - 1, 0, -1):
        best_prev_s = psi[best_path_indices[-1], t]
        best_path_indices.append(best_prev_s)

    best_path_indices.reverse()

    best_path = [states[i] for i in best_path_indices]

    return best_path, best_path_prob
