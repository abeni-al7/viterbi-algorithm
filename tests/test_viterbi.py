import numpy as np
import pytest
from src.viterbi import viterbi


def test_viterbi_rain_sun():
    """
    Tests the Viterbi algorithm with a classic Rain/Sun Hidden Markov Model.

    States: 0=Rainy, 1=Sunny
    Observations: 0=Walk, 1=Shop, 2=Clean
    """

    start_probs = np.array([0.6, 0.4])

    # Transition Matrix: rows=from, cols=to
    # R->R=0.7, R->S=0.3
    # S->R=0.4, S->S=0.6
    transition_probs = np.array([[0.7, 0.3], [0.4, 0.6]])

    # Emission Matrix: rows=state, cols=observation
    # R->W=0.1, R->S=0.4, R->C=0.5
    # S->W=0.6, S->S=0.3, S->C=0.1
    emission_probs = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])

    # 2. Define Observation Sequence
    # Walk, Shop, Clean
    obs_sequence = np.array([0, 1, 2])

    # 3. Run Viterbi
    best_path, best_prob = viterbi(
        obs_sequence, start_probs, transition_probs, emission_probs
    )

    # 4. Assertions
    # Expected calculation:
    # t=0 (Walk):
    #   P(R) = 0.6 * 0.1 = 0.06
    #   P(S) = 0.4 * 0.6 = 0.24
    # t=1 (Shop):
    #   P(R) = max(0.06*0.7, 0.24*0.4) * 0.4 = 0.096 * 0.4 = 0.0384 (from S)
    #   P(S) = max(0.06*0.3, 0.24*0.6) * 0.3 = 0.144 * 0.3 = 0.0432 (from S)
    # t=2 (Clean):
    #   P(R) = max(0.0384*0.7, 0.0432*0.4) * 0.5 = 0.02688 * 0.5 = 0.01344
    #   (from R)
    #   P(S) = max(0.0384*0.3, 0.0432*0.6) * 0.1 = 0.02592 * 0.1 = 0.002592
    #   (from S)

    expected_path = [1, 0, 0]  # Sunny, Rainy, Rainy
    expected_prob = 0.01344

    assert best_path == expected_path
    assert np.isclose(best_prob, expected_prob)


def test_viterbi_empty_sequence():
    """Tests behavior with an empty sequence (should handle
    gracefully or raise error)."""

    start_probs = np.array([0.5, 0.5])
    transition_probs = np.array([[0.5, 0.5], [0.5, 0.5]])
    emission_probs = np.array([[0.5, 0.5], [0.5, 0.5]])
    obs_sequence = np.array([])

    with pytest.raises(IndexError):
        viterbi(obs_sequence, start_probs, transition_probs, emission_probs)


def test_viterbi_single_observation():
    """Tests with a single observation."""
    start_probs = np.array([0.5, 0.5])
    transition_probs = np.array([[0.9, 0.1], [0.1, 0.9]])
    emission_probs = np.array([[0.9, 0.1], [0.1, 0.9]])

    # Observation 0
    obs_sequence = np.array([0])

    best_path, best_prob = viterbi(
        obs_sequence, start_probs, transition_probs, emission_probs
    )

    assert best_path == [0]
    assert np.isclose(best_prob, 0.45)
