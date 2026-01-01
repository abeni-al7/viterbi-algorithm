import numpy as np
import pytest
from src.viterbi import viterbi


def test_viterbi_rain_sun():
    """
    Tests the Viterbi algorithm with a classic Rain/Sun Hidden Markov Model.
    """
    states = ["Rainy", "Sunny"]

    start_probs = {"Rainy": 0.6, "Sunny": 0.4}

    transition_probs = {
        "Rainy": {"Rainy": 0.7, "Sunny": 0.3},
        "Sunny": {"Rainy": 0.4, "Sunny": 0.6},
    }

    emission_probs = {
        "Rainy": {"Walk": 0.1, "Shop": 0.4, "Clean": 0.5},
        "Sunny": {"Walk": 0.6, "Shop": 0.3, "Clean": 0.1},
    }

    obs_sequence = ["Walk", "Shop", "Clean"]

    best_path, best_prob = viterbi(
        obs_sequence, states, start_probs, transition_probs, emission_probs
    )

    expected_path = ["Sunny", "Rainy", "Rainy"]
    expected_prob = 0.01344

    assert best_path == expected_path
    # Compare log probabilities
    assert np.isclose(best_prob, np.log(expected_prob))


def test_viterbi_empty_sequence():
    """Tests behavior with an empty sequence."""
    states = ["A", "B"]
    start_probs = {"A": 0.5, "B": 0.5}
    transition_probs = {"A": {"A": 0.5, "B": 0.5}, "B": {"A": 0.5, "B": 0.5}}
    emission_probs = {"A": {"O1": 0.5, "O2": 0.5}, "B": {"O1": 0.5, "O2": 0.5}}
    obs_sequence = []

    with pytest.raises(IndexError):
        viterbi(
            obs_sequence,
            states,
            start_probs,
            transition_probs,
            emission_probs,
        )


def test_viterbi_single_observation():
    """Tests with a single observation."""
    states = ["A", "B"]
    start_probs = {"A": 0.5, "B": 0.5}
    transition_probs = {"A": {"A": 0.9, "B": 0.1}, "B": {"A": 0.1, "B": 0.9}}
    emission_probs = {"A": {"O1": 0.9, "O2": 0.1}, "B": {"O1": 0.1, "O2": 0.9}}

    obs_sequence = ["O1"]

    best_path, best_prob = viterbi(
        obs_sequence, states, start_probs, transition_probs, emission_probs
    )

    # P(A) = 0.5 * 0.9 = 0.45
    # P(B) = 0.5 * 0.1 = 0.05
    # Best is A

    assert best_path == ["A"]
    assert np.isclose(best_prob, np.log(0.45))
