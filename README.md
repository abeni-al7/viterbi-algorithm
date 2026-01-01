## Contributors

| Name | ID |
|------|----|
| Daniot Mihrete Woldetinsae | UGR/3005/15 |
| Abel Erduino | UGR/6665/15 |
| Abenezer Alebachew | UGR/4429/15 |
| Abraham Dessalegn | UGR/9136/15 |

# Viterbi Algorithm Implementation

This project implements the Viterbi algorithm in Python using NumPy. It is designed to find the most likely sequence of hidden states (the Viterbi path) that results in a sequence of observed events.

## Project Structure

```
viterbi-algorithm/
├── src/
│   ├── __init__.py
│   └── viterbi.py       # Core implementation
├── tests/
│   ├── __init__.py
│   └── test_viterbi.py  # Unit tests
├── requirements.txt     # Dependencies
└── README.md            # Documentation
```

## Installation

1.  Clone the repository.
2.  Create a virtual environment (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can use the `viterbi` function from `src.viterbi`.

```python
import numpy as np
from src.viterbi import viterbi

# Define model parameters
start_probs = np.array([0.6, 0.4])
transition_probs = np.array([[0.7, 0.3], [0.4, 0.6]])
emission_probs = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])

# Define observation sequence (indices)
observations = np.array([0, 1, 2])

# Run Viterbi
path, prob = viterbi(observations, start_probs, transition_probs, emission_probs)

print(f"Most likely path: {path}")
print(f"Probability: {prob}")
```

## Testing

Run the unit tests using `pytest`:

```bash
pytest
```

## Implementation Details

The implementation uses standard probability multiplication.
- **Inputs**: NumPy arrays for observations, start probabilities, transition matrix, and emission matrix.
- **Outputs**: The most likely sequence of state indices and its probability.
- **Complexity**: $O(T \cdot N^2)$, where $T$ is the sequence length and $N$ is the number of states.


