import numpy as np


def nearest_indices(query_times: np.ndarray, reference_times: np.ndarray) -> np.ndarray:
    """Return index of the nearest reference time for each query time."""
    insertion_indices = np.searchsorted(reference_times, query_times)
    right_indices = np.clip(insertion_indices, 0, len(reference_times) - 1)
    left_indices = np.clip(insertion_indices - 1, 0, len(reference_times) - 1)

    left_dt = np.abs(query_times - reference_times[left_indices])
    right_dt = np.abs(query_times - reference_times[right_indices])

    return np.where(left_dt <= right_dt, left_indices, right_indices)