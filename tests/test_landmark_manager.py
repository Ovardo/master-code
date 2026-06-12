import numpy as np

from master_code.landmark_manager import TentativeLandmarkManager


def test_tentative_landmark_is_promoted_after_required_hits() -> None:
    manager = TentativeLandmarkManager(M=2, N=3, gate=0.5)

    confirmed = manager.add_tentative_landmarks(
        current_step=0,
        unassociated_measurements=np.array([[4.0, 0.1]]),
        new_tentative_landmarks=np.array([[1.0, 2.0]]),
    )
    assert confirmed == []
    assert len(manager) == 1

    confirmed = manager.add_tentative_landmarks(
        current_step=1,
        unassociated_measurements=np.array([[4.1, 0.1]]),
        new_tentative_landmarks=np.array([[1.1, 2.0]]),
    )

    assert len(confirmed) == 1
    assert len(manager) == 0
    np.testing.assert_allclose(confirmed[0].position, [1.05, 2.0])
