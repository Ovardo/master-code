import numpy as np

from master_code.data_association import JCBB_association


def test_jcbb_matches_compatible_measurement() -> None:
    measurements = np.array([[5.0, 0.2]])
    predictions = np.array([[5.0, 0.2]])
    innovation_covariance = np.diag([0.1, 0.01])

    associations = JCBB_association(
        measurements,
        predictions,
        innovation_covariance,
        alpha_individual=0.99,
        alpha_joint=0.99,
    )

    np.testing.assert_array_equal(associations, [0])


def test_jcbb_leaves_measurements_unassociated_without_landmarks() -> None:
    associations = JCBB_association(
        np.array([[2.0, 0.0], [3.0, 0.1]]),
        np.empty((0, 2)),
        np.empty((0, 0)),
        alpha_individual=0.99,
        alpha_joint=0.99,
    )

    np.testing.assert_array_equal(associations, [-1, -1])
