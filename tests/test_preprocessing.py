import numpy as np

from master_code.preprocessing import relative_pose


def test_relative_pose_for_straight_motion() -> None:
    pose = relative_pose(vel_e=2.0, steer=0.0, dt=0.5)

    np.testing.assert_allclose(
        [pose.x(), pose.y(), pose.theta()],
        [1.0, 0.0, 0.0],
        atol=1e-12,
    )
