"""Reconstruct ground-truth data associations for the simulated SLAM dataset.

``data/simulated/simulatedSLAM.mat`` ships the ground-truth landmarks and poses
but *not* the association between each noisy range-bearing measurement and the
landmark that produced it. Because we know the true pose at every step, we can
recover that association: predict the noise-free measurement of every landmark
from the true pose and assign each measurement to the closest prediction.

Matching is done in *measurement space* with a Mahalanobis distance using the
known measurement noise (range / bearing). This is more principled than nearest
neighbour in world coordinates because bearing noise turns into a range-scaled
lateral error when back-projected, so a fixed Cartesian gate would be too tight
for far landmarks and too loose for near ones. The squared Mahalanobis distance
of a correct association is chi-square distributed with 2 dof, which both
validates the result and gives a natural gate for rejecting clutter.

For this dataset the matching is essentially unambiguous (closest landmark pair
is ~2 m apart and >99 % of measurements have a single candidate within the
gate), so per-measurement nearest neighbour recovers the true association.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from master_code.config import SlamConfig
from master_code.loaders.simulated import SimulatedDataLoader
from master_code.paths import FIGURES_ROOT
from master_code.utils import ssa

# --- configuration ----------------------------------------------------------
# Measurement noise used to weight the range/bearing residuals. Defaults to the
# values from the simulated SLAM config so the chi-square gate is meaningful.
_SIM_NOISE = SlamConfig.load("configs/default_sim.yaml").noise
SIGMA_RANGE: float = _SIM_NOISE.sigma_range            # [m]
SIGMA_BEARING_RAD: float = _SIM_NOISE.sigma_bearing_rad  # [rad]

# Chi-square gate (2 dof). Measurements whose best Mahalanobis distance exceeds
# the gate are labelled as clutter (-1). Set GATE_ALPHA = None to disable.
GATE_ALPHA: float | None = 0.999



CLUTTER_LABEL = -1


def expected_measurements(pose: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Noise-free range-bearing of every landmark seen from ``pose``.

    Args:
        pose: ``[x, y, theta]`` of the sensor in the world frame.
        landmarks: ``(M, 2)`` landmark positions in the world frame.

    Returns:
        ``(M, 2)`` array of ``[range, bearing]``, bearing wrapped to (-pi, pi].
    """
    x, y, theta = pose
    delta = landmarks - np.array([x, y])
    ranges = np.hypot(delta[:, 0], delta[:, 1])
    bearings = ssa(np.arctan2(delta[:, 1], delta[:, 0]) - theta)
    return np.column_stack([ranges, bearings])


def associate_step(
    measurements: np.ndarray,
    pose: np.ndarray,
    landmarks: np.ndarray,
    sigma_range: float = SIGMA_RANGE,
    sigma_bearing_rad: float = SIGMA_BEARING_RAD,
    gate: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Associate one step of measurements to ground-truth landmarks.

    Args:
        measurements: ``(N, 2)`` array of ``[range, bearing]`` measurements.
        pose: true ``[x, y, theta]`` pose at this step.
        landmarks: ``(M, 2)`` ground-truth landmark positions.
        gate: squared-Mahalanobis gate; matches above it are labelled clutter.

    Returns:
        ``labels``: ``(N,)`` int array of landmark indices, ``-1`` for clutter.
        ``maha``: ``(N,)`` squared Mahalanobis distance to the assigned landmark.
    """
    measurements = np.asarray(measurements, dtype=float).reshape(-1, 2)
    if measurements.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    predicted = expected_measurements(pose, landmarks)  # (M, 2)

    d_range = measurements[:, 0, None] - predicted[None, :, 0]      # (N, M)
    d_bearing = ssa(measurements[:, 1, None] - predicted[None, :, 1])
    maha = (d_range / sigma_range) ** 2 + (d_bearing / sigma_bearing_rad) ** 2

    labels = maha.argmin(axis=1)
    best = maha[np.arange(len(labels)), labels]

    if gate is not None:
        labels = np.where(best <= gate, labels, CLUTTER_LABEL)

    return labels.astype(int), best


def build_ground_truth_associations(
    loader: SimulatedDataLoader,
    sigma_range: float = SIGMA_RANGE,
    sigma_bearing_rad: float = SIGMA_BEARING_RAD,
    gate_alpha: float | None = GATE_ALPHA,
    max_steps: int | None = None,
) -> list[np.ndarray]:
    """Reconstruct the ground-truth association for every measurement.

    Returns a list with one int array per step; entry ``i`` is the index into
    ``loader.landmarks_gt`` of the landmark that produced measurement ``i`` (or
    ``-1`` if it was gated out as clutter). Measurement step ``k`` is taken at
    ``loader.poses_gt[k]``.
    """
    gate = None if gate_alpha is None else float(chi2.ppf(gate_alpha, df=2))

    stop = len(loader.measurements)
    if max_steps is not None:
        stop = min(stop, max_steps)

    associations: list[np.ndarray] = []
    for k in range(stop):
        labels, _ = associate_step(
            loader.measurements[k],
            loader.poses_gt[k],
            loader.landmarks_gt,
            sigma_range=sigma_range,
            sigma_bearing_rad=sigma_bearing_rad,
            gate=gate,
        )
        associations.append(labels)
    return associations


def association_diagnostics(
    loader: SimulatedDataLoader,
    associations: list[np.ndarray],
    sigma_range: float = SIGMA_RANGE,
    sigma_bearing_rad: float = SIGMA_BEARING_RAD,
) -> None:
    """Print sanity checks: chi-square fit, clutter count, double assignments."""
    maha_all: list[float] = []
    n_clutter = 0
    n_total = 0
    n_double = 0  # a landmark assigned to >1 measurement within the same step

    for k, labels in enumerate(associations):
        n_total += labels.size
        n_clutter += int(np.sum(labels == CLUTTER_LABEL))

        valid = labels[labels != CLUTTER_LABEL]
        if valid.size:
            _, counts = np.unique(valid, return_counts=True)
            n_double += int(np.sum(counts > 1))

        _, best = associate_step(
            loader.measurements[k], loader.poses_gt[k], loader.landmarks_gt,
            sigma_range=sigma_range, sigma_bearing_rad=sigma_bearing_rad,
        )
        maha_all.extend(best.tolist())

    maha = np.asarray(maha_all)
    print(f"measurements            : {n_total}")
    print(f"landmarks               : {len(loader.landmarks_gt)}")
    print(f"labelled clutter (-1)   : {n_clutter} ({100 * n_clutter / max(n_total, 1):.2f} %)")
    print(f"landmark double-assigned: {n_double} (per-step, expected 0 without clutter)")
    print(
        "Mahalanobis^2 (chi2, 2 dof): "
        f"median={np.median(maha):.2f} (theory 1.39), "
        f"mean={maha.mean():.2f} (theory 2.00), "
        f"p95={np.percentile(maha, 95):.2f} (theory 5.99)"
    )


def plot_associations(
    loader: SimulatedDataLoader,
    associations: list[np.ndarray],
    max_steps: int | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Back-project measurements through the true pose, coloured by association."""
    stop = len(associations) if max_steps is None else min(len(associations), max_steps)

    points: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for k in range(stop):
        z = loader.measurements[k]
        if len(z) == 0:
            continue
        x, y, theta = loader.poses_gt[k]
        angle = theta + z[:, 1]
        points.append(np.column_stack([x + z[:, 0] * np.cos(angle), y + z[:, 0] * np.sin(angle)]))
        labels.append(associations[k])

    points = np.vstack(points)
    labels = np.concatenate(labels)
    clutter = labels == CLUTTER_LABEL

    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
    ax.plot(loader.poses_gt[:stop + 1, 0], loader.poses_gt[:stop + 1, 1],
            color="0.4", lw=0.8)
    ax.scatter(points[~clutter, 0], points[~clutter, 1], c=labels[~clutter],
               cmap="tab20", s=6, alpha=0.5, label="Back-projected measurements")
    if clutter.any():
        ax.scatter(points[clutter, 0], points[clutter, 1], marker="x",
                   color="red", s=30, label="Gated out")
    ax.scatter(loader.landmarks_gt[:, 0], loader.landmarks_gt[:, 1],
               marker="*", s=90, edgecolor="k", facecolor="gold",
               linewidths=0.5, label="Ground-truth landmarks", zorder=5)

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    # ax.set_title("Reconstructed ground-truth associations")
    # ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    return fig, ax


def main() -> None:
    loader = SimulatedDataLoader()
    associations = build_ground_truth_associations(loader)

    association_diagnostics(loader, associations)
    fig, ax = plot_associations(loader, associations)
    plt.show()
    fig.savefig(FIGURES_ROOT / 'hello.pdf')


if __name__ == "__main__":
    main()
