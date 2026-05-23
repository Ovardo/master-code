from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hvplot.pandas  # noqa: F401
import holoviews as hv
from holoviews import streams
import numpy as np
import pandas as pd
import panel as pn

from experiment_results import ExperimentResult, load_result
from history import SLAMHistory
from slam_types import SLAMHistoryEntry


pn.extension('tabulator', design='material', template='material', loading_indicator=True)


RESULTS_DIR = Path(__file__).parents[1] / "results" / "experiments"
COVARIANCE_EXTRACTION_EVENT = "extract_covariance.joint_marginal_covariance"


@dataclass(slots=True)
class VisualizationState:
    """Current experiment selection and derived view settings."""

    result: ExperimentResult | None = None
    selected_path: Path | None = None
    history_step_indices: list[int] | None = None
    default_xlim: tuple[float, float] | None = None
    default_ylim: tuple[float, float] | None = None
    view_xlim: tuple[float, float] | None = None
    view_ylim: tuple[float, float] | None = None

def compute_plot_limits(history: SLAMHistory, padding: float = 5.0) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute stable axis limits across the full experiment history."""

    xy_blocks: list[np.ndarray] = []

    for entry in history.all_entries():
        estimate = entry.step_output.estimate

        if estimate.robot_poses is not None and len(estimate.robot_poses) > 0:
            xy_blocks.append(np.asarray(estimate.robot_poses)[:, :2])

        if estimate.landmark_positions is not None and len(estimate.landmark_positions) > 0:
            xy_blocks.append(np.asarray(estimate.landmark_positions)[:, :2])

    if not xy_blocks:
        return (-10.0, 10.0), (-10.0, 10.0)

    xy = np.vstack(xy_blocks)
    min_xy = xy.min(axis=0) - padding
    max_xy = xy.max(axis=0) + padding

    return (float(min_xy[0]), float(max_xy[0])), (float(min_xy[1]), float(max_xy[1]))


def build_step_plot(entry: SLAMHistoryEntry, state: VisualizationState):
    """Create an hvPlot overlay for the selected SLAM step."""

    estimate = entry.step_output.estimate
    plots = []

    robot_poses = estimate.robot_poses
    if robot_poses is not None and len(robot_poses) > 0:
        poses_df = pd.DataFrame(robot_poses, columns=["x", "y", "theta"])
        poses_df["pose_index"] = np.arange(len(poses_df))

        plots.append(
            poses_df.hvplot.line(
                x="x",
                y="y",
                color="#d62728",
                line_width=3,
                label="Estimated trajectory",
            )
        )

        current_pose = robot_poses[-1]
        current_pose_df = pd.DataFrame(
            {
                "x": [float(current_pose[0])],
                "y": [float(current_pose[1])],
            }
        )
        marker_angle = float(current_pose[2] - np.pi / 2.0)

        def _rotate_pose_marker(plot, _element) -> None:
            renderer = plot.state.renderers[-1]
            renderer.glyph.angle = marker_angle

        current_pose_marker = hv.Points(
            current_pose_df,
            kdims=["x", "y"],
            label="Current pose",
        ).opts(
            marker="triangle",
            size=16,
            color="#8c1d18",
            line_color="#8c1d18",
            alpha=0.9,
            hooks=[_rotate_pose_marker],
        )
        plots.append(current_pose_marker)

    landmark_positions = estimate.landmark_positions
    if landmark_positions is not None and len(landmark_positions) > 0:
        landmarks_df = pd.DataFrame(landmark_positions, columns=["x", "y"])
        landmarks_df["landmark_id"] = np.arange(len(landmarks_df))
        predicted_landmark_ids = entry.step_output.measurement_prediction.predicted_landmark_ids

        if predicted_landmark_ids is not None and len(predicted_landmark_ids) > 0:
            local_map_mask = landmarks_df["landmark_id"].isin(predicted_landmark_ids)
            local_landmarks_df = landmarks_df[local_map_mask]
            global_landmarks_df = landmarks_df[~local_map_mask]
        else:
            local_landmarks_df = landmarks_df.iloc[0:0]
            global_landmarks_df = landmarks_df

        if len(global_landmarks_df) > 0:
            plots.append(
                global_landmarks_df.hvplot.scatter(
                    x="x",
                    y="y",
                    color="#1f77b4",
                    size=70,
                    marker="circle",
                    alpha=0.8,
                    label="Estimated landmarks",
                )
            )

        if len(local_landmarks_df) > 0:
            plots.append(
                local_landmarks_df.hvplot.scatter(
                    x="x",
                    y="y",
                    color="#2ca02c",
                    size=85,
                    marker="circle",
                    alpha=0.9,
                    label="Local-map landmarks",
                )
            )

    if not plots:
        return pn.pane.Markdown("No estimated poses or landmarks are available for this step.")

    plot = plots[0]
    for overlay in plots[1:]:
        plot *= overlay

    xlim = state.view_xlim if state.view_xlim is not None else (-10.0, 10.0)
    ylim = state.view_ylim if state.view_ylim is not None else (-10.0, 10.0)

    plot = plot.opts(
        width=800,
        height=450,
        xlabel="x [m]",
        ylabel="y [m]",
        title=f"SLAM estimate at step {entry.step_index}",
        xlim=xlim,
        ylim=ylim,
        framewise=False,
        responsive=False,
        show_grid=True,
        legend_position="top_left",
        aspect="equal",
    )

    range_stream = streams.RangeXY(source=plot)

    def _store_view_range(x_range, y_range, **_kwargs) -> None:
        if x_range is not None:
            state.view_xlim = (float(x_range[0]), float(x_range[1]))
        if y_range is not None:
            state.view_ylim = (float(y_range[0]), float(y_range[1]))

    range_stream.add_subscriber(_store_view_range)
    return plot


def build_covariance_runtime_plot(result: ExperimentResult, current_measurement_step: int | None):
    """Plot covariance extraction runtime and cumulative runtime by measurement step."""

    if result.profiler is None:
        return pn.pane.Markdown("No profiler data is available for this experiment.")

    history_step_to_measurement_step = {
        int(step_index): measurement_step
        for measurement_step, step_index in enumerate(result.history.step_indices, start=1)
    }
    runtime_rows = []
    for row in result.profiler.to_rows(COVARIANCE_EXTRACTION_EVENT):
        iteration = row["iteration"]
        if iteration is None:
            continue

        measurement_step = history_step_to_measurement_step.get(int(iteration))
        if measurement_step is None:
            continue

        runtime_rows.append(
            {
                "measurement_step": measurement_step,
                "elapsed_ms": row["elapsed_ms"],
                "cumulative_ms": row["cumulative_ms"],
            }
        )

    if not runtime_rows:
        return pn.pane.Markdown(
            "No covariance extraction runtime samples were recorded for this experiment."
        )

    runtime_df = pd.DataFrame(runtime_rows)
    runtime_df = runtime_df.sort_values("measurement_step")
    runtime_df["measurement_step"] = runtime_df["measurement_step"].astype(int)

    runtime_plot = runtime_df.hvplot.line(
        x="measurement_step",
        y="elapsed_ms",
        color="#ff7f0e",
        line_width=2,
        responsive=True,
        min_height=200,
        xlabel="Measurement Step",
        ylabel="Runtime [ms]",
        title="Covariance Extraction Runtime per Measurement Step",
    ).opts(
        show_grid=True,
        tools=["hover", "tap"],
    )

    if current_measurement_step is not None:
        runtime_plot *= hv.VLine(current_measurement_step).opts(
            color="#8c1d18",
            line_width=2,
            line_dash="dashed",
        )

    cumulative_plot = runtime_df.hvplot.line(
        x="measurement_step",
        y="cumulative_ms",
        color="#9467bd",
        line_width=2,
        responsive=True,
        min_height=260,
        xlabel="Measurement Step",
        ylabel="Cumulative Runtime [ms]",
        title="Cumulative Covariance Extraction Runtime",
    ).opts(
        show_grid=True,
        tools=["hover", "tap"],
    )

    if current_measurement_step is not None:
        cumulative_plot *= hv.VLine(current_measurement_step).opts(
            color="#8c1d18",
            line_width=2,
            line_dash="dashed",
        )

    tap_stream = streams.Tap(source=runtime_plot, x=None, y=None)
    cumulative_tap_stream = streams.Tap(source=cumulative_plot, x=None, y=None)

    def _seek_iteration(x: float | None, y: float | None) -> None:
        del y
        if x is None:
            return

        measurement_steps = runtime_df["measurement_step"].to_numpy(dtype=float)
        nearest_measurement_step = int(
            runtime_df.iloc[int(np.argmin(np.abs(measurement_steps - x)))]["measurement_step"]
        )

        if 1 <= nearest_measurement_step <= player.end and player.value != nearest_measurement_step:
            player.value = nearest_measurement_step

    tap_stream.add_subscriber(_seek_iteration)
    cumulative_tap_stream.add_subscriber(_seek_iteration)

    return pn.Column(runtime_plot, cumulative_plot, sizing_mode="stretch_width")


def build_local_landmark_count_plot(result: ExperimentResult, current_measurement_step: int | None):
    """Plot the number of local-map landmarks per measurement step."""

    history_rows: list[dict[str, int]] = []
    for measurement_step, entry in enumerate(result.history.all_entries(), start=1):
        predicted_landmark_ids = entry.step_output.measurement_prediction.predicted_landmark_ids
        local_landmark_count = int(len(predicted_landmark_ids)) if predicted_landmark_ids is not None else 0
        history_rows.append(
            {
                "measurement_step": measurement_step,
                "local_landmark_count": local_landmark_count,
            }
        )

    if not history_rows:
        return pn.pane.Markdown("No SLAM history is available for this experiment.")

    history_df = pd.DataFrame(history_rows).sort_values("measurement_step")

    plot = history_df.hvplot.line(
        x="measurement_step",
        y="local_landmark_count",
        color="#2ca02c",
        line_width=2,
        responsive=True,
        min_height=260,
        xlabel="Measurement Step",
        ylabel="Local Landmarks",
        title="Number of Local-Map Landmarks per Measurement Step",
    ).opts(
        show_grid=True,
        tools=["hover", "tap"],
    )

    if current_measurement_step is not None:
        plot *= hv.VLine(current_measurement_step).opts(
            color="#8c1d18",
            line_width=2,
            line_dash="dashed",
        )

    tap_stream = streams.Tap(source=plot, x=None, y=None)

    def _seek_iteration(x: float | None, y: float | None) -> None:
        del y
        if x is None:
            return

        measurement_steps = history_df["measurement_step"].to_numpy(dtype=float)
        nearest_measurement_step = int(
            history_df.iloc[int(np.argmin(np.abs(measurement_steps - x)))]["measurement_step"]
        )

        if 1 <= nearest_measurement_step <= player.end and player.value != nearest_measurement_step:
            player.value = nearest_measurement_step

    tap_stream.add_subscriber(_seek_iteration)
    return plot


state = VisualizationState()

file_select = pn.widgets.Select(
    name="Choose a file to analyze",
    options={
        "No file": None,
        **{file_path.name: file_path for file_path in sorted(RESULTS_DIR.glob("*.pkl"))},
    },
    value=None,
)

player = pn.widgets.Player(
    name="Measurement step",
    start=1,
    end=0,
    value=1,
    step=1,
    loop_policy="loop",
    disabled=True,
    sizing_mode="stretch_width",
)

experiment_info = pn.pane.Markdown("Select an experiment result to begin.")
step_info = pn.pane.Markdown("No measurement step selected.")


def update_selected_file(event) -> None:
    """Load the selected experiment and synchronize the player widget."""

    selected_path = event.new

    if selected_path is None:
        state.result = None
        state.selected_path = None
        state.history_step_indices = None
        state.default_xlim = None
        state.default_ylim = None
        state.view_xlim = None
        state.view_ylim = None
        player.start = 1
        player.end = 0
        player.value = 1
        player.disabled = True
        experiment_info.object = "Select an experiment result to begin."
        return

    result = load_result(selected_path)
    history_step_indices = result.history.step_indices
    measurement_count = len(history_step_indices)
    initial_measurement_step = measurement_count if measurement_count > 0 else 0

    state.result = result
    state.selected_path = selected_path
    state.history_step_indices = history_step_indices
    state.default_xlim, state.default_ylim = compute_plot_limits(result.history)
    state.view_xlim = state.default_xlim
    state.view_ylim = state.default_ylim

    player.start = 1
    player.end = measurement_count
    player.value = initial_measurement_step
    player.disabled = measurement_count == 0

    experiment_info.object = (
        f"### Experiment\n"
        f"- File: `{selected_path.name}`\n"
        f"- Config: `{result.config.name}`\n"
        f"- Created: `{result.created_at_utc}`\n"
        f"- Measurement steps: `{measurement_count}`"
    )


file_select.param.watch(update_selected_file, "value")


@pn.depends(player.param.value, file_select.param.value, watch=False)
def step_view(measurement_step: int, _selected_file: Path | None):
    """Render the current SLAM map view for the selected step."""

    if state.result is None or state.history_step_indices is None:
        step_info.object = "No measurement step selected."
        return pn.pane.Markdown("Select an experiment result to see the SLAM estimate.")

    if not (1 <= measurement_step <= len(state.history_step_indices)):
        step_info.object = f"Measurement step `{measurement_step}` is not available."
        return pn.pane.Markdown(
            f"Measurement step `{measurement_step}` is not available in this experiment."
        )

    history_step_index = state.history_step_indices[measurement_step - 1]
    entry = state.result.history.require(history_step_index)
    predicted_landmark_ids = entry.step_output.measurement_prediction.predicted_landmark_ids
    local_map_landmark_count = int(len(predicted_landmark_ids)) if predicted_landmark_ids is not None else 0
    landmark_count = len(entry.step_output.estimate.landmark_positions) if entry.step_output.estimate.landmark_positions is not None else 0
    step_info.object = (
        f"### Current Step\n"
        f"- Measurement step: `{measurement_step}`\n"
        f"- Odometry step: `{history_step_index}`\n"
        f"- #landmarks: `{landmark_count}`\n"
        f"- #local_landmarks: `{local_map_landmark_count}`"
    )
    return build_step_plot(entry, state)


@pn.depends(player.param.value, file_select.param.value, watch=False)
def profiler_view(measurement_step: int, _selected_file: Path | None):
    """Render profiler plots for the selected experiment."""

    if state.result is None:
        return pn.pane.Markdown("Select an experiment result to see profiler data.")

    return pn.Column(
        build_covariance_runtime_plot(state.result, measurement_step),
        build_local_landmark_count_plot(state.result, measurement_step),
        sizing_mode="stretch_width",
    )

sidebar = pn.layout.WidgetBox(
    file_select,
    experiment_info,
    step_info,
    player,
    max_width=350,
    sizing_mode="stretch_width",
).servable(area="sidebar")

main = pn.Column(
    step_view,
    profiler_view,
    align="center",
    sizing_mode="stretch_both",
).servable(area="main")

app = pn.Row(sidebar, main)
