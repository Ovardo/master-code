import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.widgets import CheckButtons


from master_code.data_loader import VictoriaParkLoader, find_gnss_outliers
from master_code.preprocessing import detect_trees
from master_code.paths import FIGURES_ROOT

RANGE_CMAP = "viridis_r"
LIDAR_INFINITY_RANGE = 80.0
CARTESIAN_PLOT_RANGE = 85.0
CROP_BOX_COLOR = "#f59e0b"
SCAN_LINE_COLOR = "#22d3ee"

def plot_sensor_timing(lidar, odometry, gnss, window_seconds=2):
    T_lsr = lidar[:, 0]
    T_odo = odometry[:, 0]
    T_gps = gnss[:, 0]

    lsr_dt = np.diff(T_lsr)
    odo_dt = np.diff(T_odo)

    avg_lsr_sample_time = np.mean(lsr_dt)
    avg_odo_sample_time = np.mean(odo_dt)

    avg_lsr_freq = 1.0 / avg_lsr_sample_time
    avg_odo_freq = 1.0 / avg_odo_sample_time

    print(f"Average lidar sample time: {avg_lsr_sample_time:.4f} s")
    print(f"Average odometry sample time: {avg_odo_sample_time:.4f} s")
    print(f"Average lidar frequency: {avg_lsr_freq:.2f} Hz")
    print(f"Average odometry frequency: {avg_odo_freq:.2f} Hz")

    t0 = min(T_lsr[0], T_odo[0], T_gps[0])
    t1 = t0 + window_seconds

    T_lsr = T_lsr[(T_lsr >= t0) & (T_lsr <= t1)] - t0
    T_odo = T_odo[(T_odo >= t0) & (T_odo <= t1)] - t0
    T_gps = T_gps[(T_gps >= t0) & (T_gps <= t1)] - t0

    plt.figure(figsize=(10, 5))

    plt.vlines(T_lsr, 0, 1, color='r', label=f'Laser scan times ({avg_lsr_freq:.2f} Hz, {avg_lsr_sample_time*1000:.2f} ms)')
    plt.vlines(T_odo, 0, 0.5, color='b', label=f'Odometry times ({avg_odo_freq:.2f} Hz, {avg_odo_sample_time*1000:.2f} ms)')
    plt.vlines(T_gps, 0, 1.5, color='g', label=f'gnss times')
    plt.title("Timing of Laser Scans, Odometry, and gnss Measurements")
    plt.xlabel("Time (s)")
    plt.ylim(0, 1.75)
    plt.yticks([])
    plt.legend()
    plt.grid()
    plt.tight_layout()


def plot_raw_gps(gnss, t0=None, lidar_scans_times=None):
    """
    Plot raw gnss position measurements.

    ``gnss`` is expected to have columns [timestamp, x, y], where x/y are in
    meters and timestamp is in seconds.
    """


    if t0 is None:
        t0 = gnss[0, 0]

    gps_times = gnss[:, 0] - t0

    # Find closest lidar scan indices to each GPS time
    lidar_times = lidar_scans_times - t0
    closest_scan_step = np.argmin(np.abs(lidar_times[:, np.newaxis] - gps_times[np.newaxis, :]), axis=0)

    gps_fig, gps_axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    gps_points = gps_axis.scatter(
        gnss[:, 1],
        gnss[:, 2],
        c=closest_scan_step,
        cmap="turbo",
        s=10,
        linewidths=0,
        label=r"$z_{\text{gnss}}$",
    )
    gps_axis.scatter(
        gnss[0, 1],
        gnss[0, 2],
        color="tab:blue",
        marker="o",
        s=70,
        linewidths=1.6,
        label=r"$z_{\text{gnss}, 1}$",
    )
    gps_axis.scatter(
        gnss[-1, 1],
        gnss[-1, 2],
        color="tab:red",
        marker="s",
        s=70,
        linewidths=1.6,
        label=r"$z_{\text{gnss}, N}$",
    )
    gps_axis.set_title("Raw gnss positions")
    gps_axis.set_xlabel("Easting (m)")
    gps_axis.set_ylabel("Northing (m)")
    gps_axis.axis("equal")
    gps_axis.legend()
    gps_axis.grid(True, which="both", linewidth=0.4)
    gps_fig.colorbar(gps_points, ax=gps_axis, label="Time (s)")

    return gps_fig



def plot_raw_odometry(odometry, t0=None):
    """
    Plot raw wheel odometry measurements.

    ``odometry`` is expected to have columns [timestamp, velocity, steering],
    where velocity is in m/s, steering is in radians, and timestamp is in
    seconds.
    """
    if t0 is None:
        t0 = odometry[0, 0]

    odometry_times = odometry[:, 0] - t0

    odo_fig, odo_axes = plt.subplots(
        2,
        1,
        figsize=(10, 5),
        sharex=True,
        constrained_layout=True,
    )

    odo_axes[0].plot(odometry_times, odometry[:, 1], color="tab:blue", linewidth=0.8)
    odo_axes[0].set_title("Raw odometry measurements")
    odo_axes[0].set_ylabel("Velocity (m/s)")
    odo_axes[0].grid(True, which="both", linewidth=0.4)

    odo_axes[1].plot(odometry_times, odometry[:, 2], color="tab:orange", linewidth=0.8)
    odo_axes[1].set_xlabel("Time (s)")
    odo_axes[1].set_ylabel("Steering (rad)")
    odo_axes[1].grid(True, which="both", linewidth=0.4)
    odo_axes[1].set_xlim(odometry_times[0], odometry_times[-1])

    return odo_fig


def _as_detection_array(detections):
    """Normalize detector output to an (M, 2) range-bearing array."""
    detections = np.asarray(detections, dtype=float)
    if detections.size == 0:
        return np.zeros((0, 2), dtype=float)

    if detections.ndim == 1:
        detections = detections.reshape(1, -1)

    return detections[:, :2]


def _tree_detections_by_scan(lidar_scans):
    return [_as_detection_array(detect_trees(scan)) for scan in lidar_scans]


def _bearing_to_beam_index(bearing, num_beams):
    angle_resolution = np.pi / (num_beams - 1)
    return np.rint((bearing + np.pi / 2) / angle_resolution).astype(int)


def _range_colorbar_label(range_limit):
    if range_limit is None:
        return "Range (m)"
    return f"Range (m), >{range_limit:g} m = inf"


def _tree_detection_images(lidar_scans, detections_by_scan=None):
    """Return beam-by-scan tree masks and ranges from raw lidar scans."""
    num_scans, num_beams = lidar_scans.shape
    if detections_by_scan is None:
        detections_by_scan = _tree_detections_by_scan(lidar_scans)

    detection_mask = np.zeros((num_beams, num_scans), dtype=bool)
    detection_ranges = np.full((num_beams, num_scans), np.nan, dtype=float)

    for scan_idx, detections in enumerate(detections_by_scan):
        if detections.size == 0:
            continue

        bearing = detections[:, 1]
        beam_idx = _bearing_to_beam_index(bearing, num_beams)
        valid = (
            np.isfinite(detections[:, 0])
            & np.isfinite(bearing)
            & (0 <= beam_idx)
            & (beam_idx < num_beams)
        )
        beam_idx = beam_idx[valid]
        ranges = detections[valid, 0]
        detection_mask[beam_idx, scan_idx] = True
        detection_ranges[beam_idx, scan_idx] = ranges

    return detection_mask, detection_ranges


def _plot_range_bearing_guides(axis, max_range, num_beams, range_step=20.0, beam_step=60):
    if max_range is None or max_range <= 0:
        return

    bearings = np.linspace(-np.pi / 2, np.pi / 2, 361)
    ranges = np.arange(range_step, max_range + 1e-9, range_step)

    for range_m in ranges:
        lateral = range_m * np.sin(bearings)
        forward = range_m * np.cos(bearings)
        axis.plot(lateral, forward, color="0.75", linewidth=0.7, zorder=0)
        axis.text(
            0.0,
            min(range_m, 0.97 * max_range),
            f"{range_m:g} m",
            color="0.45",
            fontsize=8,
            ha="center",
            va="bottom",
        )

    beam_indices = np.arange(beam_step, num_beams - 1, beam_step)
    for beam_idx in beam_indices:
        bearing = beam_idx * np.pi / (num_beams - 1) - np.pi / 2
        axis.plot(
            [0.0, max_range * np.sin(bearing)],
            [0.0, max_range * np.cos(bearing)],
            color="0.78",
            linewidth=0.7,
            zorder=0,
        )
        label_range = 0.94 * max_range
        axis.text(
            label_range * np.sin(bearing),
            label_range * np.cos(bearing),
            f"beam {beam_idx:g}",
            color="0.45",
            fontsize=8,
            ha="center",
            va="center",
        )


def plot_raw_measurements(
    lidar,
    odometry,
    gnss,
    show_tree_overlay=True,
    range_limit=LIDAR_INFINITY_RANGE,
):
    t0 = min(lidar[0, 0], odometry[0, 0])

    lidar_times = lidar[:, 0] - t0
    odometry_times = odometry[:, 0] - t0
    t_end = max(lidar_times[-1], odometry_times[-1])

    sensor_fig, sensor_axes = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True,
        constrained_layout=True,
        height_ratios=[2, 1, 1],
    )

    lidar_scans = lidar[:, 1:]
    tree_mask, _tree_ranges = _tree_detection_images(lidar_scans)
    lidar_extent = [lidar_times[0], lidar_times[-1], 0, lidar_scans.shape[1] - 1]
    masked_tree_mask = np.ma.masked_where(~tree_mask, tree_mask)

    lidar_image = sensor_axes[0].imshow(
        lidar_scans.T,
        aspect="auto",
        origin="lower",
        extent=lidar_extent,
        cmap=RANGE_CMAP,
        vmin=0.0,
        vmax=range_limit,
    )
    tree_overlay = None
    if tree_mask.any():
        tree_overlay = sensor_axes[0].imshow(
            masked_tree_mask,
            aspect="auto",
            origin="lower",
            extent=lidar_extent,
            cmap=ListedColormap(["red"]),
            interpolation="nearest",
            alpha=0.85,
            visible=show_tree_overlay,
    )
    sensor_axes[0].set_title("Raw lidar ranges and odometry")
    sensor_axes[0].set_ylabel("Beam index")
    sensor_axes[0].grid(True, which="both", color="white", alpha=0.25, linewidth=0.4)
    sensor_fig.colorbar(lidar_image, ax=sensor_axes[0], label=_range_colorbar_label(range_limit))

    if tree_overlay is not None:
        toggle_axis = sensor_axes[0].inset_axes([0.78, 0.84, 0.2, 0.12])
        toggle_axis.set_facecolor("white")
        tree_toggle = CheckButtons(
            toggle_axis,
            ["Tree overlay"],
            [tree_overlay.get_visible()],
        )

        def toggle_tree_overlay(_label):
            tree_overlay.set_visible(not tree_overlay.get_visible())
            sensor_fig.canvas.draw_idle()

        tree_toggle.on_clicked(toggle_tree_overlay)
        sensor_fig._tree_overlay_toggle = tree_toggle

    sensor_axes[1].plot(odometry_times, odometry[:, 1], color="tab:blue")
    sensor_axes[1].set_ylabel("Velocity (m/s)")
    sensor_axes[1].grid(True, which="both", linewidth=0.4)

    sensor_axes[2].plot(odometry_times, odometry[:, 2], color="tab:orange")
    sensor_axes[2].set_xlabel("Time (s)")
    sensor_axes[2].set_ylabel("Steering (rad)")
    sensor_axes[2].grid(True, which="both", linewidth=0.4)
    sensor_axes[2].set_xlim(0, t_end)

def plot_lidar_tree_detection_summary(
    lidar,
    crop_time_range=None,
    crop_beam_range=None,
    selected_scan_idx=None,
    crop_duration_seconds=60.0,
    crop_beam_padding=25,
    max_plot_range=CARTESIAN_PLOT_RANGE,
    range_limit=LIDAR_INFINITY_RANGE,
    show_polar_guides=True,
):
    """
    Thesis-style overview of raw lidar scans and derived tree measurements.

    The top subplot shows the full raw lidar image, with a box for the crop and
    a vertical line for the selected scan. The lower-left subplot shows the
    cropped lidar image with tree detections overlaid. The lower-right subplot
    shows one scan transformed from range-bearing to Cartesian coordinates.
    Ranges larger than ``range_limit`` are treated as infinite/no-return values
    and are not shown in the Cartesian scan view.
    """
    lidar_times = lidar[:, 0] - lidar[0, 0]
    lidar_scans = lidar[:, 1:]
    num_scans, num_beams = lidar_scans.shape

    detections_by_scan = _tree_detections_by_scan(lidar_scans)
    tree_mask, _tree_ranges = _tree_detection_images(lidar_scans, detections_by_scan)
    detection_counts = np.array([detections.shape[0] for detections in detections_by_scan])

    if crop_time_range is not None:
        crop_t0, crop_t1 = sorted(crop_time_range)
        crop_candidates = np.flatnonzero((crop_t0 <= lidar_times) & (lidar_times <= crop_t1))
    else:
        crop_candidates = np.arange(num_scans)

    if selected_scan_idx is None:
        if crop_candidates.size > 0 and np.max(detection_counts[crop_candidates]) > 0:
            selected_scan_idx = int(crop_candidates[np.argmax(detection_counts[crop_candidates])])
        elif np.max(detection_counts) > 0:
            selected_scan_idx = int(np.argmax(detection_counts))
        else:
            selected_scan_idx = num_scans // 2
    selected_scan_idx = int(np.clip(selected_scan_idx, 0, num_scans - 1))
    selected_time = lidar_times[selected_scan_idx]

    if crop_time_range is None:
        half_duration = crop_duration_seconds / 2.0
        crop_t0 = max(float(lidar_times[0]), float(selected_time - half_duration))
        crop_t1 = min(float(lidar_times[-1]), float(selected_time + half_duration))

    crop_start = int(np.searchsorted(lidar_times, crop_t0, side="left"))
    crop_stop = int(np.searchsorted(lidar_times, crop_t1, side="right") - 1)
    crop_start = int(np.clip(crop_start, 0, num_scans - 1))
    crop_stop = int(np.clip(crop_stop, crop_start, num_scans - 1))
    if crop_start == crop_stop and num_scans > 1:
        if crop_stop < num_scans - 1:
            crop_stop += 1
        else:
            crop_start -= 1

    if crop_beam_range is None:
        crop_beam_indices = []
        for detections in detections_by_scan[crop_start:crop_stop + 1]:
            if detections.size == 0:
                continue
            beam_idx = _bearing_to_beam_index(detections[:, 1], num_beams)
            valid = (0 <= beam_idx) & (beam_idx < num_beams)
            crop_beam_indices.extend(beam_idx[valid].tolist())

        if crop_beam_indices:
            beam_min = min(crop_beam_indices) - crop_beam_padding
            beam_max = max(crop_beam_indices) + crop_beam_padding
        else:
            beam_center = num_beams // 2
            beam_min = beam_center - 60
            beam_max = beam_center + 60
    else:
        beam_min, beam_max = sorted(crop_beam_range)

    beam_min = int(np.clip(beam_min, 0, num_beams - 2))
    beam_max = int(np.clip(beam_max, beam_min + 1, num_beams - 1))

    full_extent = [lidar_times[0], lidar_times[-1], 0, num_beams - 1]
    crop_extent = [lidar_times[crop_start], lidar_times[crop_stop], beam_min, beam_max]
    crop_scan_slice = slice(crop_start, crop_stop + 1)
    crop_beam_slice = slice(beam_min, beam_max + 1)

    finite_range_mask = np.isfinite(lidar_scans) & (lidar_scans > 0)
    if range_limit is not None:
        finite_range_mask &= lidar_scans <= range_limit

    valid_ranges = lidar_scans[finite_range_mask]
    if range_limit is not None:
        range_vmax = range_limit
    elif valid_ranges.size > 0:
        range_vmax = float(np.nanpercentile(valid_ranges, 99))
        if max_plot_range is not None:
            range_vmax = min(range_vmax, max_plot_range)
    else:
        range_vmax = max_plot_range

    fig, axes = plt.subplot_mosaic(
        [["overview", "overview"], ["crop", "cartesian"]],
        figsize=(12, 8),
        constrained_layout=True,
    )
    overview_axis = axes["overview"]
    crop_axis = axes["crop"]
    cartesian_axis = axes["cartesian"]

    lidar_image = overview_axis.imshow(
        lidar_scans.T,
        aspect="auto",
        origin="lower",
        extent=full_extent,
        vmin=0.0,
        vmax=range_vmax,
        cmap=RANGE_CMAP,
    )
    overview_axis.set_title("Full raw lidar image")
    overview_axis.set_ylabel("Beam index")
    overview_axis.set_xlabel("Time (s)")

    crop_width = lidar_times[crop_stop] - lidar_times[crop_start]
    crop_height = beam_max - beam_min
    overview_axis.add_patch(
        Rectangle(
            (lidar_times[crop_start], beam_min),
            crop_width,
            crop_height,
            fill=False,
            edgecolor="black",
            linewidth=3.0,
            alpha=0.7,
        )
    )
    overview_axis.add_patch(
        Rectangle(
            (lidar_times[crop_start], beam_min),
            crop_width,
            crop_height,
            fill=False,
            edgecolor=CROP_BOX_COLOR,
            linewidth=1.5,
            label="Crop in subplot 2",
        )
    )
    overview_axis.axvline(
        selected_time,
        color=SCAN_LINE_COLOR,
        linewidth=1.5,
        label="Scan in subplot 3",
    )
    overview_axis.legend(loc="upper right", framealpha=0.9)

    crop_image = crop_axis.imshow(
        lidar_scans[crop_scan_slice, crop_beam_slice].T,
        aspect="auto",
        origin="lower",
        extent=crop_extent,
        vmin=0.0,
        vmax=range_vmax,
        cmap=RANGE_CMAP,
    )
    crop_tree_mask = tree_mask[crop_beam_slice, crop_scan_slice]
    if crop_tree_mask.any():
        crop_axis.imshow(
            np.ma.masked_where(~crop_tree_mask, crop_tree_mask),
            aspect="auto",
            origin="lower",
            extent=crop_extent,
            cmap=ListedColormap(["red"]),
            interpolation="nearest",
            alpha=0.85,
        )
    crop_axis.axvline(selected_time, color=SCAN_LINE_COLOR, linewidth=1.5)
    crop_axis.set_title("Cropped lidar image with detected trees")
    crop_axis.set_xlabel("Time (s)")
    crop_axis.set_ylabel("Beam index")
    for spine in crop_axis.spines.values():
        spine.set_edgecolor(CROP_BOX_COLOR)
        spine.set_linewidth(2.0)

    sensor_bearings = np.linspace(-np.pi / 2, np.pi / 2, num_beams)
    selected_scan = lidar_scans[selected_scan_idx]
    raw_valid = np.isfinite(selected_scan) & (selected_scan > 0)
    if range_limit is not None:
        raw_valid &= selected_scan <= range_limit
    if max_plot_range is not None:
        raw_valid &= selected_scan <= max_plot_range

    raw_lateral = selected_scan[raw_valid] * np.sin(sensor_bearings[raw_valid])
    raw_forward = selected_scan[raw_valid] * np.cos(sensor_bearings[raw_valid])
    guide_range = max_plot_range if max_plot_range is not None else range_vmax
    if guide_range is not None and show_polar_guides:
        _plot_range_bearing_guides(cartesian_axis, guide_range, num_beams)

    cartesian_axis.scatter(
        raw_lateral,
        raw_forward,
        s=6,
        color="0.35",
        alpha=0.45,
        linewidths=0,
        label="Raw lidar returns",
        zorder=3,
    )

    selected_detections = detections_by_scan[selected_scan_idx]
    if selected_detections.size > 0:
        detection_ranges = selected_detections[:, 0]
        detection_bearings = selected_detections[:, 1]
        valid_detections = np.isfinite(detection_ranges) & np.isfinite(detection_bearings)
        if range_limit is not None:
            valid_detections &= detection_ranges <= range_limit
        if max_plot_range is not None:
            valid_detections &= detection_ranges <= max_plot_range

        tree_lateral = detection_ranges[valid_detections] * np.sin(detection_bearings[valid_detections])
        tree_forward = detection_ranges[valid_detections] * np.cos(detection_bearings[valid_detections])
        cartesian_axis.scatter(
            tree_lateral,
            tree_forward,
            s=16,
            color="red",
            linewidths=0,
            label="Detected trees",
            zorder=2,
        )

    cartesian_axis.scatter(0.0, 0.0, marker="+", s=90, color="tab:blue", zorder=4)
    cartesian_axis.set_title(f"Scan {selected_scan_idx} in sensor frame")
    cartesian_axis.set_xlabel("Lateral x (m)")
    cartesian_axis.set_ylabel("Forward y (m)")
    cartesian_axis.set_aspect("equal", adjustable="box")
    cartesian_axis.grid(True, linewidth=0.4)
    cartesian_axis.legend(loc="best")

    if max_plot_range is not None:
        cartesian_axis.set_xlim(-max_plot_range, max_plot_range)
        cartesian_axis.set_ylim(0.0, max_plot_range)

    fig.colorbar(lidar_image, ax=[overview_axis, crop_axis], label=_range_colorbar_label(range_limit))
    return fig

def plot_gnss(gnss):
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 4), constrained_layout=True)
    
    gnss_norm = np.linalg.norm(gnss[:, 1:3], axis=1)

    outliers_idx = find_gnss_outliers(gnss)
    print(f"GNSS outlier indices: {outliers_idx.tolist()}")

    # plot consecutive GNSS measurements connected by a thin line with some transparency
    ax[0].plot(gnss[:, 1], gnss[:, 2], color="steelblue", linewidth=0.8, alpha=0.4)
    ax[0].scatter(gnss[:, 1], gnss[:, 2], s=10, color="steelblue", label="gnss positions")
    if outliers_idx.size > 0:
        ax[0].scatter(gnss[outliers_idx, 1], gnss[outliers_idx, 2], s=30, color="red", label="gnss outliers", zorder=5)
    
    ax[1].scatter(gnss[:,0], gnss_norm, s=10, color="steelblue", label="gnss position norm")
    if outliers_idx.size > 0:
        ax[1].scatter(gnss[outliers_idx, 0], gnss_norm[outliers_idx], s=10, color="red", label="gnss outliers", zorder=5)

    return fig, outliers_idx


def main():
    data_loader = VictoriaParkLoader()
    
    lidar = data_loader.lidar
    odometry = data_loader.odometry
    gnss = data_loader.gnss

    plot_gnss(gnss)
    plot_sensor_timing(lidar, odometry, gnss, window_seconds=1400)
    plot_raw_measurements(lidar, odometry, gnss)

    fig = plot_raw_odometry(odometry)
    fig.set_size_inches(11.0, 4.0)  # width, height in inches
    fig.savefig(f"{FIGURES_ROOT}/odometry.pdf", bbox_inches="tight")

    fig = plot_raw_gps(gnss, lidar_scans_times=lidar[:,0])
    fig.set_size_inches(6.0, 5.5)  # width, height in inches
    fig.savefig(f"{FIGURES_ROOT}/gnss.pdf", bbox_inches="tight")
    
    fig = plot_lidar_tree_detection_summary(lidar)
    fig.set_size_inches(11.0, 5.5)  # width, height in inches
    fig.savefig(f"{FIGURES_ROOT}/lidar.pdf", bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    main()
