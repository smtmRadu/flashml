import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def plot_tensor(
    tensor,
    title="Tensor Heatmap",
    cmap="winter",
    annotate_threshold=10,
    element_display_size_px=50,
    max_figure_dim_px=1200,
    hide_axis_labels_if_total_elements_gt=1000,
):
    """
    Plots a 0D, 1D, 2D, or 3D tensor primarily as heatmaps with indices.
    Figure size is proportional to tensor shape for square elements.
    Adjusts aspect for 1D. Uses constrained_layout for 3D.
    Hides row/column annotations for large tensors.

    Args:
        tensor (torch.tensor, numpy.ndarray or list): The tensor to plot.
        title (str, optional): The overall title for the plot(s). Defaults to "Tensor Heatmap".
        cmap (str, optional): The colormap to use. Defaults to "winter".
        annotate_threshold (int, optional): For cell *value* annotations. If a dimension
                                           (rows or cols) has this many elements or fewer,
                                           cell values will be annotated. Set to 0 to disable.
                                           Defaults to 10.
        element_display_size_px (int, optional): Desired display size of each square tensor
                                                 element in pixels. Defaults to 50.
        max_figure_dim_px (int, optional): Maximum width or height of the generated
                                           figure in pixels. Defaults to 1200.
        hide_axis_labels_if_total_elements_gt (int, optional): If the total number of
                                           elements in a 2D view (e.g., a 2D tensor or a
                                           slice of a 3D tensor) exceeds this,
                                           row and column tick labels are hidden. Defaults to 1000.
    """
    if not isinstance(tensor, np.ndarray):
        try:
            tensor = np.array(tensor.cpu())
        except Exception as e:
            print(f"Error: Could not convert input to NumPy array. {e}")
            return

    try:
        dpi = plt.rcParams["figure.dpi"]
        if dpi <= 0:
            dpi = 100.0
    except KeyError:
        dpi = 100.0

    element_size_inches = element_display_size_px / dpi
    max_fig_dim_inches = max_figure_dim_px / dpi
    min_practical_fig_dim_inches = 0.5

    def set_window_top_left():
        try:
            manager = plt.get_current_fig_manager()
            if manager is None:
                return
            backend = plt.get_backend()
            if backend == "TkAgg":
                current_geometry = manager.window.geometry()
                size_part = current_geometry.split("+")[0]
                if "x" in size_part:
                    manager.window.geometry(f"{size_part}+0+0")
                else:
                    manager.window.wm_geometry("+0+0")
            elif backend.startswith("Qt"):
                manager.window.move(0, 0)
            elif backend == "WXAgg":
                manager.window.SetPosition((0, 0))
            elif backend.startswith("GTK"):
                manager.window.move(0, 0)
        except Exception:
            pass

    def annotate_heatmap_cells(
        ax, data_slice, cmap_obj, norm_obj, cell_val_annotate_thresh
    ):
        if not (
            cell_val_annotate_thresh > 0
            and data_slice.ndim == 2  # Ensure data_slice is 2D
            and data_slice.shape[0] <= cell_val_annotate_thresh
            and data_slice.shape[1] <= cell_val_annotate_thresh
        ):
            return

        for i in range(data_slice.shape[0]):
            for j in range(data_slice.shape[1]):
                val = data_slice[i, j]
                bg_color_rgba = cmap_obj(norm_obj(val))
                luminance = (
                    0.299 * bg_color_rgba[0]
                    + 0.587 * bg_color_rgba[1]
                    + 0.114 * bg_color_rgba[2]
                )
                text_color = "white" if luminance < 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2g}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

    def _calculate_dynamic_figsize(
        num_rows, num_cols, elem_size_in, max_dim_in, min_dim_in
    ):
        if num_rows <= 0:
            num_rows = 1
        if num_cols <= 0:
            num_cols = 1

        fig_w_ideal = num_cols * elem_size_in
        fig_h_ideal = num_rows * elem_size_in

        scale = 1.0
        if fig_w_ideal > max_dim_in and fig_w_ideal > 0:
            scale = min(scale, max_dim_in / fig_w_ideal)
        if fig_h_ideal > max_dim_in and fig_h_ideal > 0:
            scale = min(scale, max_dim_in / fig_h_ideal)

        final_w = max(fig_w_ideal * scale, min_dim_in if num_cols > 0 else elem_size_in)
        final_h = max(fig_h_ideal * scale, min_dim_in if num_rows > 0 else elem_size_in)
        return (final_w, final_h)

    if tensor.size == 0:
        empty_fig_size = _calculate_dynamic_figsize(
            5, 10, element_size_inches, max_fig_dim_inches, min_practical_fig_dim_inches
        )
        empty_fig_size = (min(empty_fig_size[0], 8), min(empty_fig_size[1], 4))

        fig, ax = plt.subplots(figsize=empty_fig_size)
        ax.text(
            0.5,
            0.5,
            f"Tensor is empty (shape: {tensor.shape}).\nCannot plot.",
            ha="center",
            va="center",
            fontsize=12,
            color="orange",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Empty Tensor", fontsize=12)
        plt.tight_layout()
        set_window_top_left()
        plt.show()
        return

    dim = tensor.ndim
    cmap_obj = plt.get_cmap(cmap)

    if dim == 0:
        current_figsize = _calculate_dynamic_figsize(
            1, 1, element_size_inches, max_fig_dim_inches, min_practical_fig_dim_inches
        )
        fig, ax = plt.subplots(figsize=current_figsize)
        data_0d = np.array([[tensor.item()]])
        vmin, vmax = data_0d.min(), data_0d.max()
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(data_0d, cmap=cmap_obj, norm=norm, aspect="auto")

        ax.set_xticks([0])
        ax.set_xticklabels(["0"])
        ax.set_yticks([0])
        ax.set_yticklabels(["0"])
        ax.set_title(f"{title}\n0D Tensor (Scalar)", fontsize=10)
        annotate_heatmap_cells(ax, data_0d, cmap_obj, norm, 1)
        fig.colorbar(
            im, ax=ax, label="Value", orientation="vertical", fraction=0.1, pad=0.1
        )
        plt.tight_layout()

    elif dim == 1:
        num_elements = tensor.shape[0]
        data_1d = tensor.reshape(1, -1)
        rows_display, cols_display = data_1d.shape

        current_figsize = _calculate_dynamic_figsize(
            rows_display,
            cols_display,
            element_size_inches,
            max_fig_dim_inches,
            min_practical_fig_dim_inches,
        )
        fig, ax = plt.subplots(figsize=current_figsize)

        vmin, vmax = data_1d.min(), data_1d.max()
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(data_1d, cmap=cmap_obj, norm=norm, aspect="auto")

        show_labels = data_1d.size <= hide_axis_labels_if_total_elements_gt
        ax.set_yticks([0])
        if show_labels:
            ax.set_yticklabels(["0 (Row)"])
            ax.set_xticks(np.arange(cols_display))
            ax.set_xticklabels([str(i) for i in np.arange(cols_display)], fontsize=8)
            ax.set_xlabel("Column Index", fontsize=10)
        else:
            ax.set_yticklabels([""])
            ax.set_xticks([])

        ax.set_title(f"{title}\n1D Tensor ({num_elements} elements)", fontsize=12)
        annotate_heatmap_cells(ax, data_1d, cmap_obj, norm, annotate_threshold)
        fig.colorbar(
            im, ax=ax, label="Value", orientation="vertical", fraction=0.08, pad=0.04
        )
        plt.tight_layout()

    elif dim == 2:
        rows_display, cols_display = tensor.shape
        current_figsize = _calculate_dynamic_figsize(
            rows_display,
            cols_display,
            element_size_inches,
            max_fig_dim_inches,
            min_practical_fig_dim_inches,
        )
        fig, ax = plt.subplots(figsize=current_figsize)

        vmin, vmax = tensor.min(), tensor.max()
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(tensor, cmap=cmap_obj, norm=norm, aspect="auto")

        show_labels = tensor.size <= hide_axis_labels_if_total_elements_gt
        if show_labels:
            ax.set_yticks(np.arange(rows_display))
            ax.set_yticklabels([str(i) for i in np.arange(rows_display)], fontsize=8)
            ax.set_ylabel("Row Index", fontsize=10)
            ax.set_xticks(np.arange(cols_display))
            ax.set_xticklabels([str(i) for i in np.arange(cols_display)], fontsize=8)
            ax.set_xlabel("Column Index", fontsize=10)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_title(f"{title}\n2D Tensor ({rows_display}x{cols_display})", fontsize=12)
        annotate_heatmap_cells(ax, tensor, cmap_obj, norm, annotate_threshold)
        fig.colorbar(im, ax=ax, label="Value")
        plt.tight_layout()

    elif dim == 3:
        num_slices, slice_rows, slice_cols = tensor.shape
        if num_slices == 0:
            print("Error: 3D tensor has 0 slices.")
            return

        grid_cols = int(np.ceil(np.sqrt(num_slices)))
        grid_rows = int(np.ceil(num_slices / grid_cols))

        total_elements_high = grid_rows * slice_rows
        total_elements_wide = grid_cols * slice_cols

        current_figsize = _calculate_dynamic_figsize(
            total_elements_high,
            total_elements_wide,
            element_size_inches,
            max_fig_dim_inches,
            min_practical_fig_dim_inches,
        )

        fig, axes = plt.subplots(
            grid_rows,
            grid_cols,
            figsize=current_figsize,
            squeeze=False,
            constrained_layout=True,
        )
        axes_flat = axes.flatten()

        global_min, global_max = tensor.min(), tensor.max()
        if global_min == global_max:
            global_min -= 0.5
            global_max += 0.5
        norm = mcolors.Normalize(vmin=global_min, vmax=global_max)
        mappable = None

        for i in range(num_slices):
            ax = axes_flat[i]
            slice_data = tensor[i, :, :]
            current_mappable = ax.imshow(
                slice_data, cmap=cmap_obj, norm=norm, aspect="auto"
            )
            if i == 0:
                mappable = current_mappable

            show_labels_slice = slice_data.size <= hide_axis_labels_if_total_elements_gt
            if show_labels_slice:
                ax.set_yticks(np.arange(slice_rows))
                ax.set_yticklabels([str(j) for j in np.arange(slice_rows)], fontsize=7)
                ax.set_ylabel("Row", fontsize=8)
                ax.set_xticks(np.arange(slice_cols))
                ax.set_xticklabels([str(k) for k in np.arange(slice_cols)], fontsize=7)
                ax.set_xlabel("Col", fontsize=8)
            else:
                ax.set_xticks([])
                ax.set_yticks([])

            ax.set_title(f"Slice: {i}", fontsize=9)
            annotate_heatmap_cells(ax, slice_data, cmap_obj, norm, annotate_threshold)

        for i in range(num_slices, len(axes_flat)):
            axes_flat[i].axis("off")

        fig.suptitle(
            f"{title}\n3D Tensor ({num_slices} Slices as Heatmaps)",
            fontsize=14,
            fontweight="bold",
        )
        if mappable is not None:
            fig.colorbar(
                mappable,
                ax=axes.ravel().tolist(),
                label="Value",
                shrink=0.7,
                aspect=25,
                pad=0.02,
            )

    else:
        error_fig_size = _calculate_dynamic_figsize(
            3, 10, element_size_inches, max_fig_dim_inches, min_practical_fig_dim_inches
        )
        error_fig_size = (min(error_fig_size[0], 6), min(error_fig_size[1], 2.5))  # Cap

        fig, ax = plt.subplots(figsize=error_fig_size)
        ax.text(
            0.5,
            0.5,
            f"Cannot plot {dim}D tensor.\nSupports 0D, 1D, 2D, and 3D.",
            ha="center",
            va="center",
            fontsize=12,
            color="red",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Unsupported Tensor Dimension", fontsize=12)
        plt.tight_layout()

    set_window_top_left()
    plt.show()
