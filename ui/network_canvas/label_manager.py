import numpy as np
import logging


class LabelManager:
    def __init__(self, canvas):
        self.canvas = canvas

    def _update_labels_and_bars(
        self, node_mask, interlayer_edge_counts, show_labels, show_stats_bars
    ):
        """Update node labels and bar charts"""
        logger = logging.getLogger(__name__)
        logger.info(
            f"Updating labels and bars: show_labels={show_labels}, show_stats_bars={show_stats_bars}"
        )

        # If not showing labels or bars, hide them and return early
        if not show_labels and not show_stats_bars:
            self.canvas.node_labels.visible = False
            self.canvas.node_count_bars.visible = False
            self.canvas.edge_count_bars.visible = False
            return

        if (
            self.canvas.data_manager.node_ids is None
            or self.canvas.data_manager.visible_layers is None
            or self.canvas.data_manager.layer_names is None
            or not self.canvas.data_manager.visible_layers
        ):
            # Hide everything if we don't have the necessary data
            self.canvas.node_labels.visible = False
            self.canvas.node_count_bars.visible = False
            self.canvas.edge_count_bars.visible = False
            return

        # Prepare data for labels and bar charts
        label_positions = []
        label_texts = []
        label_colors = []

        node_count_line_points = []
        node_count_line_colors = []
        edge_count_line_points = []
        edge_count_line_colors = []

        # Determine the top layer index (the lowest layer index that is visible)
        top_layer_idx = (
            min(self.canvas.data_manager.visible_layers)
            if self.canvas.data_manager.visible_layers
            else -1
        )
        logger.info(f"Top visible layer index: {top_layer_idx}")

        # Track which base nodes we've already processed
        processed_base_nodes = set()

        # Process visible nodes in the top layer first
        visible_indices = np.where(node_mask)[0]

        # Group nodes by their base ID
        base_node_to_indices = {}
        for idx in visible_indices:
            node_id = self.canvas.data_manager.node_ids[idx]
            base_node = node_id.split("_")[0] if "_" in node_id else node_id

            if base_node not in base_node_to_indices:
                base_node_to_indices[base_node] = []
            base_node_to_indices[base_node].append(idx)

        # For each base node, find the node in the top-most visible layer
        for base_node, indices in base_node_to_indices.items():
            # Find the node in the top-most layer
            top_node_idx = None
            top_node_layer = float("inf")

            for idx in indices:
                node_layer_idx = idx // self.canvas.data_manager.nodes_per_layer
                if (
                    node_layer_idx in self.canvas.data_manager.visible_layers
                    and node_layer_idx < top_node_layer
                ):
                    top_node_idx = idx
                    top_node_layer = node_layer_idx

            if top_node_idx is None:
                continue  # No visible nodes for this base node

            # Process only the top node for this base node
            idx = top_node_idx
            node_layer_idx = idx // self.canvas.data_manager.nodes_per_layer

            # Get node ID and base node
            node_id = self.canvas.data_manager.node_ids[idx]
            label_text = node_id.split("_")[0] if "_" in node_id else node_id

            # Count active nodes with this base ID across all layers
            active_node_count = self.canvas.data_manager.count_active_nodes_for_base(
                base_node, idx
            )

            # Get edge count if available
            edge_count = interlayer_edge_counts.get(base_node, 0)

            # Update labels if needed
            if show_labels:
                # Format label text with counts
                if active_node_count > 0 or edge_count > 0:
                    label_text_with_counts = (
                        f"{label_text} [{active_node_count}/{edge_count // 2}]"
                    )
                else:
                    label_text_with_counts = label_text

                # Add a small offset to position labels better
                pos = self.canvas.data_manager.node_positions[idx].copy()
                pos[1] += 0.01  # Offset in y direction
                label_positions.append(pos)
                label_texts.append(label_text_with_counts)

                # Get the layer name for this node
                layer_name = self.canvas.data_manager.layer_names[node_layer_idx]

                # Use the layer color from the data manager
                if (
                    self.canvas.data_manager.layer_colors_rgba
                    and layer_name in self.canvas.data_manager.layer_colors_rgba
                ):
                    label_color = self.canvas.data_manager.layer_colors_rgba[
                        layer_name
                    ].copy()
                    # Make labels with 0 interlayer connections more transparent
                    if edge_count == 0:
                        label_color[3] = 0.4
                else:
                    label_color = np.array([1.0, 1.0, 0.0, 1.0])
                    if edge_count == 0:
                        label_color[3] = 0.8

                label_colors.append(label_color)

            # Update bar charts if needed
            if show_stats_bars and (active_node_count > 0 or edge_count > 0):
                # Scale factors for bar widths
                max_bar_width = 0.3
                node_count_width = min(max_bar_width, 0.002 * active_node_count)
                edge_count_width = min(max_bar_width, 0.002 * (edge_count // 2))

                # Node count bar (top bar)
                if node_count_width > 0:
                    start_point = self.canvas.data_manager.node_positions[idx].copy()
                    end_point = start_point.copy()
                    end_point[0] += node_count_width

                    node_count_line_points.append(start_point)
                    node_count_line_points.append(end_point)

                    node_color = np.array([1.0, 0.0, 1.0, 1])  # Magenta for node count
                    node_count_line_colors.append(node_color)
                    node_count_line_colors.append(node_color)

                # Edge count bar (bottom bar)
                if edge_count_width > 0:
                    start_point = self.canvas.data_manager.node_positions[idx].copy()
                    start_point[1] -= 0.005  # Offset second barchart

                    end_point = start_point.copy()
                    end_point[0] += edge_count_width

                    edge_count_line_points.append(start_point)
                    edge_count_line_points.append(end_point)

                    edge_color = np.array([0.0, 1.0, 0.0, 1])  # Green for edge count
                    if edge_count == 0:
                        edge_color[3] = 0.3

                    edge_count_line_colors.append(edge_color)
                    edge_count_line_colors.append(edge_color)

        # Update the label visual
        if show_labels and label_positions:
            self.canvas.node_labels.pos = np.array(label_positions)
            self.canvas.node_labels.text = label_texts
            self.canvas.node_labels.color = np.array(label_colors)
            self.canvas.node_labels.visible = True
            self.canvas.node_labels.order = 1  # Higher order means drawn later (on top)
        else:
            self.canvas.node_labels.visible = False

        # Update the bar chart visuals
        if show_stats_bars and node_count_line_points:
            self.canvas.node_count_bars.set_data(
                pos=np.array(node_count_line_points),
                color=np.array(node_count_line_colors),
                connect="segments",
                width=3,
            )
            self.canvas.node_count_bars.visible = True
            self.canvas.node_count_bars.order = 0.8  # Draw behind labels
        else:
            self.canvas.node_count_bars.visible = False

        if show_stats_bars and edge_count_line_points:
            self.canvas.edge_count_bars.set_data(
                pos=np.array(edge_count_line_points),
                color=np.array(edge_count_line_colors),
                connect="segments",
                width=3,
            )
            self.canvas.edge_count_bars.visible = True
            self.canvas.edge_count_bars.order = (
                0.9  # Draw behind labels but above node count bars
            )
        else:
            self.canvas.edge_count_bars.visible = False

        logger.info(
            f"Label visibility set to {self.canvas.node_labels.visible}, showing {len(label_positions)} labels"
        )
        logger.info(
            f"Bar visibility set to {self.canvas.node_count_bars.visible} and {self.canvas.edge_count_bars.visible}"
        )
