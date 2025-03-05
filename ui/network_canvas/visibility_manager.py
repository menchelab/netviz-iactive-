import numpy as np
import logging


class VisibilityManager:
    def __init__(self, canvas):
        self.canvas = canvas

    def update_visibility(
        self,
        node_mask=None,
        edge_mask=None,
        show_intralayer=True,
        show_nodes=True,
        show_labels=True,
        bottom_labels_only=True,
        show_stats_bars=False,
        intralayer_width=1.0,
        interlayer_width=1.0,
        intralayer_opacity=1.0,
        interlayer_opacity=1.0,
        node_size=1.0,
        node_opacity=1.0,
        antialias=True,
    ):
        """Update the visibility of nodes and edges based on masks and display settings"""
        logger = logging.getLogger(__name__)
        logger.info(
            f"Updating visibility with show_intralayer={show_intralayer}, show_nodes={show_nodes}, "
            f"show_labels={show_labels}, show_stats_bars={show_stats_bars}, "
            f"intralayer_width={intralayer_width}, interlayer_width={interlayer_width}, "
            f"node_size={node_size}, node_opacity={node_opacity}, antialias={antialias}"
        )

        # Use masks from data manager if available and not provided
        if node_mask is None and self.canvas.data_manager is not None:
            node_mask = self.canvas.data_manager.current_node_mask
        if edge_mask is None and self.canvas.data_manager is not None:
            edge_mask = self.canvas.data_manager.current_edge_mask

        # Store the current masks for later use
        self.canvas.current_node_mask = node_mask
        self.canvas.current_edge_mask = edge_mask

        # Get interlayer edge counts from data manager if available
        if self.canvas.data_manager is not None:
            interlayer_edge_counts = (
                self.canvas.data_manager.get_interlayer_edge_counts()
            )
        else:
            interlayer_edge_counts = self._calculate_interlayer_edge_counts(edge_mask)

        # Update node display with size and opacity settings
        if show_nodes:
            self.canvas.node_manager.update_node_display(
                size_scale=node_size,
                opacity=node_opacity
            )
            self.canvas.scatter.visible = True
        else:
            self.canvas.scatter.visible = False

        # Update labels and bars
        self.canvas.label_manager._update_labels_and_bars(
            node_mask, interlayer_edge_counts, show_labels, show_stats_bars
        )

        # Update edge display with width, opacity and antialias settings
        self.canvas.edge_manager._update_edge_visibility(
            edge_mask,
            show_intralayer,
            intralayer_width=intralayer_width,
            interlayer_width=interlayer_width,
            intralayer_opacity=intralayer_opacity,
            interlayer_opacity=interlayer_opacity,
            antialias=antialias
        )

    def _calculate_interlayer_edge_counts(self, edge_mask):
        """Calculate interlayer edge counts for each node"""
        interlayer_edge_counts = {}

        if (
            self.canvas.link_pairs is None
            or self.canvas.node_ids is None
            or self.canvas.nodes_per_layer is None
        ):
            return interlayer_edge_counts

        for i, (start_idx, end_idx) in enumerate(self.canvas.link_pairs):
            # Skip edges that aren't visible
            if not edge_mask[i]:
                continue

            # Skip intralayer edges
            start_layer = start_idx // self.canvas.nodes_per_layer
            end_layer = end_idx // self.canvas.nodes_per_layer
            if start_layer == end_layer:
                continue

            # Count this edge for both source and destination nodes
            src_id = self.canvas.node_ids[start_idx].split("_")[0]
            dst_id = self.canvas.node_ids[end_idx].split("_")[0]

            if src_id not in interlayer_edge_counts:
                interlayer_edge_counts[src_id] = 0
            if dst_id not in interlayer_edge_counts:
                interlayer_edge_counts[dst_id] = 0

            interlayer_edge_counts[src_id] += 1
            interlayer_edge_counts[dst_id] += 1

        return interlayer_edge_counts

    def _update_node_visibility(self, node_mask, show_nodes):
        """Update the visibility of nodes"""
        logger = logging.getLogger(__name__)

        if not show_nodes:
            # Hide all nodes
            self.canvas.scatter.visible = False
            return

        self.canvas.scatter.visible = True

        # Get visible node positions and colors
        visible_positions = self.canvas.node_positions[node_mask]
        visible_colors = self.canvas.node_colors_rgba[node_mask]
        visible_sizes = self.canvas.node_sizes[node_mask]

        logger.info(f"Showing {len(visible_positions)} nodes")

        # Update the scatter visual
        if len(visible_positions) > 0:
            self.canvas.scatter.set_data(
                pos=visible_positions, face_color=visible_colors, size=visible_sizes
            )
        else:
            # Create a dummy point that's invisible
            self.canvas.scatter.set_data(
                pos=np.array([[0, 0, 0]]),
                face_color=np.array([[0, 0, 0, 0]]),
                size=np.array([0]),
            )
