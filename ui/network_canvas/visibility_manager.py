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
    ):
        """Update the visibility of nodes and edges based on masks"""
        logger = logging.getLogger(__name__)
        logger.info(
            f"Updating visibility with show_intralayer={show_intralayer}, show_nodes={show_nodes}, show_labels={show_labels}, show_stats_bars={show_stats_bars}"
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

        self._update_node_visibility(node_mask, show_nodes)

        self.canvas.label_manager._update_labels_and_bars(
            node_mask, interlayer_edge_counts, show_labels, show_stats_bars
        )

        self.canvas.edge_manager._update_edge_visibility(edge_mask, show_intralayer)

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
        """Update node visibility based on mask"""
        logger = logging.getLogger(__name__)

        if not show_nodes:
            # Hide all nodes
            self.canvas.scatter.visible = False
            return

        self.canvas.scatter.visible = True

        # Handle case when node_mask is None
        if node_mask is None:
            logger.warning("Node mask is None, showing all nodes")
            node_mask = np.ones(len(self.canvas.node_positions), dtype=bool)
        
        # Ensure node_mask is a boolean array with the correct length
        if not isinstance(node_mask, np.ndarray) or len(node_mask) != len(self.canvas.node_positions):
            logger.warning(f"Invalid node mask, showing all nodes. Mask type: {type(node_mask)}, length: {len(node_mask) if hasattr(node_mask, '__len__') else 'N/A'}")
            node_mask = np.ones(len(self.canvas.node_positions), dtype=bool)
        
        # Convert to boolean if not already
        if node_mask.dtype != bool:
            logger.warning(f"Converting node mask from {node_mask.dtype} to boolean")
            node_mask = node_mask.astype(bool)

        try:
            # Get visible node positions and colors
            visible_indices = np.where(node_mask)[0]
            logger.info(f"Visible indices: {len(visible_indices)} out of {len(node_mask)}")
            
            # Check if node_positions is available
            if self.canvas.node_positions is None or len(self.canvas.node_positions) == 0:
                logger.error("No node positions available")
                return
                
            visible_positions = self.canvas.node_positions[visible_indices]
            
            # Check if node_colors_rgba is available
            if self.canvas.node_colors_rgba is None:
                logger.warning("No node colors available, using default white")
                visible_colors = np.ones((len(visible_indices), 4))  # White with full opacity
            else:
                visible_colors = self.canvas.node_colors_rgba[visible_indices]
                
            # Check if node_sizes is available
            if self.canvas.node_sizes is None:
                logger.warning("No node sizes available, using default size")
                visible_sizes = np.ones(len(visible_indices)) * 5  # Default size
            else:
                visible_sizes = self.canvas.node_sizes[visible_indices]
            
            logger.info(f"Showing {len(visible_positions)} nodes")

            # Update the scatter visual
            if len(visible_positions) > 0:
                self.canvas.scatter.set_data(
                    pos=visible_positions, face_color=visible_colors, size=visible_sizes
                )
            else:
                # If no nodes are visible, set empty data
                logger.warning("No nodes visible, setting empty data")
                self.canvas.scatter.set_data(
                    pos=np.zeros((0, 3)), face_color=np.zeros((0, 4)), size=np.zeros(0)
                )
        except Exception as e:
            logger.error(f"Error updating node visibility: {str(e)}")
            logger.error(f"node_mask type: {type(node_mask)}, shape: {node_mask.shape if hasattr(node_mask, 'shape') else 'N/A'}")
            logger.error(f"node_positions shape: {self.canvas.node_positions.shape if hasattr(self.canvas.node_positions, 'shape') else 'N/A'}")
            logger.error(f"node_colors_rgba shape: {self.canvas.node_colors_rgba.shape if hasattr(self.canvas.node_colors_rgba, 'shape') else 'N/A'}")
            
            # Fallback: show all nodes
            logger.warning("Falling back to showing all nodes")
            self.canvas.scatter.set_data(
                pos=self.canvas.node_positions,
                face_color=self.canvas.node_colors_rgba,
                size=self.canvas.node_sizes
            )
