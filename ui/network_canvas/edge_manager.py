import numpy as np
import logging


class EdgeManager:
    def __init__(self, canvas):
        self.canvas = canvas

    def _add_intralayer_edge(
        self, start_idx, end_idx, edge_idx, intralayer_pos, intralayer_colors
    ):
        """Add an intralayer edge to the data"""
        # Get the layer for this edge
        layer_idx = start_idx // self.canvas.data_manager.nodes_per_layer
        layer_name = self.canvas.data_manager.layer_names[layer_idx]

        # Use the layer color from the data manager with lower opacity
        if (
            self.canvas.data_manager.layer_colors_rgba
            and layer_name in self.canvas.data_manager.layer_colors_rgba
        ):
            edge_color = self.canvas.data_manager.layer_colors_rgba[layer_name].copy()
            edge_color[3] = 0.3  # Set opacity to 30% for horizontal edges
        else:
            # Use the original color with lower opacity if no mapping is found
            edge_color = self.canvas.data_manager.link_colors_rgba[edge_idx].copy()
            edge_color[3] = 0.3

        intralayer_pos.append(self.canvas.data_manager.node_positions[start_idx])
        intralayer_pos.append(self.canvas.data_manager.node_positions[end_idx])
        intralayer_colors.append(edge_color)
        intralayer_colors.append(edge_color)

    def _add_interlayer_edge(self, start_idx, end_idx, edge_idx, interlayer_edges):
        """Add an interlayer edge to the data"""
        # Inter-layer edge: use the color of the starting layer with higher opacity
        start_layer_idx = start_idx // self.canvas.data_manager.nodes_per_layer
        start_layer_name = self.canvas.data_manager.layer_names[start_layer_idx]

        # Use the layer color from the data manager with higher opacity
        if (
            self.canvas.data_manager.layer_colors_rgba
            and start_layer_name in self.canvas.data_manager.layer_colors_rgba
        ):
            edge_color = self.canvas.data_manager.layer_colors_rgba[
                start_layer_name
            ].copy()
            edge_color[3] = 0.8  # Higher opacity for interlayer edges
        else:
            # Use the original color with higher opacity if no mapping is found
            edge_color = self.canvas.data_manager.link_colors_rgba[edge_idx].copy()
            edge_color[3] = 0.8

        # Store the edge for later processing
        interlayer_edges.append(
            (
                self.canvas.data_manager.node_positions[start_idx].copy(),
                self.canvas.data_manager.node_positions[end_idx].copy(),
                edge_color,
            )
        )

    def _process_interlayer_edges(self, interlayer_edges):
        """Process interlayer edges to add offsets for overlapping edges"""
        interlayer_pos = []
        interlayer_colors = []

        # Group edges by their base node (x,y coordinates)
        edge_groups = {}
        for start_pos, end_pos, color in interlayer_edges:
            # Create a key based on x,y coordinates (ignoring z)
            # Round to 3 decimal places to handle floating point precision issues
            key = (
                round(start_pos[0], 3),
                round(start_pos[1], 3),
                round(end_pos[0], 3),
                round(end_pos[1], 3),
            )

            if key in edge_groups:
                edge_groups[key].append((start_pos, end_pos, color))
            else:
                edge_groups[key] = [(start_pos, end_pos, color)]

        # Now apply offsets to each group
        for key, edges in edge_groups.items():
            # If there's only one edge in this group, no need for offset
            if len(edges) == 1:
                start_pos, end_pos, color = edges[0]
                interlayer_pos.append(start_pos)
                interlayer_pos.append(end_pos)
                interlayer_colors.append(color)
                interlayer_colors.append(color)
            else:
                # Apply different offsets to each edge in the group
                for i, (start_pos, end_pos, color) in enumerate(edges):
                    # Calculate offset direction based on index
                    offset_scale = 0.001 + len(edges) * 0.0005 * 0.1

                    # Calculate offset direction in a circular pattern
                    angle = 2 * np.pi * i / len(edges)
                    offset_x = offset_scale * np.cos(angle)
                    offset_y = offset_scale * np.sin(angle)

                    # Apply offset to both start and end positions
                    start_pos_offset = start_pos.copy()
                    end_pos_offset = end_pos.copy()

                    start_pos_offset[0] += offset_x
                    start_pos_offset[1] += offset_y
                    end_pos_offset[0] += offset_x
                    end_pos_offset[1] += offset_y

                    interlayer_pos.append(start_pos_offset)
                    interlayer_pos.append(end_pos_offset)
                    interlayer_colors.append(color)
                    interlayer_colors.append(color)

        return interlayer_pos, interlayer_colors

    def _set_intralayer_lines(self, intralayer_pos, intralayer_colors):
        """Update the intralayer lines visual"""
        if intralayer_pos:
            self.canvas.intralayer_lines.set_data(
                pos=np.array(intralayer_pos), color=np.array(intralayer_colors), width=1
            )
        else:
            # Create a dummy point that's invisible
            self.canvas.intralayer_lines.set_data(
                pos=np.array([[0, 0, 0], [0, 0, 0]]),
                color=np.array([[0, 0, 0, 0], [0, 0, 0, 0]]),
                width=1,
            )

    def _set_interlayer_lines(self, interlayer_pos, interlayer_colors):
        """Update the interlayer lines visual"""
        if interlayer_pos:
            self.canvas.interlayer_lines.set_data(
                pos=np.array(interlayer_pos),
                color=np.array(interlayer_colors),
                width=1,  # Thicker width for interlayer edges
            )
        else:
            # Create a dummy point that's invisible
            self.canvas.interlayer_lines.set_data(
                pos=np.array([[0, 0, 0], [0, 0, 0]]),
                color=np.array([[0, 0, 0, 0], [0, 0, 0, 0]]),
                width=1,
            )

    def _update_edge_visibility(self, edge_mask, show_intralayer):
        """Update the visibility of edges"""
        logger = logging.getLogger(__name__)
        
        # Handle case when edge_mask is None
        if edge_mask is None:
            logger.warning("Edge mask is None, showing all edges")
            edge_mask = np.ones(len(self.canvas.link_pairs), dtype=bool)
        
        # Ensure edge_mask is a boolean array with the correct length
        if not isinstance(edge_mask, np.ndarray) or len(edge_mask) != len(self.canvas.link_pairs):
            logger.warning(f"Invalid edge mask, showing all edges. Mask type: {type(edge_mask)}, length: {len(edge_mask) if hasattr(edge_mask, '__len__') else 'N/A'}")
            edge_mask = np.ones(len(self.canvas.link_pairs), dtype=bool)
        
        # Convert to boolean if not already
        if edge_mask.dtype != bool:
            logger.warning(f"Converting edge mask from {edge_mask.dtype} to boolean")
            edge_mask = edge_mask.astype(bool)

        # Separate intralayer and interlayer edges
        intralayer_pos = []
        intralayer_colors = []
        interlayer_edges = []  # For interlayer edges, store (start_pos, end_pos, color)

        intralayer_count = 0
        interlayer_count = 0

        try:
            # Process all edges
            visible_indices = np.where(edge_mask)[0]
            logger.info(f"Visible edge indices: {len(visible_indices)} out of {len(edge_mask)}")
            
            for i in visible_indices:
                start_idx, end_idx = self.canvas.link_pairs[i]
                
                # Check if this is a horizontal (intra-layer) edge or interlayer edge
                start_z = self.canvas.node_positions[start_idx][2]
                end_z = self.canvas.node_positions[end_idx][2]
                is_horizontal = abs(start_z - end_z) < 0.001

                if is_horizontal:
                    intralayer_count += 1
                    # Only add intralayer edges if they should be shown
                    if show_intralayer:
                        self._add_intralayer_edge(
                            start_idx, end_idx, i, intralayer_pos, intralayer_colors
                        )
                else:
                    interlayer_count += 1
                    self._add_interlayer_edge(start_idx, end_idx, i, interlayer_edges)

            logger.info(f"Processed {intralayer_count} intralayer edges and {interlayer_count} interlayer edges")

            # Process interlayer edges
            interlayer_pos, interlayer_colors = self._process_interlayer_edges(interlayer_edges)

            # Update the line visuals
            self._set_intralayer_lines(intralayer_pos, intralayer_colors)
            self._set_interlayer_lines(interlayer_pos, interlayer_colors)
        except Exception as e:
            logger.error(f"Error updating edge visibility: {str(e)}")
            logger.error(f"edge_mask type: {type(edge_mask)}, shape: {edge_mask.shape if hasattr(edge_mask, 'shape') else 'N/A'}")
            logger.error(f"link_pairs shape: {self.canvas.link_pairs.shape if hasattr(self.canvas.link_pairs, 'shape') else 'N/A'}")
            
            # Fallback: hide all edges
            logger.warning("Falling back to hiding all edges")
            self._set_intralayer_lines([], [])
            self._set_interlayer_lines([], [])
