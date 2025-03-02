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

        # Separate intralayer and interlayer edges
        intralayer_pos = []
        intralayer_colors = []
        interlayer_edges = []  # For interlayer edges, store (start_pos, end_pos, color)

        intralayer_count = 0
        interlayer_count = 0

        # Process all edges
        visible_edges = self.canvas.link_pairs[edge_mask]

        for i, (start_idx, end_idx) in enumerate(visible_edges):
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

        logger.info(
            f"Found {intralayer_count} intralayer edges and {interlayer_count} interlayer edges"
        )

        # Process interlayer edges with offsets
        interlayer_pos, interlayer_colors = self._process_interlayer_edges(
            interlayer_edges
        )

        # Update intralayer lines
        self._set_intralayer_lines(intralayer_pos, intralayer_colors)

        # Update interlayer lines
        self._set_interlayer_lines(interlayer_pos, interlayer_colors)

    def update_with_optimized_data(self, edge_connections, edge_colors, edge_importance=None, show_intralayer=True):
        """
        Update edges with optimized data from NetworkDataManager.
        
        Parameters:
        -----------
        edge_connections : numpy.ndarray
            Array of edge connections (start_idx, end_idx)
        edge_colors : numpy.ndarray
            Array of edge colors (RGBA)
        edge_importance : numpy.ndarray, optional
            Array of edge importance scores for level-of-detail rendering
        show_intralayer : bool
            Whether to show intralayer edges
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Validate input data
            if edge_connections is None or len(edge_connections) == 0:
                logger.warning("No edge connections provided")
                self.canvas.interlayer_lines.visible = False
                self.canvas.intralayer_lines.visible = False
                return
                
            if edge_colors is None or len(edge_colors) != len(edge_connections):
                logger.warning(f"Invalid edge colors: expected {len(edge_connections)}, got {len(edge_colors) if edge_colors is not None else 0}")
                # Use default colors if none provided
                edge_colors = np.ones((len(edge_connections), 4)) * np.array([0.7, 0.7, 0.7, 0.5])
            
            # Store the data for later use
            self.canvas.link_pairs = edge_connections
            self.canvas.link_colors_rgba = edge_colors
            
            # Separate intralayer and interlayer edges
            if show_intralayer:
                # Process all edges with the _update_edge_visibility method
                self._update_edge_visibility(np.ones(len(edge_connections), dtype=bool), show_intralayer)
            else:
                # If intralayer edges are hidden, only process interlayer edges
                # Get node positions from the canvas
                node_positions = self.canvas.node_positions
                
                # Identify interlayer edges
                interlayer_mask = np.zeros(len(edge_connections), dtype=bool)
                for i, (start_idx, end_idx) in enumerate(edge_connections):
                    if start_idx < len(node_positions) and end_idx < len(node_positions):
                        start_z = node_positions[start_idx][2]
                        end_z = node_positions[end_idx][2]
                        is_interlayer = abs(start_z - end_z) >= 0.001
                        interlayer_mask[i] = is_interlayer
                
                # Process only interlayer edges
                self._update_edge_visibility(interlayer_mask, False)
            
            logger.debug(f"Updated edges with optimized data: {len(edge_connections)} edges")
        except Exception as e:
            logger.error(f"Error updating edges with optimized data: {e}")
            import traceback
            traceback.print_exc()
