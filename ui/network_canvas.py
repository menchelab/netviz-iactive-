import numpy as np
from vispy import scene
from vispy.scene.visuals import Markers, Line, Text, Rectangle
import logging
from utils.color_utils import hex_to_rgba

class NetworkCanvas:
    def __init__(self, parent=None, data_manager=None):
        logger = logging.getLogger(__name__)
        logger.info("Creating canvas...")

        # Store reference to data manager
        self.data_manager = data_manager
        if not data_manager:
            logger.error("NetworkCanvas requires a data_manager")
            raise ValueError("NetworkCanvas requires a data_manager")

        # Create canvas and view
        self.canvas = scene.SceneCanvas(keys='interactive', size=(1200, 768))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.fov = 45
        self.view.camera.distance = 3

        # Create visuals
        self.scatter = Markers()
        self.view.add(self.scatter)

        # Create separate line visuals for intralayer and interlayer edges
        self.intralayer_lines = Line(pos=np.zeros((0, 3)), color=np.zeros((0, 4)),
                                   connect='segments', width=1)
        self.view.add(self.intralayer_lines)

        self.interlayer_lines = Line(pos=np.zeros((0, 3)), color=np.zeros((0, 4)),
                                   connect='segments', width=2)
        self.view.add(self.interlayer_lines)
        
        # Create text labels for nodes
        self.node_labels = Text(pos=np.array([[0, 0, 0]]), text=[''], color='white', font_size=6)
        self.node_labels.visible = False
        self.view.add(self.node_labels)
        
        # Create line visuals for bar charts
        self.node_count_bars = Line(pos=np.zeros((0, 3)), color=np.zeros((0, 4)),
                                  connect='segments', width=3)
        self.node_count_bars.visible = False
        self.view.add(self.node_count_bars)
        
        self.edge_count_bars = Line(pos=np.zeros((0, 3)), color=np.zeros((0, 4)),
                                  connect='segments', width=3)
        self.edge_count_bars.visible = False
        self.view.add(self.edge_count_bars)
        
        # Store current visibility state
        self.current_node_mask = None
        self.current_edge_mask = None
        self.visible_layers = None

    def load_data(self, node_positions=None, link_pairs=None, link_colors=None, node_colors=None, node_ids=None):
        """Load data either directly or from the data manager"""
        if self.data_manager is not None:
            # Use data from the manager
            self.node_positions = self.data_manager.node_positions
            self.link_pairs = self.data_manager.link_pairs
            self.link_colors_rgba = self.data_manager.link_colors_rgba
            self.node_colors_rgba = self.data_manager.node_colors_rgba
            self.node_sizes = self.data_manager.node_sizes
            self.active_nodes = self.data_manager.active_nodes
            self.node_ids = self.data_manager.node_ids
            self.layer_names = self.data_manager.layer_names
            self.nodes_per_layer = self.data_manager.nodes_per_layer
        else:
            # Use directly provided data
            self.node_positions = node_positions
            self.link_pairs = link_pairs
            self.node_ids = node_ids

            # Convert link colors from hex to RGBA with enhanced saturation
            self.link_colors_rgba = self._enhance_colors(link_colors)

            # Initialize node colors with white
            self.node_colors_rgba = np.ones((len(node_positions), 4))

            if node_colors:
                for i, color_hex in enumerate(node_colors):
                    self.node_colors_rgba[i] = hex_to_rgba(color_hex)
            
            # Initialize node sizes array - default size for all nodes
            self.node_sizes = np.ones(len(node_positions)) * 3
            
            # Determine which nodes actually exist in each layer based on edges
            self.active_nodes = np.zeros(len(node_positions), dtype=bool)
            for start_idx, end_idx in self.link_pairs:
                self.active_nodes[start_idx] = True
                self.active_nodes[end_idx] = True
            
            # Set larger size for active nodes
            self.node_sizes[self.active_nodes] = 9  # 3x larger

    def _enhance_colors(self, colors):
        """Enhance color saturation and convert hex to RGBA"""
        enhanced_colors = []
        for color in colors:
            rgba = hex_to_rgba(color, alpha=0.9)
            max_val = max(rgba[0], rgba[1], rgba[2])
            min_val = min(rgba[0], rgba[1], rgba[2])

            if max_val - min_val < 0.3:
                dominant_idx = np.argmax(rgba[:3])
                for i in range(3):
                    if i == dominant_idx:
                        rgba[i] = min(1.0, rgba[i] * 1.3)
                    else:
                        rgba[i] = max(0.0, rgba[i] * 0.7)

            enhanced_colors.append(rgba)
        return enhanced_colors

    def set_layer_colors(self, layer_colors):
        """Set the layer colors mapping"""
        logger = logging.getLogger(__name__)
        logger.debug(f"Setting layer colors: {layer_colors}")
        
        self.layer_colors = layer_colors
        
        # Convert hex colors to RGBA
        self.layer_colors_rgba = {}
        for layer_name, color_hex in layer_colors.items():
            rgba = hex_to_rgba(color_hex, alpha=1.0)
            self.layer_colors_rgba[layer_name] = rgba
            logger.debug(f"Layer {layer_name}: {color_hex} -> {rgba}")

    def update_visibility(self, node_mask=None, edge_mask=None, show_intralayer=True, show_nodes=True, show_labels=True, bottom_labels_only=True, show_stats_bars=False):
        """Update the visibility of nodes and edges based on masks"""
        logger = logging.getLogger(__name__)
        logger.info(f"Updating visibility with show_intralayer={show_intralayer}, show_nodes={show_nodes}, show_labels={show_labels}, show_stats_bars={show_stats_bars}")
        
        # Use masks from data manager if available and not provided
        if node_mask is None and self.data_manager is not None:
            node_mask = self.data_manager.current_node_mask
        if edge_mask is None and self.data_manager is not None:
            edge_mask = self.data_manager.current_edge_mask
        
        # Store the current masks for later use
        self.current_node_mask = node_mask
        self.current_edge_mask = edge_mask
        
        # Get interlayer edge counts from data manager if available
        if self.data_manager is not None:
            interlayer_edge_counts = self.data_manager.get_interlayer_edge_counts()
        else:
            interlayer_edge_counts = self._calculate_interlayer_edge_counts(edge_mask)
        
        # Update node visibility
        self._update_node_visibility(node_mask, show_nodes)
        
        # Update labels and bar charts
        self._update_labels_and_bars(node_mask, interlayer_edge_counts, show_labels, show_stats_bars)
        
        # Update edge visibility
        self._update_edge_visibility(edge_mask, show_intralayer)

    def _calculate_interlayer_edge_counts(self, edge_mask):
        """Calculate interlayer edge counts for each node"""
        interlayer_edge_counts = {}
        
        if self.link_pairs is None or self.node_ids is None or self.nodes_per_layer is None:
            return interlayer_edge_counts
            
        for i, (start_idx, end_idx) in enumerate(self.link_pairs):
            # Skip edges that aren't visible
            if not edge_mask[i]:
                continue
                
            # Skip intralayer edges
            start_layer = start_idx // self.nodes_per_layer
            end_layer = end_idx // self.nodes_per_layer
            if start_layer == end_layer:
                continue
            
            # Count this edge for both source and destination nodes
            src_id = self.node_ids[start_idx].split('_')[0]
            dst_id = self.node_ids[end_idx].split('_')[0]
            
            interlayer_edge_counts[src_id] = interlayer_edge_counts.get(src_id, 0) + 1
            interlayer_edge_counts[dst_id] = interlayer_edge_counts.get(dst_id, 0) + 1
        
        return interlayer_edge_counts

    def _update_node_visibility(self, node_mask, show_nodes):
        """Update the visibility of nodes"""
        logger = logging.getLogger(__name__)
        logger.info(f"Updating node visibility: show_nodes={show_nodes}")
        
        if not show_nodes:
            logger.info("Hiding nodes as show_nodes is False")
            self.scatter.visible = False
            return
        
        # Filter positions and colors based on node mask
        if np.any(node_mask):
            visible_positions = self.data_manager.node_positions[node_mask]
            visible_colors = self.data_manager.node_colors_rgba[node_mask]
            visible_sizes = self.data_manager.node_sizes[node_mask]
            
            self.scatter.set_data(
                pos=visible_positions,
                face_color=visible_colors,
                size=visible_sizes,
                edge_width=0
            )
            self.scatter.visible = True
            logger.info(f"Showing {len(visible_positions)} nodes")
        else:
            logger.info("No nodes to show after applying mask")
            self.scatter.visible = False

    def _update_labels_and_bars(self, node_mask, interlayer_edge_counts, show_labels, show_stats_bars):
        """Update node labels and bar charts"""
        logger = logging.getLogger(__name__)
        logger.info(f"Updating labels and bars: show_labels={show_labels}, show_stats_bars={show_stats_bars}")
        
        # If not showing labels or bars, hide them and return early
        if not show_labels and not show_stats_bars:
            self.node_labels.visible = False
            self.node_count_bars.visible = False
            self.edge_count_bars.visible = False
            return
        
        if (self.data_manager.node_ids is None or self.data_manager.visible_layers is None or 
            self.data_manager.layer_names is None or not self.data_manager.visible_layers):
            # Hide everything if we don't have the necessary data
            self.node_labels.visible = False
            self.node_count_bars.visible = False
            self.edge_count_bars.visible = False
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
        top_layer_idx = min(self.data_manager.visible_layers) if self.data_manager.visible_layers else -1
        logger.info(f"Top visible layer index: {top_layer_idx}")
        
        # Track which base nodes we've already processed
        processed_base_nodes = set()
        
        # Process visible nodes in the top layer first
        visible_indices = np.where(node_mask)[0]
        
        # Group nodes by their base ID
        base_node_to_indices = {}
        for idx in visible_indices:
            node_id = self.data_manager.node_ids[idx]
            base_node = node_id.split('_')[0] if '_' in node_id else node_id
            
            if base_node not in base_node_to_indices:
                base_node_to_indices[base_node] = []
            base_node_to_indices[base_node].append(idx)
        
        # For each base node, find the node in the top-most visible layer
        for base_node, indices in base_node_to_indices.items():
            # Find the node in the top-most layer
            top_node_idx = None
            top_node_layer = float('inf')
            
            for idx in indices:
                node_layer_idx = idx // self.data_manager.nodes_per_layer
                if node_layer_idx in self.data_manager.visible_layers and node_layer_idx < top_node_layer:
                    top_node_idx = idx
                    top_node_layer = node_layer_idx
            
            if top_node_idx is None:
                continue  # No visible nodes for this base node
            
            # Process only the top node for this base node
            idx = top_node_idx
            node_layer_idx = idx // self.data_manager.nodes_per_layer
            
            # Get node ID and base node
            node_id = self.data_manager.node_ids[idx]
            label_text = node_id.split('_')[0] if '_' in node_id else node_id
            
            # Count active nodes with this base ID across all layers
            active_node_count = self.data_manager.count_active_nodes_for_base(base_node, idx)
            
            # Get edge count if available
            edge_count = interlayer_edge_counts.get(base_node, 0)
            
            # Update labels if needed
            if show_labels:
                # Format label text with counts
                if active_node_count > 0 or edge_count > 0:
                    label_text_with_counts = f"{label_text} [{active_node_count}/{edge_count//2}]"
                else:
                    label_text_with_counts = label_text
                    
                # Add a small offset to position labels better
                pos = self.data_manager.node_positions[idx].copy()
                pos[1] += 0.01  # Offset in y direction
                label_positions.append(pos)
                label_texts.append(label_text_with_counts)
                
                # Get the layer name for this node
                layer_name = self.data_manager.layer_names[node_layer_idx]
                
                # Use the layer color from the data manager
                if self.data_manager.layer_colors_rgba and layer_name in self.data_manager.layer_colors_rgba:
                    label_color = self.data_manager.layer_colors_rgba[layer_name].copy()
                    # Make labels with 0 interlayer connections more transparent
                    if edge_count == 0:
                        label_color[3] = 0.6  # 60% opacity
                else:
                    label_color = np.array([1.0, 1.0, 0.0, 1.0])
                    if edge_count == 0:
                        label_color[3] = 0.6
                
                label_colors.append(label_color)
            
            # Update bar charts if needed
            if show_stats_bars and (active_node_count > 0 or edge_count > 0):
                # Scale factors for bar widths
                max_bar_width = 0.1
                node_count_width = min(max_bar_width, 0.005 * active_node_count)
                edge_count_width = min(max_bar_width, 0.005 * (edge_count//2))
                
                # Node count bar (top bar)
                if node_count_width > 0:
                    start_point = self.data_manager.node_positions[idx].copy()
                    end_point = start_point.copy()
                    end_point[0] += node_count_width
                    
                    node_count_line_points.append(start_point)
                    node_count_line_points.append(end_point)
                    
                    node_color = np.array([1.0, 0.0, 1.0, 1])  # Magenta for node count
                    node_count_line_colors.append(node_color)
                    node_count_line_colors.append(node_color)
                
                # Edge count bar (bottom bar)
                if edge_count_width > 0:
                    start_point = self.data_manager.node_positions[idx].copy()
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
            self.node_labels.pos = np.array(label_positions)
            self.node_labels.text = label_texts
            self.node_labels.color = np.array(label_colors)
            self.node_labels.visible = True
            self.node_labels.order = 1  # Higher order means drawn later (on top)
        else:
            self.node_labels.visible = False
        
        # Update the bar chart visuals
        if show_stats_bars and node_count_line_points:
            self.node_count_bars.set_data(
                pos=np.array(node_count_line_points),
                color=np.array(node_count_line_colors),
                connect='segments',
                width=3
            )
            self.node_count_bars.visible = True
            self.node_count_bars.order = 0.8  # Draw behind labels
        else:
            self.node_count_bars.visible = False
        
        if show_stats_bars and edge_count_line_points:
            self.edge_count_bars.set_data(
                pos=np.array(edge_count_line_points),
                color=np.array(edge_count_line_colors),
                connect='segments',
                width=3
            )
            self.edge_count_bars.visible = True
            self.edge_count_bars.order = 0.9  # Draw behind labels but above node count bars
        else:
            self.edge_count_bars.visible = False
        
        logger.info(f"Label visibility set to {self.node_labels.visible}, showing {len(label_positions)} labels")
        logger.info(f"Bar visibility set to {self.node_count_bars.visible} and {self.edge_count_bars.visible}")

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
        visible_edges = self.link_pairs[edge_mask]
        visible_edge_colors = [self.link_colors_rgba[i] for i, mask in enumerate(edge_mask) if mask]
        
        for i, (start_idx, end_idx) in enumerate(visible_edges):
            # Check if this is a horizontal (intra-layer) edge or interlayer edge
            start_z = self.node_positions[start_idx][2]
            end_z = self.node_positions[end_idx][2]
            is_horizontal = abs(start_z - end_z) < 0.001
            
            if is_horizontal:
                intralayer_count += 1
                # Only add intralayer edges if they should be shown
                if show_intralayer:
                    self._add_intralayer_edge(start_idx, end_idx, i, intralayer_pos, intralayer_colors)
            else:
                interlayer_count += 1
                self._add_interlayer_edge(start_idx, end_idx, i, interlayer_edges)
        
        logger.info(f"Found {intralayer_count} intralayer edges and {interlayer_count} interlayer edges")
        
        # Process interlayer edges with offsets
        interlayer_pos, interlayer_colors = self._process_interlayer_edges(interlayer_edges)
        
        # Update intralayer lines
        self._set_intralayer_lines(intralayer_pos, intralayer_colors)
        
        # Update interlayer lines
        self._set_interlayer_lines(interlayer_pos, interlayer_colors)

    def _add_intralayer_edge(self, start_idx, end_idx, edge_idx, intralayer_pos, intralayer_colors):
        """Add an intralayer edge to the data"""
        # Get the layer for this edge
        layer_idx = start_idx // self.data_manager.nodes_per_layer
        layer_name = self.data_manager.layer_names[layer_idx]
        
        # Use the layer color from the data manager with lower opacity
        if self.data_manager.layer_colors_rgba and layer_name in self.data_manager.layer_colors_rgba:
            edge_color = self.data_manager.layer_colors_rgba[layer_name].copy()
            edge_color[3] = 0.3  # Set opacity to 30% for horizontal edges
        else:
            # Use the original color with lower opacity if no mapping is found
            edge_color = self.data_manager.link_colors_rgba[edge_idx].copy()
            edge_color[3] = 0.3
        
        intralayer_pos.append(self.data_manager.node_positions[start_idx])
        intralayer_pos.append(self.data_manager.node_positions[end_idx])
        intralayer_colors.append(edge_color)
        intralayer_colors.append(edge_color)

    def _add_interlayer_edge(self, start_idx, end_idx, edge_idx, interlayer_edges):
        """Add an interlayer edge to the data"""
        # Inter-layer edge: use the color of the starting layer with higher opacity
        start_layer_idx = start_idx // self.data_manager.nodes_per_layer
        start_layer_name = self.data_manager.layer_names[start_layer_idx]
        
        # Use the layer color from the data manager with higher opacity
        if self.data_manager.layer_colors_rgba and start_layer_name in self.data_manager.layer_colors_rgba:
            edge_color = self.data_manager.layer_colors_rgba[start_layer_name].copy()
            edge_color[3] = 0.8  # Higher opacity for interlayer edges
        else:
            # Use the original color with higher opacity if no mapping is found
            edge_color = self.data_manager.link_colors_rgba[edge_idx].copy()
            edge_color[3] = 0.8
        
        # Store the edge for later processing
        interlayer_edges.append((
            self.data_manager.node_positions[start_idx].copy(),
            self.data_manager.node_positions[end_idx].copy(),
            edge_color
        ))

    def _process_interlayer_edges(self, interlayer_edges):
        """Process interlayer edges to add offsets for overlapping edges"""
        interlayer_pos = []
        interlayer_colors = []
        
        # Group edges by their base node (x,y coordinates)
        edge_groups = {}
        for start_pos, end_pos, color in interlayer_edges:
            # Create a key based on x,y coordinates (ignoring z)
            # Round to 3 decimal places to handle floating point precision issues
            key = (round(start_pos[0], 3), round(start_pos[1], 3), 
                   round(end_pos[0], 3), round(end_pos[1], 3))
            
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
                    offset_scale = 0.002
                    
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
            self.intralayer_lines.set_data(
                pos=np.array(intralayer_pos),
                color=np.array(intralayer_colors),
                width=1
            )
        else:
            # Create a dummy point that's invisible
            self.intralayer_lines.set_data(
                pos=np.array([[0, 0, 0], [0, 0, 0]]),
                color=np.array([[0, 0, 0, 0], [0, 0, 0, 0]]),
                width=1
            )

    def _set_interlayer_lines(self, interlayer_pos, interlayer_colors):
        """Update the interlayer lines visual"""
        if interlayer_pos:
            self.interlayer_lines.set_data(
                pos=np.array(interlayer_pos),
                color=np.array(interlayer_colors),
                width=2  # Thicker width for interlayer edges
            )
        else:
            # Create a dummy point that's invisible
            self.interlayer_lines.set_data(
                pos=np.array([[0, 0, 0], [0, 0, 0]]),
                color=np.array([[0, 0, 0, 0], [0, 0, 0, 0]]),
                width=1
            )