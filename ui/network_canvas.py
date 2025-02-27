import numpy as np
from vispy import scene
from vispy.scene.visuals import Markers, Line, Text, Rectangle
import logging
from utils.color_utils import hex_to_rgba

class NetworkCanvas:
    def __init__(self, parent=None):
        logger = logging.getLogger(__name__)
        logger.info("Creating canvas...")

        self.canvas = scene.SceneCanvas(keys='interactive', size=(1200, 768))

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.fov = 45
        self.view.camera.distance = 8

        self.scatter = Markers()
        self.view.add(self.scatter)

        # Create separate line visuals for intralayer and interlayer edges
        self.intralayer_lines = Line(pos=np.zeros((0, 3)), color=np.zeros((0, 4)),
                                   connect='segments', width=1)
        self.view.add(self.intralayer_lines)

        self.interlayer_lines = Line(pos=np.zeros((0, 3)), color=np.zeros((0, 4)),
                                   connect='segments', width=2)  # Thicker width
        self.view.add(self.interlayer_lines)
        
        # Create text labels for nodes - initialize with a dummy position that will be invisible
        self.node_labels = Text(pos=np.array([[0, 0, 0]]), text=[''], color='white', font_size=6)
        self.node_labels.visible = False  # Start with labels invisible
        self.view.add(self.node_labels)
        
        # Create bar charts using Line visuals instead of Rectangle
        self.node_count_bars = Line(pos=np.zeros((0, 3)), color=np.zeros((0, 4)),
                                   connect='segments', width=3)  # Thicker width for visibility
        self.view.add(self.node_count_bars)
        
        self.edge_count_bars = Line(pos=np.zeros((0, 3)), color=np.zeros((0, 4)),
                                   connect='segments', width=3)  # Thicker width for visibility
        self.view.add(self.edge_count_bars)

        self.node_positions = None
        self.link_pairs = None
        self.link_colors_rgba = None
        self.node_colors_rgba = None
        self.node_sizes = None
        self.active_nodes = None
        self.node_ids = None
        self.visible_layers = None
        self.layer_names = None
        self.nodes_per_layer = None
        self.node_mask = None
        self.layer_colors = None  # Add this to store layer colors
        self.layer_colors_rgba = None  # Add this to store layer colors in RGBA format

    def load_data(self, node_positions, link_pairs, link_colors, node_colors=None, node_ids=None):
        self.node_positions = node_positions
        self.link_pairs = link_pairs
        self.node_ids = node_ids

        # Convert link colors from hex to RGBA with enhanced saturation
        self.link_colors_rgba = []
        for color in link_colors:
            rgba = hex_to_rgba(color, alpha=0.9)  # Increased opacity
            max_val = max(rgba[0], rgba[1], rgba[2])
            min_val = min(rgba[0], rgba[1], rgba[2])

            if max_val - min_val < 0.3:
                dominant_idx = np.argmax(rgba[:3])
                for i in range(3):
                    if i == dominant_idx:
                        rgba[i] = min(1.0, rgba[i] * 1.3)
                    else:
                        rgba[i] = max(0.0, rgba[i] * 0.7)

            self.link_colors_rgba.append(rgba)

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

    def update_visibility(self, node_mask, edge_mask, show_intralayer=True, show_nodes=True, show_labels=True, bottom_labels_only=True, show_stats_bars=False):
        """Update the visibility of nodes and edges based on masks"""
        logger = logging.getLogger(__name__)
        logger.info(f"Updating visibility with show_intralayer={show_intralayer}, show_nodes={show_nodes}, show_labels={show_labels}, show_stats_bars={show_stats_bars}")
        
        # Store the current masks for later use
        self.current_node_mask = node_mask
        self.current_edge_mask = edge_mask
        
        # Calculate interlayer edge counts for each node
        interlayer_edge_counts = {}
        if self.link_pairs is not None and self.node_ids is not None:
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
        
        # Update node visibility
        if self.node_positions is not None:
            # Update the scatter plot with visible nodes
            visible_positions = self.node_positions[node_mask]
            visible_colors = self.node_colors_rgba[node_mask]
            visible_sizes = self.node_sizes[node_mask]
            
            if len(visible_positions) > 0 and show_nodes:
                self.scatter.set_data(
                    pos=visible_positions,
                    size=visible_sizes,
                    edge_width=0,
                    face_color=visible_colors
                )
                self.scatter.visible = True
            else:
                # Hide nodes
                self.scatter.visible = False
        
        # Prepare data for labels and bar charts
        if self.node_ids is not None and self.visible_layers is not None and self.layer_names is not None:
            if len(self.visible_layers) > 0:
                # Get nodes in all visible layers
                label_positions = []
                label_texts = []
                label_colors = []
                
                # Lists for bar chart data
                bar_positions = []
                node_count_widths = []
                edge_count_widths = []
                node_count_colors = []
                edge_count_colors = []
                bar_visible_indices = []  # Changed from visible_indices to bar_visible_indices
                
                # Determine the top layer index (the lowest layer index that is visible)
                top_layer_idx = min(self.visible_layers) if self.visible_layers else -1
                
                # Process visible nodes
                visible_indices = np.where(node_mask)[0]
                
                # Create a set to track which base nodes we've already labeled
                labeled_base_nodes = set()
                
                for idx in visible_indices:
                    node_layer_idx = idx // self.nodes_per_layer
                    
                    # Only show labels for nodes in the top layer
                    if node_layer_idx != top_layer_idx:
                        # If this is not the top layer
                        base_node = self.node_ids[idx].split('_')[0] if '_' in self.node_ids[idx] else self.node_ids[idx]
                        
                        # If we've already labeled this base node, skip it
                        if base_node in labeled_base_nodes:
                            continue
                        
                        # Check if this base node exists in the top layer
                        top_layer_node_idx = top_layer_idx * self.nodes_per_layer + (idx % self.nodes_per_layer)
                        
                        # If the node exists in the top layer (regardless of active status), skip it
                        # We'll show it when we process the top layer
                        if top_layer_node_idx < len(self.node_positions) and node_mask[top_layer_node_idx]:
                            continue
                        
                        # Mark this base node as labeled
                        labeled_base_nodes.add(base_node)
                    
                    # Get node ID and base node
                    node_id = self.node_ids[idx]
                    # Extract the part before the first underscore for the label
                    if '_' in node_id:
                        label_text = node_id.split('_')[0]
                    else:
                        label_text = node_id
                    
                    base_node = label_text
                    
                    # Add interlayer edge count if available
                    if base_node in interlayer_edge_counts:
                        edge_count = interlayer_edge_counts[base_node]
                        
                        # Count active nodes with this base ID across all layers
                        active_node_count = 0
                        for current_layer_idx in self.visible_layers:
                            node_idx_in_layer = current_layer_idx * self.nodes_per_layer + (idx % self.nodes_per_layer)
                            if (node_idx_in_layer < len(self.node_positions) and 
                                node_mask[node_idx_in_layer] and 
                                self.active_nodes[node_idx_in_layer]):
                                active_node_count += 1
                        
                        # For labels, add both counts to the label text
                        if show_labels:
                            label_text_with_counts = f"{label_text} [{active_node_count}/{edge_count//2}]"
                            
                            # Add a small offset to position labels better
                            pos = self.node_positions[idx].copy()
                            # Offset in y direction
                            pos[1] += 0.01
                            label_positions.append(pos)
                            label_texts.append(label_text_with_counts)
                            
                            # Get the layer name for this node
                            layer_name = self.layer_names[node_layer_idx]
                            
                            # Use the layer color from our mapping
                            if self.layer_colors_rgba and layer_name in self.layer_colors_rgba:
                                label_color = self.layer_colors_rgba[layer_name].copy()
                                # Make labels with 0 interlayer connections more transparent
                                if edge_count == 0:
                                    label_color[3] = 0.6  # 60% opacity for nodes with no interlayer connections
                            else:
                                label_color = np.array([1.0, 1.0, 0.0, 1.0])
                                # Make labels with 0 interlayer connections more transparent
                                if edge_count == 0:
                                    label_color[3] = 0.6
                            
                            label_colors.append(label_color)
                        
                        # For bar charts, store the data regardless of label visibility
                        if show_stats_bars:
                            bar_pos = self.node_positions[idx].copy()
                            # Position bars below the node
                            bar_pos[1] -= 0.05
                            bar_positions.append(bar_pos)
                            
                            # Scale factors for bar widths
                            max_bar_width = 0.1
                            node_count_width = min(max_bar_width, 0.005 * active_node_count)
                            edge_count_width = min(max_bar_width, 0.005 * (edge_count//2))
                            
                            node_count_widths.append(node_count_width)
                            edge_count_widths.append(edge_count_width)
                            
                            node_color = np.array([1.0, 0.0, 1.0, 1]) 
                            edge_color = np.array([0.0, 1.0, 0.0, 1]) 
                            
                            node_count_colors.append(node_color)
                            edge_count_colors.append(edge_color)
                            bar_visible_indices.append(idx)  # Use bar_visible_indices instead
                    else:
                        edge_count = 0
                        
                        # Count active nodes with this base ID across all layers
                        active_node_count = 0
                        for current_layer_idx in self.visible_layers:
                            node_idx_in_layer = current_layer_idx * self.nodes_per_layer + (idx % self.nodes_per_layer)
                            if (node_idx_in_layer < len(self.node_positions) and 
                                node_mask[node_idx_in_layer] and 
                                self.active_nodes[node_idx_in_layer]):
                                active_node_count += 1
                        
                        # For labels, add only active node count if there are no interlayer edges
                        if show_labels:
                            if active_node_count > 0:
                                label_text_with_counts = f"{label_text} [{active_node_count}/0]"
                            else:
                                label_text_with_counts = label_text
                                
                            # Add a small offset to position labels better
                            pos = self.node_positions[idx].copy()
                            # Offset in y direction
                            pos[1] += 0.01
                            label_positions.append(pos)
                            label_texts.append(label_text_with_counts)
                            
                            # Get the layer name for this node
                            layer_name = self.layer_names[node_layer_idx]
                            
                            # Use the layer color from our mapping
                            if self.layer_colors_rgba and layer_name in self.layer_colors_rgba:
                                label_color = self.layer_colors_rgba[layer_name].copy()
                                # Make labels with 0 interlayer connections more transparent
                                if edge_count == 0:
                                    label_color[3] = 0.6  # 60% opacity for nodes with no interlayer connections
                            else:
                                label_color = np.array([1.0, 1.0, 0.0, 1.0])
                                # Make labels with 0 interlayer connections more transparent
                                if edge_count == 0:
                                    label_color[3] = 0.6
                            
                            label_colors.append(label_color)
                        
                        # For bar charts, store the data if there are active nodes, regardless of label visibility
                        if show_stats_bars and active_node_count > 0:
                            bar_pos = self.node_positions[idx].copy()
                            # Position bars below the node
                            bar_pos[1] -= 0.05
                            bar_positions.append(bar_pos)
                            
                            # Scale factor for bar width
                            max_bar_width = 2
                            node_count_width = min(max_bar_width, 0.02 * active_node_count)
                            
                            node_count_widths.append(node_count_width)
                            edge_count_widths.append(0)  # No interlayer edges
                            
                            # Use light blue for node count bars and light green for edge count bars
                            node_color = np.array([0.4, 0.7, 1.0, 0.8])  # Light blue with 80% opacity
                            edge_color = np.array([0.4, 0.9, 0.4, 0.3])  # Light green with 30% opacity for zero edges
                            
                            node_count_colors.append(node_color)
                            edge_count_colors.append(edge_color)
                            bar_visible_indices.append(idx)  # Use bar_visible_indices instead
                
                # Update the labels if show_labels is true
                if show_labels and label_positions:
                    logger.info(f"Setting {len(label_positions)} labels with colors")
                    self.node_labels.pos = np.array(label_positions)
                    self.node_labels.text = label_texts
                    self.node_labels.color = np.array(label_colors)
                    self.node_labels.visible = True
                    # Bring labels to front
                    self.node_labels.order = 1  # Higher order means drawn later (on top)
                else:
                    # Hide the labels
                    self.node_labels.visible = False
                
                # Update the bar charts if show_stats_bars is true
                if show_stats_bars and bar_positions:
                    # For Line visuals, we need start and end points for each bar
                    node_count_line_points = []
                    edge_count_line_points = []
                    node_count_line_colors = []
                    edge_count_line_colors = []
                    
                    for i, pos in enumerate(bar_positions):
                        # Node count bar (top bar)
                        if node_count_widths[i] > 0:
                            # Start point of the bar - exactly at node position
                            start_point = self.node_positions[bar_visible_indices[i]].copy()  # Use bar_visible_indices
                            # End point of the bar (extending to the right)
                            end_point = start_point.copy()
                            end_point[0] += node_count_widths[i]
                            
                            # Add points to the list
                            node_count_line_points.append(start_point)
                            node_count_line_points.append(end_point)
                            
                            # Add colors (same color for both points)
                            node_count_line_colors.append(node_count_colors[i])
                            node_count_line_colors.append(node_count_colors[i])
                        
                        # Edge count bar (bottom bar)
                        if edge_count_widths[i] > 0:
                            # Start point of the bar - exactly at node position but slightly lower
                            start_point = self.node_positions[bar_visible_indices[i]].copy()  # Use bar_visible_indices
                            start_point[1] -= 0.005  # Offset second barchart
                            
                            # End point of the bar (extending to the right)
                            end_point = start_point.copy()
                            end_point[0] += edge_count_widths[i]
                            
                            # Add points to the list
                            edge_count_line_points.append(start_point)
                            edge_count_line_points.append(end_point)
                            
                            # Add colors (same color for both points)
                            edge_count_line_colors.append(edge_count_colors[i])
                            edge_count_line_colors.append(edge_count_colors[i])
                    
                    # Update the line visuals
                    if node_count_line_points:
                        self.node_count_bars.set_data(
                            pos=np.array(node_count_line_points),
                            color=np.array(node_count_line_colors),
                            connect='segments',
                            width=3
                        )
                        self.node_count_bars.visible = True
                    else:
                        self.node_count_bars.visible = False
                    
                    if edge_count_line_points:
                        self.edge_count_bars.set_data(
                            pos=np.array(edge_count_line_points),
                            color=np.array(edge_count_line_colors),
                            connect='segments',
                            width=3  # Thicker width for better visibility
                        )
                        self.edge_count_bars.visible = True
                    else:
                        self.edge_count_bars.visible = False
                    
                    # Bring bars to front but behind labels
                    self.node_count_bars.order = 0.8
                    self.edge_count_bars.order = 0.9
                else:
                    # Hide the bars
                    self.node_count_bars.visible = False
                    self.edge_count_bars.visible = False
            else:
                # No visible layers
                self.node_labels.visible = False
                self.node_count_bars.visible = False
                self.edge_count_bars.visible = False
        else:
            # No node IDs or layer information
            self.node_labels.visible = False
            self.node_count_bars.visible = False
            self.edge_count_bars.visible = False

        # Count edges by type for debugging
        intralayer_count = 0
        interlayer_count = 0

        # Separate intralayer and interlayer edges
        intralayer_pos = []
        intralayer_colors = []

        # For interlayer edges, we'll create a list of tuples (start_pos, end_pos, color)
        # so we can process them together for offsetting
        interlayer_edges = []

        # Process all edges
        visible_edges = self.link_pairs[edge_mask]
        visible_edge_colors = [self.link_colors_rgba[i] for i, mask in enumerate(edge_mask) if mask]
        
        for i, (start_idx, end_idx) in enumerate(visible_edges):
            # Check if this is a horizontal (intra-layer) edge or interlayer edge
            # If nodes are in the same layer, their z-coordinates will be the same
            start_z = self.node_positions[start_idx][2]
            end_z = self.node_positions[end_idx][2]
            is_horizontal = abs(start_z - end_z) < 0.001

            if is_horizontal:
                intralayer_count += 1
                # Only add intralayer edges if they should be shown
                if show_intralayer:
                    # Get the layer for this edge
                    layer_idx = start_idx // self.nodes_per_layer
                    layer_name = self.layer_names[layer_idx]
                    
                    # Use the layer color from our mapping with lower opacity
                    if self.layer_colors_rgba and layer_name in self.layer_colors_rgba:
                        edge_color = self.layer_colors_rgba[layer_name].copy()
                        edge_color[3] = 0.3  # Set opacity to 30% for horizontal edges
                    else:
                        # Use the original color with lower opacity if no mapping is found
                        edge_color = visible_edge_colors[i].copy()
                        edge_color[3] = 0.3

                    intralayer_pos.append(self.node_positions[start_idx])
                    intralayer_pos.append(self.node_positions[end_idx])
                    intralayer_colors.append(edge_color)
                    intralayer_colors.append(edge_color)
            else:
                interlayer_count += 1
                # Inter-layer edge: use the color of the starting layer with higher opacity
                start_layer_idx = start_idx // self.nodes_per_layer
                start_layer_name = self.layer_names[start_layer_idx]
                
                # Use the layer color from our mapping with higher opacity
                if self.layer_colors_rgba and start_layer_name in self.layer_colors_rgba:
                    edge_color = self.layer_colors_rgba[start_layer_name].copy()
                    edge_color[3] = 0.8  # Higher opacity for interlayer edges
                else:
                    # Use the original color with higher opacity if no mapping is found
                    edge_color = visible_edge_colors[i].copy()
                    edge_color[3] = 0.8

                # Store the edge for later processing
                interlayer_edges.append((
                    self.node_positions[start_idx].copy(),
                    self.node_positions[end_idx].copy(),
                    edge_color
                ))

        logger.info(f"Found {intralayer_count} intralayer edges and {interlayer_count} interlayer edges")

        # Now process interlayer edges to add offsets
        interlayer_pos = []
        interlayer_colors = []

        # Group edges by their base node (x,y coordinates)
        edge_groups = {}
        for start_pos, end_pos, color in interlayer_edges:
            # Create a key based on x,y coordinates (ignoring z)
            # We round to 3 decimal places to handle floating point precision issues
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

        # Update intralayer lines
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

        # Update interlayer lines
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