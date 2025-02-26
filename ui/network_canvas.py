import numpy as np
from vispy import scene
from vispy.scene.visuals import Markers, Line, Text
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
        self.node_labels = Text(pos=np.array([[0, 0, 0]]), text=[''], color='white', font_size=8)
        self.node_labels.visible = False  # Start with labels invisible
        self.view.add(self.node_labels)

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

    def update_visibility(self, node_mask, edge_mask, show_intralayer=True, show_nodes=True, show_labels=True):
        """Update the visibility of nodes and edges based on masks"""
        logger = logging.getLogger(__name__)
        logger.info(f"Updating visibility with show_intralayer={show_intralayer}, show_nodes={show_nodes}, show_labels={show_labels}")

        # Store the node mask for later use
        self.node_mask = node_mask

        # Handle case when no nodes are visible
        if not np.any(node_mask):
            self.scatter.set_data(np.zeros((1, 3)), edge_color='black',
                                face_color=np.array([[0, 0, 0, 0]]), size=0)
            self.intralayer_lines.set_data(pos=np.zeros((0, 3)),
                                        color=np.zeros((0, 4)))
            self.interlayer_lines.set_data(pos=np.zeros((0, 3)),
                                        color=np.zeros((0, 4)))
            # Don't modify the position, just hide the labels
            self.node_labels.visible = False
            return

        # Update scatter plot
        visible_nodes = self.node_positions[node_mask]
        visible_colors = self.node_colors_rgba[node_mask]
        visible_sizes = self.node_sizes[node_mask]  # Use the node sizes array
        
        if show_nodes:
            self.scatter.set_data(visible_nodes, edge_color='black', 
                                face_color=visible_colors, size=visible_sizes)
        else:
            # Make nodes invisible but keep their positions
            self.scatter.set_data(visible_nodes, edge_color='black',
                                face_color=np.zeros_like(visible_colors), size=0)

        # Update node labels
        if show_labels and self.node_ids is not None and self.visible_layers is not None:
            # Find the top and bottom visible layers
            if len(self.visible_layers) > 0:
                top_layer = min(self.visible_layers)
                bottom_layer = max(self.visible_layers)
                
                # Get nodes in top and bottom layers
                label_positions = []
                label_texts = []
                
                # Process visible nodes
                visible_indices = np.where(node_mask)[0]
                for idx in visible_indices:
                    layer_idx = idx // self.nodes_per_layer
                    if layer_idx == top_layer or layer_idx == bottom_layer:
                        # Only show labels for active nodes
                        if self.active_nodes[idx]:
                            node_id = self.node_ids[idx]
                            # Add a small offset to position labels better
                            pos = self.node_positions[idx].copy()
                            # Offset in y direction
                            pos[1] += 0.05
                            label_positions.append(pos)
                            label_texts.append(str(node_id))
                
                # Update the labels
                if label_positions:
                    self.node_labels.pos = np.array(label_positions)
                    self.node_labels.text = label_texts
                    self.node_labels.color = 'white'  # Make labels visible
                    self.node_labels.visible = True
                    # Bring labels to front
                    self.node_labels.order = 1  # Higher order means drawn later (on top)
                else:
                    # Use a dummy position but hide the labels
                    self.node_labels.visible = False
            else:
                # Use a dummy position but hide the labels
                self.node_labels.visible = False
        else:
            # Hide the labels without changing position
            self.node_labels.visible = False

        # Get visible edges
        visible_edges = self.link_pairs[edge_mask]
        visible_colors = [self.link_colors_rgba[i] for i, mask in enumerate(edge_mask) if mask]

        # Count edges by type for debugging
        intralayer_count = 0
        interlayer_count = 0

        # Separate intralayer and interlayer edges
        intralayer_pos = []
        intralayer_colors = []

        # For interlayer edges, we'll create a list of tuples (start_pos, end_pos, color)
        # so we can process them together for offsetting
        interlayer_edges = []

        # First, collect all edges
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
                    # Intra-layer edge: use lower opacity
                    edge_color = visible_colors[i].copy()
                    edge_color[3] = 0.3  # Set opacity to 30% for horizontal edges

                    intralayer_pos.append(self.node_positions[start_idx])
                    intralayer_pos.append(self.node_positions[end_idx])
                    intralayer_colors.append(edge_color)
                    intralayer_colors.append(edge_color)
            else:
                interlayer_count += 1
                # Inter-layer edge: use the color of the starting layer with higher opacity
                edge_color = visible_colors[i].copy()
                edge_color[3] = 0.8  # Higher opacity for interlayer edges

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