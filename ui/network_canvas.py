import numpy as np
from vispy import scene
from vispy.scene.visuals import Markers, Line
import logging
from utils.color_utils import hex_to_rgba

class NetworkCanvas:
    def __init__(self, parent=None):
        logger = logging.getLogger(__name__)
        logger.info("Creating canvas...")
        
        # Create canvas
        self.canvas = scene.SceneCanvas(keys='interactive', size=(800, 600))
        
        # Create view
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.fov = 45
        self.view.camera.distance = 8
        
        # Initialize visualization elements
        self.scatter = Markers()
        self.view.add(self.scatter)
        
        # Create separate line visuals for intralayer and interlayer edges
        self.intralayer_lines = Line(pos=np.zeros((0, 3)), color=np.zeros((0, 4)),
                                   connect='segments', width=1)
        self.view.add(self.intralayer_lines)
        
        self.interlayer_lines = Line(pos=np.zeros((0, 3)), color=np.zeros((0, 4)),
                                   connect='segments', width=4)  # Thicker width
        self.view.add(self.interlayer_lines)
        
        # Initialize data
        self.node_positions = None
        self.link_pairs = None
        self.link_colors_rgba = None
        self.node_colors_rgba = None
    
    def load_data(self, node_positions, link_pairs, link_colors, node_colors=None):
        """Load network data into the visualization"""
        self.node_positions = node_positions
        self.link_pairs = link_pairs
        
        # Convert link colors from hex to RGBA with enhanced saturation
        self.link_colors_rgba = []
        for color in link_colors:
            rgba = hex_to_rgba(color, alpha=0.9)  # Increased opacity
            # Enhance color saturation by reducing any gray component
            # This makes colors more vibrant
            max_val = max(rgba[0], rgba[1], rgba[2])
            min_val = min(rgba[0], rgba[1], rgba[2])
            
            # If there's a significant gray component (all channels similar)
            if max_val - min_val < 0.3:  # Small difference between channels
                # Find dominant channel and enhance it
                dominant_idx = np.argmax(rgba[:3])
                for i in range(3):
                    if i == dominant_idx:
                        rgba[i] = min(1.0, rgba[i] * 1.3)  # Boost dominant channel
                    else:
                        rgba[i] = max(0.0, rgba[i] * 0.7)  # Reduce other channels
            
            self.link_colors_rgba.append(rgba)
        
        # Convert node colors to RGBA
        self.node_colors_rgba = np.ones((len(node_positions), 4))  # Default white
        
        if node_colors:
            for i, color_hex in enumerate(node_colors):
                self.node_colors_rgba[i] = hex_to_rgba(color_hex)
    
    def update_visibility(self, node_mask, edge_mask, show_intralayer=True):
        """Update the visibility of nodes and edges based on masks"""
        logger = logging.getLogger(__name__)
        logger.info(f"Updating visibility with show_intralayer={show_intralayer}")
        
        # Handle case when no nodes are visible
        if not np.any(node_mask):
            self.scatter.set_data(np.zeros((1, 3)), edge_color='black',
                                face_color=np.array([[0, 0, 0, 0]]), size=0)
            self.intralayer_lines.set_data(pos=np.zeros((0, 3)),
                                        color=np.zeros((0, 4)))
            self.interlayer_lines.set_data(pos=np.zeros((0, 3)),
                                        color=np.zeros((0, 4)))
            return
        
        # Update scatter plot
        visible_nodes = self.node_positions[node_mask]
        visible_colors = self.node_colors_rgba[node_mask]
        self.scatter.set_data(visible_nodes, edge_color='black', 
                            face_color=visible_colors, size=3)
        
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
                    offset_scale = 0.002  # Increased scale for more visible offset

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