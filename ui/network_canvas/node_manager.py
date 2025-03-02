import numpy as np
import logging
from utils.color_utils import hex_to_rgba


class NodeManager:
    def __init__(self, canvas):
        self.canvas = canvas

    def load_data(
        self,
        node_positions=None,
        link_pairs=None,
        link_colors=None,
        node_colors=None,
        node_ids=None,
    ):
        """Load data either directly or from the data manager"""
        if self.canvas.data_manager is not None:
            # Use data from the manager
            self.canvas.node_positions = self.canvas.data_manager.node_positions
            self.canvas.link_pairs = self.canvas.data_manager.link_pairs
            self.canvas.link_colors_rgba = self.canvas.data_manager.link_colors_rgba
            self.canvas.node_colors_rgba = self.canvas.data_manager.node_colors_rgba
            self.canvas.node_sizes = self.canvas.data_manager.node_sizes
            self.canvas.active_nodes = self.canvas.data_manager.active_nodes
            self.canvas.node_ids = self.canvas.data_manager.node_ids
            self.canvas.layer_names = self.canvas.data_manager.layer_names
            self.canvas.nodes_per_layer = self.canvas.data_manager.nodes_per_layer
        else:
            # Use directly provided data
            self.canvas.node_positions = node_positions
            self.canvas.link_pairs = link_pairs
            self.canvas.node_ids = node_ids

            # Convert link colors from hex to RGBA with enhanced saturation
            self.canvas.link_colors_rgba = self._enhance_colors(link_colors)

            # Initialize node colors with white
            self.canvas.node_colors_rgba = np.ones((len(node_positions), 4))

            if node_colors:
                for i, color_hex in enumerate(node_colors):
                    self.canvas.node_colors_rgba[i] = hex_to_rgba(color_hex)

            # Initialize node sizes array - default size for all nodes
            self.canvas.node_sizes = np.ones(len(node_positions)) * 3

            # Determine which nodes actually exist in each layer based on edges
            self.canvas.active_nodes = np.zeros(len(node_positions), dtype=bool)
            for start_idx, end_idx in self.canvas.link_pairs:
                self.canvas.active_nodes[start_idx] = True
                self.canvas.active_nodes[end_idx] = True

            # Set larger size for active nodes
            self.canvas.node_sizes[self.canvas.active_nodes] = 9  # 3x larger

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
        logger = logging.getLogger(__name__)
        logger.debug(f"Setting layer colors: {layer_colors}")

        self.canvas.layer_colors = layer_colors

        self.canvas.layer_colors_rgba = {}
        for layer_name, color_hex in layer_colors.items():
            rgba = hex_to_rgba(color_hex, alpha=1.0)
            self.canvas.layer_colors_rgba[layer_name] = rgba
            logger.debug(f"Layer {layer_name}: {color_hex} -> {rgba}")

    def update_with_optimized_data(self, node_positions, node_colors, node_sizes, visible=True):
        """
        Update nodes with optimized data from NetworkDataManager.
        
        Parameters:
        -----------
        node_positions : numpy.ndarray
            Array of node positions (x, y, z)
        node_colors : numpy.ndarray
            Array of node colors (RGBA)
        node_sizes : numpy.ndarray
            Array of node sizes
        visible : bool
            Whether to show the nodes
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Validate input data
            if node_positions is None or len(node_positions) == 0:
                logger.warning("No node positions provided")
                self.canvas.scatter.visible = False
                return
                
            if node_colors is None or len(node_colors) != len(node_positions):
                logger.warning(f"Invalid node colors: expected {len(node_positions)}, got {len(node_colors) if node_colors is not None else 0}")
                # Use default colors if none provided
                node_colors = np.ones((len(node_positions), 4))
                
            if node_sizes is None or len(node_sizes) != len(node_positions):
                logger.warning(f"Invalid node sizes: expected {len(node_positions)}, got {len(node_sizes) if node_sizes is not None else 0}")
                # Use default size if none provided
                node_sizes = np.ones(len(node_positions)) * 3
            
            # Update the node visual with the optimized data
            self.canvas.scatter.set_data(
                pos=node_positions,
                size=node_sizes,
                face_color=node_colors,
                edge_width=0,
                edge_color=None
            )
            
            # Set visibility
            self.canvas.scatter.visible = visible
            
            # Store the data for later use
            self.canvas.node_positions = node_positions
            self.canvas.node_colors_rgba = node_colors
            self.canvas.node_sizes = node_sizes
            
            logger.debug(f"Updated nodes with optimized data: {len(node_positions)} nodes")
        except Exception as e:
            logger.error(f"Error updating nodes with optimized data: {e}")
            import traceback
            traceback.print_exc()
