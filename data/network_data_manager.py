import numpy as np
import logging
import networkx as nx
from utils.color_utils import hex_to_rgba


class NetworkDataManager:
    """
    Central manager for network data and calculations.
    Stores and caches frequently used calculations to improve performance.
    """

    def __init__(self, data_dir=None):
        logger = logging.getLogger(__name__)
        logger.info("Initializing NetworkDataManager")

        # Store data directory
        self._data_dir = data_dir

        # Core data
        self.node_positions = None
        self.link_pairs = None
        self.link_colors = None
        self.link_colors_rgba = None
        self.node_colors = None
        self.node_colors_rgba = None
        self.node_ids = None
        self.layers = None
        self.layer_names = None
        self.node_clusters = None
        self.unique_clusters = None
        self.node_origins = None
        self.unique_origins = None
        self.layer_colors = None
        self.layer_colors_rgba = None

        # Derived data
        self.nodes_per_layer = None
        self.active_nodes = None
        self.node_sizes = None
        self.base_node_mapping = None  # Maps base node IDs to their indices in each layer

        # Visibility state
        self.visible_layers = None
        self.visible_clusters = None
        self.visible_origins = None
        self.current_node_mask = None
        self.current_edge_mask = None

        # Cached calculations
        self._reset_cache()

    def _reset_cache(self):
        """Reset all cached calculations"""
        self.interlayer_edge_counts = {}
        self.layer_connections = None
        self.networkx_graph = None
        self._last_nx_visible_only = None

    def set_data_dir(self, data_dir):
        """Set the data directory"""
        self._data_dir = data_dir

    def load_data(
        self,
        node_positions,
        link_pairs,
        link_colors,
        node_ids,
        layers,
        node_clusters,
        unique_clusters,
        node_colors=None,
        node_origins=None,
        unique_origins=None,
        layer_colors=None,
    ):
        """Load network data and initialize derived values"""
        logger = logging.getLogger(__name__)
        logger.info("Loading data into NetworkDataManager")

        # Store core data
        self.node_positions = node_positions
        self.link_pairs = link_pairs
        self.link_colors = link_colors
        self.node_ids = node_ids
        self.layers = layers
        self.layer_names = {i: layer for i, layer in enumerate(layers)}
        self.node_clusters = node_clusters
        self.unique_clusters = unique_clusters
        self.node_origins = node_origins or {}
        self.unique_origins = unique_origins or []
        self.layer_colors = layer_colors or {}

        # Extract cluster colors from metadata
        # Each cluster should have a consistent color in the metadata
        self.cluster_colors = {}
        for cluster in unique_clusters:
            # Find a node with this cluster and get its color
            for i, node_id in enumerate(node_ids):
                if node_clusters.get(node_id) == cluster and i < len(node_colors):
                    self.cluster_colors[cluster] = node_colors[i]
                    break

        # Calculate nodes per layer
        self.nodes_per_layer = len(node_positions) // len(layers)

        # Convert link colors to RGBA
        self.link_colors_rgba = self._enhance_colors(link_colors)

        # Initialize node colors with white if not provided
        if node_colors is None:
            node_colors = ["#FFFFFF"] * len(node_positions)
            if layer_colors:
                for i in range(len(node_positions)):
                    layer_idx = i // self.nodes_per_layer
                    layer_name = layers[layer_idx]
                    node_colors[i] = layer_colors.get(layer_name, "#CCCCCC")

        self.node_colors = node_colors

        # Convert node colors to RGBA
        self.node_colors_rgba = np.ones((len(node_positions), 4))
        for i, color_hex in enumerate(node_colors):
            self.node_colors_rgba[i] = hex_to_rgba(color_hex)

        # Convert layer colors to RGBA
        self.layer_colors_rgba = {}
        for layer_name, color_hex in self.layer_colors.items():
            self.layer_colors_rgba[layer_name] = hex_to_rgba(color_hex, alpha=1.0)

        # Determine which nodes are active (have connections)
        self.active_nodes = np.zeros(len(node_positions), dtype=bool)
        for start_idx, end_idx in self.link_pairs:
            self.active_nodes[start_idx] = True
            self.active_nodes[end_idx] = True

        # Set node sizes based on activity
        self.node_sizes = np.ones(len(node_positions)) * 3  # Default size
        self.node_sizes[self.active_nodes] = 9  # Larger for active nodes

        # Create base node mapping
        self._create_base_node_mapping()

        # Reset cached calculations
        self._reset_cache()

        logger.info("Data loading complete")

    def _create_base_node_mapping(self):
        """Create a mapping from base node IDs to their indices in each layer"""
        self.base_node_mapping = {}

        for i, node_id in enumerate(self.node_ids):
            base_node = node_id.split("_")[0] if "_" in node_id else node_id
            layer_idx = i // self.nodes_per_layer

            if base_node not in self.base_node_mapping:
                self.base_node_mapping[base_node] = {}

            self.base_node_mapping[base_node][layer_idx] = i

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

    def update_visibility(self, visible_layers, visible_clusters, visible_origins):
        """Update visibility masks based on filter settings"""
        logger = logging.getLogger(__name__)
        logger.info("Updating visibility masks")

        # Check if visibility actually changed
        visibility_changed = (
            self.visible_layers != visible_layers or
            self.visible_clusters != visible_clusters or
            self.visible_origins != visible_origins
        )

        if not visibility_changed:
            return self.current_node_mask, self.current_edge_mask

        # Update visibility state
        self.visible_layers = visible_layers
        self.visible_clusters = visible_clusters
        self.visible_origins = visible_origins

        # Create node mask based on layers, clusters, and origins
        node_mask = np.zeros(len(self.node_positions), dtype=bool)

        for i, node_id in enumerate(self.node_ids):
            # Check layer visibility
            layer_idx = i // self.nodes_per_layer
            if layer_idx not in visible_layers:
                continue

            # Check cluster visibility
            cluster = self.node_clusters[node_id]
            if cluster not in visible_clusters:
                continue

            # Check origin visibility
            origin = self.node_origins.get(node_id, "Unknown")
            if origin not in visible_origins:
                continue

            # Node passes all filters
            node_mask[i] = True

        # Create edge mask based on node visibility
        edge_mask = np.zeros(len(self.link_pairs), dtype=bool)
        for i, (start_idx, end_idx) in enumerate(self.link_pairs):
            if node_mask[start_idx] and node_mask[end_idx]:
                edge_mask[i] = True

        # Store current masks
        self.current_node_mask = node_mask
        self.current_edge_mask = edge_mask

        # Reset cached calculations since visibility changed
        self._reset_cache()

        return node_mask, edge_mask

    def get_interlayer_edge_counts(self):
        """Calculate and cache interlayer edge counts for each node"""
        if self.interlayer_edge_counts:
            return self.interlayer_edge_counts

        edge_counts = {}

        for i, (start_idx, end_idx) in enumerate(self.link_pairs):
            # Skip edges that aren't visible
            if not self.current_edge_mask[i]:
                continue

            # Skip intralayer edges
            start_layer = start_idx // self.nodes_per_layer
            end_layer = end_idx // self.nodes_per_layer
            if start_layer == end_layer:
                continue

            # Count this edge for both source and destination nodes
            src_id = self.node_ids[start_idx].split("_")[0]
            dst_id = self.node_ids[end_idx].split("_")[0]

            edge_counts[src_id] = edge_counts.get(src_id, 0) + 1
            edge_counts[dst_id] = edge_counts.get(dst_id, 0) + 1

        self.interlayer_edge_counts = edge_counts
        return edge_counts

    def get_layer_connections(self, filter_to_visible=False):
        """
        Calculate and cache layer connection matrix

        Parameters:
        -----------
        filter_to_visible : bool
            If True, return a matrix containing only visible layers, clusters, and origins

        Returns:
        --------
        numpy.ndarray
            Matrix of connection counts between layers
        """
        # Early exit if no visible filters
        if filter_to_visible:
            if not self.visible_layers or not self.visible_clusters or not self.visible_origins:
                return np.array([])

        # Calculate the full connection matrix if not cached
        if self.layer_connections is None:
            # Initialize connection matrix
            num_layers = len(self.layers)
            self.layer_connections = np.zeros((num_layers, num_layers), dtype=int)

            # Count connections between layers
            for i, (start_idx, end_idx) in enumerate(self.link_pairs):
                if not self.current_edge_mask[i]:
                    continue

                start_layer = start_idx // self.nodes_per_layer
                end_layer = end_idx // self.nodes_per_layer

                # Get node IDs for cluster and origin checks
                start_node_id = self.node_ids[start_idx]
                end_node_id = self.node_ids[end_idx]

                # Check clusters
                start_cluster = self.node_clusters.get(start_node_id)
                end_cluster = self.node_clusters.get(end_node_id)

                # Check origins
                start_origin = self.node_origins.get(start_node_id, "Unknown")
                end_origin = self.node_origins.get(end_node_id, "Unknown")

                # Apply all filters if requested
                if filter_to_visible:
                    # Check layer visibility
                    if start_layer not in self.visible_layers or end_layer not in self.visible_layers:
                        continue

                    # Check cluster visibility
                    if start_cluster not in self.visible_clusters or end_cluster not in self.visible_clusters:
                        continue

                    # Check origin visibility
                    if start_origin not in self.visible_origins or end_origin not in self.visible_origins:
                        continue

                # Add connection to matrix
                self.layer_connections[start_layer, end_layer] += 1
                if start_layer != end_layer:
                    self.layer_connections[end_layer, start_layer] += 1  # Count both directions

        # If we don't need to filter, return the full matrix
        if not filter_to_visible:
            return self.layer_connections.copy()


        # Create a filtered matrix with only visible layers
        num_visible = len(self.visible_layers)
        filtered_connections = np.zeros((num_visible, num_visible), dtype=int)

        # Copy values from the full matrix to the filtered matrix
        for i, orig_i in enumerate(self.visible_layers):
            for j, orig_j in enumerate(self.visible_layers):
                filtered_connections[i, j] = self.layer_connections[orig_i, orig_j]

        return filtered_connections

    def get_networkx_graph(self, visible_only=True):
        """Create and cache a NetworkX graph representation"""
        if (
            self.networkx_graph is not None
            and visible_only == self._last_nx_visible_only
        ):
            return self.networkx_graph

        G = nx.Graph()

        # Add nodes
        for i, node_id in enumerate(self.node_ids):
            if not visible_only or self.current_node_mask[i]:
                layer_idx = i // self.nodes_per_layer
                G.add_node(
                    i,
                    id=node_id,
                    layer=layer_idx,
                    cluster=self.node_clusters.get(node_id, "Unknown"),
                    origin=self.node_origins.get(node_id, "Unknown"),
                )

        # Add edges
        for i, (start_idx, end_idx) in enumerate(self.link_pairs):
            if not visible_only or self.current_edge_mask[i]:
                G.add_edge(start_idx, end_idx)

        self.networkx_graph = G
        self._last_nx_visible_only = visible_only
        return G

    def count_active_nodes_for_base(self, base_node, node_idx):
        """Count active nodes with this base ID across all visible layers"""
        active_node_count = 0

        for layer_idx in self.visible_layers:
            node_idx_in_layer = layer_idx * self.nodes_per_layer + (
                node_idx % self.nodes_per_layer
            )
            if (
                node_idx_in_layer < len(self.node_positions)
                and self.current_node_mask[node_idx_in_layer]
                and self.active_nodes[node_idx_in_layer]
            ):
                active_node_count += 1

        return active_node_count
