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
        
        # Performance settings
        self.chunk_size = 100000  # Default chunk size for processing large data
        self.max_nodes_for_visualization = 100000  # Default threshold

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
        
        # Store node colors
        self.node_colors = node_colors
        
        # Initialize node_colors_rgba
        if node_colors is not None:
            try:
                # Convert hex colors to RGBA
                self.node_colors_rgba = np.array([hex_to_rgba(color) for color in node_colors])
            except Exception as e:
                logger.error(f"Error converting node colors to RGBA: {e}")
                # Default to white
                self.node_colors_rgba = np.ones((len(node_positions), 4))
        else:
            # Default to white if no colors provided
            self.node_colors_rgba = np.ones((len(node_positions), 4))

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

        # Always use the optimized approach
        return self._update_visibility_optimized()
        
    def _update_visibility_optimized(self):
        """
        Optimized visibility update for networks of any size.
        Uses vectorized operations where possible and processes data in chunks to improve performance.
        """
        logger = logging.getLogger(__name__)
        
        # Create layer mask using vectorized operations
        layer_mask = np.zeros(len(self.node_positions), dtype=bool)
        for layer_idx in self.visible_layers:
            start_idx = layer_idx * self.nodes_per_layer
            end_idx = start_idx + self.nodes_per_layer
            layer_mask[start_idx:end_idx] = True
        
        # Initialize node mask with layer mask
        node_mask = layer_mask.copy()
        
        # Process clusters in batches to reduce memory pressure
        if self.visible_clusters and len(self.visible_clusters) < len(self.unique_clusters):
            # Create a set for faster lookups
            visible_clusters_set = set(self.visible_clusters)
            
            # Process in chunks
            for chunk_start in range(0, len(self.node_positions), self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, len(self.node_positions))
                
                # Process this chunk
                for i in range(chunk_start, chunk_end):
                    if node_mask[i]:  # Only check nodes that passed the layer filter
                        node_id = self.node_ids[i]
                        cluster = self.node_clusters.get(node_id)
                        if cluster not in visible_clusters_set:
                            node_mask[i] = False
        
        # Process origins in batches
        if self.visible_origins and len(self.visible_origins) < len(self.unique_origins):
            # Create a set for faster lookups
            visible_origins_set = set(self.visible_origins)
            
            # Process in chunks
            for chunk_start in range(0, len(self.node_positions), self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, len(self.node_positions))
                
                # Process this chunk
                for i in range(chunk_start, chunk_end):
                    if node_mask[i]:  # Only check nodes that passed previous filters
                        node_id = self.node_ids[i]
                        origin = self.node_origins.get(node_id, "Unknown")
                        if origin not in visible_origins_set:
                            node_mask[i] = False
        
        # Create edge mask based on node visibility
        # Process edges in chunks to reduce memory pressure
        edge_mask = np.zeros(len(self.link_pairs), dtype=bool)
        
        for chunk_start in range(0, len(self.link_pairs), self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, len(self.link_pairs))
            
            # Process this chunk of edges
            for i in range(chunk_start, chunk_end):
                start_idx, end_idx = self.link_pairs[i]
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

        logger = logging.getLogger(__name__)
        logger.info("Creating NetworkX graph from filtered data")
        
        # Always use the optimized approach
        # If we want only visible nodes, use the current mask
        if visible_only and self.current_node_mask is not None:
            # Get indices of visible nodes
            visible_indices = np.where(self.current_node_mask)[0]
            
            # Create a mapping from original indices to new indices in the graph
            idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(visible_indices)}
            
            # Create graph with only visible nodes
            G = nx.Graph()
            
            # Add nodes with their attributes
            for i, orig_idx in enumerate(visible_indices):
                node_id = self.node_ids[orig_idx]
                layer_idx = orig_idx // self.nodes_per_layer
                G.add_node(
                    i,  # Use new index in the graph
                    id=node_id,
                    original_idx=orig_idx,  # Store original index for reference
                    layer=layer_idx,
                    cluster=self.node_clusters.get(node_id, "Unknown"),
                    origin=self.node_origins.get(node_id, "Unknown"),
                )
            
            # Add edges between visible nodes
            if self.current_edge_mask is not None:
                # Process edges in chunks
                for chunk_start in range(0, len(self.link_pairs), self.chunk_size):
                    chunk_end = min(chunk_start + self.chunk_size, len(self.link_pairs))
                    
                    for i in range(chunk_start, chunk_end):
                        if self.current_edge_mask[i]:
                            start_idx, end_idx = self.link_pairs[i]
                            # Only add if both endpoints are in our visible set
                            if start_idx in idx_mapping and end_idx in idx_mapping:
                                G.add_edge(
                                    idx_mapping[start_idx],
                                    idx_mapping[end_idx]
                                )
            
            self.networkx_graph = G
            self._last_nx_visible_only = visible_only
            return G
        
        # If we have too many nodes even after filtering, return a simplified graph
        elif len(self.node_positions) > 1000000:  # 1 million nodes
            logger.warning("Network too large for full NetworkX conversion, returning simplified graph")
            return self._create_simplified_graph(max_nodes=100000)
        
        # Standard implementation for all other cases
        G = nx.Graph()

        # Add nodes
        for i, node_id in enumerate(self.node_ids):
            if not visible_only or (self.current_node_mask is not None and self.current_node_mask[i]):
                layer_idx = i // self.nodes_per_layer
                G.add_node(
                    i,
                    id=node_id,
                    layer=layer_idx,
                    cluster=self.node_clusters.get(node_id, "Unknown"),
                    origin=self.node_origins.get(node_id, "Unknown"),
                )

        # Add edges in chunks
        for chunk_start in range(0, len(self.link_pairs), self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, len(self.link_pairs))
            
            for i in range(chunk_start, chunk_end):
                start_idx, end_idx = self.link_pairs[i]
                if not visible_only or (self.current_edge_mask is not None and self.current_edge_mask[i]):
                    G.add_edge(start_idx, end_idx)

        self.networkx_graph = G
        self._last_nx_visible_only = visible_only
        return G
        
    def _create_simplified_graph(self, max_nodes=100000):
        """Create a simplified graph with a subset of nodes for very large networks"""
        logger = logging.getLogger(__name__)
        logger.info(f"Creating simplified graph with max {max_nodes} nodes")
        
        G = nx.Graph()
        
        # If we have a current mask, use it to prioritize visible nodes
        if self.current_node_mask is not None and np.any(self.current_node_mask):
            # Get indices of visible nodes
            visible_indices = np.where(self.current_node_mask)[0]
            
            # If we have too many visible nodes, sample them
            if len(visible_indices) > max_nodes:
                # Sample nodes, prioritizing those with more connections
                # This ensures we keep the most important nodes in the network
                
                # Count connections for each node
                node_connections = np.zeros(len(self.node_positions), dtype=int)
                for i, (start_idx, end_idx) in enumerate(self.link_pairs):
                    if self.current_edge_mask is None or self.current_edge_mask[i]:
                        node_connections[start_idx] += 1
                        node_connections[end_idx] += 1
                
                # Get connection counts for visible nodes
                visible_connections = node_connections[visible_indices]
                
                # Sort visible indices by connection count (descending)
                sorted_indices = visible_indices[np.argsort(-visible_connections)]
                
                # Take top nodes by connection count, plus some random ones for diversity
                top_count = int(max_nodes * 0.8)  # 80% most connected
                random_count = max_nodes - top_count  # 20% random
                
                top_indices = sorted_indices[:top_count]
                
                # For the random portion, exclude the already selected top indices
                remaining_indices = np.setdiff1d(visible_indices, top_indices)
                random_indices = np.random.choice(
                    remaining_indices, 
                    min(random_count, len(remaining_indices)), 
                    replace=False
                )
                
                # Combine top and random indices
                selected_indices = np.concatenate([top_indices, random_indices])
            else:
                # If we have fewer visible nodes than max_nodes, use all of them
                selected_indices = visible_indices
        else:
            # If no mask, just take a random sample
            selected_indices = np.random.choice(
                len(self.node_positions), 
                min(max_nodes, len(self.node_positions)), 
                replace=False
            )
        
        # Create mapping from original indices to new indices
        idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
        
        # Add nodes
        for new_idx, orig_idx in enumerate(selected_indices):
            node_id = self.node_ids[orig_idx]
            layer_idx = orig_idx // self.nodes_per_layer
            G.add_node(
                new_idx,
                id=node_id,
                original_idx=orig_idx,
                layer=layer_idx,
                cluster=self.node_clusters.get(node_id, "Unknown"),
                origin=self.node_origins.get(node_id, "Unknown"),
            )
        
        # Add edges between selected nodes
        for i, (start_idx, end_idx) in enumerate(self.link_pairs):
            if (start_idx in idx_mapping and end_idx in idx_mapping and
                (self.current_edge_mask is None or self.current_edge_mask[i])):
                G.add_edge(
                    idx_mapping[start_idx],
                    idx_mapping[end_idx]
                )
        
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

    def load_data_with_prefilter(
        self,
        node_positions,
        link_pairs,
        link_colors,
        node_ids,
        layers,
        node_clusters,
        unique_clusters,
        prefilter_criteria=None,
        node_colors=None,
        node_origins=None,
        unique_origins=None,
        layer_colors=None,
    ):
        """
        Load network data with pre-filtering for large networks.
        This method is optimized for handling millions of nodes by filtering before full loading.
        
        Parameters:
        -----------
        prefilter_criteria : dict
            Criteria for pre-filtering the network, can include:
            - max_nodes: int, maximum number of nodes to keep
            - layers: list, specific layers to keep
            - clusters: list, specific clusters to keep
            - origins: list, specific origins to keep
            - node_ids: list, specific node IDs to keep
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Loading network data with pre-filtering")
        
        # Check if we need to handle this as a large network
        total_nodes = len(node_positions)
        is_large_network = total_nodes > self.max_nodes_for_visualization
        if is_large_network:
            logger.info(f"Large network detected: {total_nodes} nodes")
        
        # Apply pre-filtering if criteria provided and it's a large network
        if prefilter_criteria and is_large_network:
            # Create initial mask for all nodes (all True)
            prefilter_mask = np.ones(len(node_positions), dtype=bool)
            
            # Filter by layers if specified
            if 'layers' in prefilter_criteria:
                layer_mask = np.zeros(len(node_positions), dtype=bool)
                for i, node_id in enumerate(node_ids):
                    layer_idx = i // (len(node_positions) // len(layers))
                    if layers[layer_idx] in prefilter_criteria['layers']:
                        layer_mask[i] = True
                prefilter_mask = prefilter_mask & layer_mask
            
            # Filter by clusters if specified
            if 'clusters' in prefilter_criteria:
                cluster_mask = np.zeros(len(node_positions), dtype=bool)
                for i, node_id in enumerate(node_ids):
                    if node_clusters.get(node_id) in prefilter_criteria['clusters']:
                        cluster_mask[i] = True
                prefilter_mask = prefilter_mask & cluster_mask
            
            # Filter by origins if specified
            if 'origins' in prefilter_criteria and node_origins:
                origin_mask = np.zeros(len(node_positions), dtype=bool)
                for i, node_id in enumerate(node_ids):
                    if node_origins.get(node_id, "Unknown") in prefilter_criteria['origins']:
                        origin_mask[i] = True
                prefilter_mask = prefilter_mask & origin_mask
            
            # Filter by specific node IDs if specified
            if 'node_ids' in prefilter_criteria:
                node_id_mask = np.zeros(len(node_positions), dtype=bool)
                for i, node_id in enumerate(node_ids):
                    if node_id in prefilter_criteria['node_ids']:
                        node_id_mask[i] = True
                prefilter_mask = prefilter_mask & node_id_mask
            
            # Apply maximum node limit if specified
            if 'max_nodes' in prefilter_criteria:
                max_nodes = prefilter_criteria['max_nodes']
                if np.sum(prefilter_mask) > max_nodes:
                    # Get indices of True values in the mask
                    true_indices = np.where(prefilter_mask)[0]
                    # Randomly select max_nodes indices
                    selected_indices = np.random.choice(true_indices, max_nodes, replace=False)
                    # Create a new mask with only the selected indices
                    new_mask = np.zeros(len(node_positions), dtype=bool)
                    new_mask[selected_indices] = True
                    prefilter_mask = new_mask
            
            # Filter edges based on node mask
            edge_mask = np.zeros(len(link_pairs), dtype=bool)
            for i, (start_idx, end_idx) in enumerate(link_pairs):
                if prefilter_mask[start_idx] and prefilter_mask[end_idx]:
                    edge_mask[i] = True
            
            # Apply filters to data
            filtered_node_positions = node_positions[prefilter_mask]
            
            # Handle node colors properly
            if node_colors is not None:
                if isinstance(node_colors, np.ndarray) and node_colors.shape[0] == len(node_positions):
                    # If it's a NumPy array with the right shape, filter it directly
                    filtered_node_colors = node_colors[prefilter_mask]
                elif isinstance(node_colors, list) and len(node_colors) == len(node_positions):
                    # If it's a list, filter it manually
                    filtered_node_colors = [node_colors[i] for i in range(len(node_positions)) if prefilter_mask[i]]
                else:
                    # If it's in an unexpected format, log a warning and use default colors
                    logger.warning(f"Node colors in unexpected format: {type(node_colors)}")
                    filtered_node_colors = None
            else:
                filtered_node_colors = None
                
            filtered_node_ids = [id for i, id in enumerate(node_ids) if prefilter_mask[i]]
            
            # Create mapping from old indices to new indices
            old_to_new_idx = {}
            new_idx = 0
            for old_idx, include in enumerate(prefilter_mask):
                if include:
                    old_to_new_idx[old_idx] = new_idx
                    new_idx += 1
            
            # Remap edge indices
            filtered_link_pairs = []
            filtered_link_colors = []
            for i, (start_idx, end_idx) in enumerate(link_pairs):
                if edge_mask[i]:
                    filtered_link_pairs.append([old_to_new_idx[start_idx], old_to_new_idx[end_idx]])
                    filtered_link_colors.append(link_colors[i])
            
            # Create filtered node clusters and origins
            filtered_node_clusters = {node_id: node_clusters[node_id] for node_id in filtered_node_ids if node_id in node_clusters}
            filtered_node_origins = {node_id: node_origins[node_id] for node_id in filtered_node_ids if node_id in node_origins} if node_origins else {}
            
            # Log filtering results
            logger.info(f"Pre-filtering reduced nodes from {len(node_positions)} to {len(filtered_node_positions)}")
            logger.info(f"Pre-filtering reduced edges from {len(link_pairs)} to {len(filtered_link_pairs)}")
            
            # Load the filtered data using the standard method
            self.load_data(
                filtered_node_positions,
                filtered_link_pairs,
                filtered_link_colors,
                filtered_node_ids,
                layers,
                filtered_node_clusters,
                unique_clusters,
                filtered_node_colors,
                filtered_node_origins,
                unique_origins,
                layer_colors
            )
        else:
            # If no pre-filtering needed, use standard load method
            self.load_data(
                node_positions,
                link_pairs,
                link_colors,
                node_ids,
                layers,
                node_clusters,
                unique_clusters,
                node_colors,
                node_origins,
                unique_origins,
                layer_colors
            )

    def get_filtered_data_for_vispy(self):
        """Get filtered data for Vispy visualization"""
        # Always use the optimized approach
        return self._get_filtered_data_optimized()
        
    def _get_filtered_data_optimized(self):
        """
        Optimized method to get filtered data for visualization.
        Uses pre-computed masks and returns only the necessary data.
        """
        # Use the current masks to filter data
        filtered_node_positions = self.node_positions[self.current_node_mask]
        
        # Handle node colors - might be None, a list, or a NumPy array
        if self.node_colors_rgba is None:
            # Create default colors if none exist
            filtered_node_colors = np.ones((len(filtered_node_positions), 4))
        elif isinstance(self.node_colors_rgba, np.ndarray):
            # If it's already a NumPy array, use it directly
            filtered_node_colors = self.node_colors_rgba[self.current_node_mask]
        else:
            # If it's a list or other format, filter it manually
            visible_indices = np.where(self.current_node_mask)[0]
            if isinstance(self.node_colors_rgba, list) and len(self.node_colors_rgba) == len(self.node_positions):
                filtered_node_colors = [self.node_colors_rgba[i] for i in visible_indices]
                # Convert to NumPy array for consistency
                try:
                    filtered_node_colors = np.array(filtered_node_colors)
                except:
                    # If conversion fails, use default colors
                    filtered_node_colors = np.ones((len(filtered_node_positions), 4))
            else:
                # Default to white if colors can't be filtered
                filtered_node_colors = np.ones((len(filtered_node_positions), 4))
                
        # Handle node sizes
        filtered_node_sizes = self.node_sizes[self.current_node_mask]
        
        # Get indices of visible nodes for edge mapping
        visible_indices = np.where(self.current_node_mask)[0]
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(visible_indices)}
        
        # Filter edges
        filtered_edges = []
        filtered_edge_colors = []
        
        # Process edges in chunks
        for chunk_start in range(0, len(self.link_pairs), self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, len(self.link_pairs))
            
            for i in range(chunk_start, chunk_end):
                if self.current_edge_mask[i]:
                    start_idx, end_idx = self.link_pairs[i]
                    # Map to new indices
                    new_start = index_map[start_idx]
                    new_end = index_map[end_idx]
                    filtered_edges.append((new_start, new_end))
                    filtered_edge_colors.append(self.link_colors[i])
        
        # Convert to numpy arrays
        filtered_edges = np.array(filtered_edges, dtype=np.uint32)
        filtered_edge_colors = np.array(filtered_edge_colors)
        
        return {
            'node_positions': filtered_node_positions,
            'node_colors': filtered_node_colors,
            'node_sizes': filtered_node_sizes,
            'edge_connections': filtered_edges,
            'edge_colors': filtered_edge_colors
        }

    def extract_network_subset(self, criteria, max_nodes=None, include_neighbors=False, neighbor_depth=1):
        """
        Extract a subset of the network based on specified criteria.
        This is optimized for extracting smaller portions for NetworkX analysis and Matplotlib visualization.
        
        Parameters:
        -----------
        criteria : dict
            Criteria for selecting nodes, can include:
            - layers: list of layer indices to include
            - clusters: list of cluster names to include
            - origins: list of origin names to include
            - node_ids: list of specific node IDs to include
            - node_indices: list of specific node indices to include
            - edge_indices: list of specific edge indices to include
            - base_nodes: list of base node IDs to include (will include all layers)
        
        max_nodes : int, optional
            Maximum number of nodes to include in the subset
        
        include_neighbors : bool, default=False
            Whether to include neighbors of selected nodes
            
        neighbor_depth : int, default=1
            How many hops of neighbors to include
        
        Returns:
        --------
        dict
            Dictionary containing:
            - 'networkx_graph': NetworkX graph of the subset
            - 'node_indices': Original indices of included nodes
            - 'edge_indices': Original indices of included edges
            - 'node_positions': NumPy array of node positions
            - 'node_colors': NumPy array of node colors (RGBA)
            - 'edge_connections': NumPy array of edge connections (remapped to new indices)
            - 'edge_colors': NumPy array of edge colors (RGBA)
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Extracting network subset based on criteria")
        
        # Initialize mask for all nodes (all False)
        subset_mask = np.zeros(len(self.node_positions), dtype=bool)
        
        # Apply criteria to select nodes
        if 'layers' in criteria:
            for layer_idx in criteria['layers']:
                start_idx = layer_idx * self.nodes_per_layer
                end_idx = start_idx + self.nodes_per_layer
                subset_mask[start_idx:end_idx] = True
        
        if 'clusters' in criteria:
            cluster_set = set(criteria['clusters'])
            for i, node_id in enumerate(self.node_ids):
                if self.node_clusters.get(node_id) in cluster_set:
                    subset_mask[i] = True
        
        if 'origins' in criteria and self.node_origins:
            origin_set = set(criteria['origins'])
            for i, node_id in enumerate(self.node_ids):
                if self.node_origins.get(node_id, "Unknown") in origin_set:
                    subset_mask[i] = True
        
        if 'node_ids' in criteria:
            node_id_set = set(criteria['node_ids'])
            for i, node_id in enumerate(self.node_ids):
                if node_id in node_id_set:
                    subset_mask[i] = True
        
        if 'node_indices' in criteria:
            for idx in criteria['node_indices']:
                if 0 <= idx < len(self.node_positions):
                    subset_mask[idx] = True
        
        if 'base_nodes' in criteria:
            base_node_set = set(criteria['base_nodes'])
            for i, node_id in enumerate(self.node_ids):
                base_id = node_id.split('_')[0] if '_' in node_id else node_id
                if base_id in base_node_set:
                    subset_mask[i] = True
        
        # If we specified edge indices, include their endpoints
        if 'edge_indices' in criteria:
            for edge_idx in criteria['edge_indices']:
                if 0 <= edge_idx < len(self.link_pairs):
                    start_idx, end_idx = self.link_pairs[edge_idx]
                    subset_mask[start_idx] = True
                    subset_mask[end_idx] = True
        
        # Include neighbors if requested
        if include_neighbors and neighbor_depth > 0:
            # Start with the current selection
            current_selection = np.where(subset_mask)[0]
            expanded_mask = subset_mask.copy()
            
            # For each depth level
            for depth in range(neighbor_depth):
                # Find all neighbors of the current selection
                neighbors = set()
                for edge_idx, (start_idx, end_idx) in enumerate(self.link_pairs):
                    if start_idx in current_selection and not expanded_mask[end_idx]:
                        neighbors.add(end_idx)
                    elif end_idx in current_selection and not expanded_mask[start_idx]:
                        neighbors.add(start_idx)
                
                # Add neighbors to the expanded mask
                for neighbor in neighbors:
                    expanded_mask[neighbor] = True
                
                # Update current selection for next iteration
                current_selection = np.where(expanded_mask)[0]
                
                # If we've included all nodes, stop early
                if np.all(expanded_mask):
                    break
            
            # Use the expanded mask
            subset_mask = expanded_mask
        
        # Apply maximum node limit if specified
        if max_nodes is not None and np.sum(subset_mask) > max_nodes:
            # Get indices of selected nodes
            selected_indices = np.where(subset_mask)[0]
            
            # Randomly select max_nodes indices
            chosen_indices = np.random.choice(selected_indices, max_nodes, replace=False)
            
            # Create a new mask with only the chosen indices
            new_mask = np.zeros(len(self.node_positions), dtype=bool)
            new_mask[chosen_indices] = True
            subset_mask = new_mask
        
        # Get the indices of selected nodes
        node_indices = np.where(subset_mask)[0]
        
        # If no nodes selected, return empty result
        if len(node_indices) == 0:
            logger.warning("No nodes match the specified criteria")
            return {
                'networkx_graph': nx.Graph(),
                'node_indices': np.array([]),
                'edge_indices': np.array([]),
                'node_positions': np.array([]),
                'node_colors': np.array([]),
                'edge_connections': np.array([]),
                'edge_colors': np.array([])
            }
        
        # Create mapping from original indices to new indices
        idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
        
        # Find edges where both endpoints are in the subset
        edge_indices = []
        edge_connections = []
        edge_colors = []
        
        for i, (start_idx, end_idx) in enumerate(self.link_pairs):
            if subset_mask[start_idx] and subset_mask[end_idx]:
                edge_indices.append(i)
                # Remap node indices to the new index space
                edge_connections.append([idx_mapping[start_idx], idx_mapping[end_idx]])
                edge_colors.append(self.link_colors_rgba[i])
        
        # Convert to NumPy arrays
        edge_indices = np.array(edge_indices)
        edge_connections = np.array(edge_connections, dtype=np.int32)
        edge_colors = np.array(edge_colors, dtype=np.float32)
        
        # Extract node data
        node_positions = self.node_positions[node_indices]
        node_colors = self.node_colors_rgba[node_indices]
        
        # Create NetworkX graph from the subset
        G = nx.Graph()
        
        # Add nodes with attributes
        for new_idx, orig_idx in enumerate(node_indices):
            node_id = self.node_ids[orig_idx]
            layer_idx = orig_idx // self.nodes_per_layer
            G.add_node(
                new_idx,
                id=node_id,
                original_idx=orig_idx,
                layer=layer_idx,
                cluster=self.node_clusters.get(node_id, "Unknown"),
                origin=self.node_origins.get(node_id, "Unknown"),
            )
        
        # Add edges
        for i, (src, dst) in enumerate(edge_connections):
            G.add_edge(src, dst, color=edge_colors[i])
        
        logger.info(f"Extracted subset with {len(node_indices)} nodes and {len(edge_indices)} edges")
        
        return {
            'networkx_graph': G,
            'node_indices': node_indices,
            'edge_indices': edge_indices,
            'node_positions': node_positions,
            'node_colors': node_colors,
            'edge_connections': edge_connections,
            'edge_colors': edge_colors
        }

    def optimize_for_vispy(self, max_visible_edges=100000):
        """
        Optimize the network data for efficient Vispy visualization of large networks.
        This method applies level-of-detail techniques to maintain interactive performance.
        
        Parameters:
        -----------
        max_visible_edges : int, default=100000
            Maximum number of edges to display at full detail
            
        Returns:
        --------
        dict
            Dictionary containing optimized data for Vispy:
            - 'node_positions': NumPy array of node positions
            - 'node_colors': NumPy array of node colors (RGBA)
            - 'node_sizes': NumPy array of node sizes
            - 'edge_connections': NumPy array of edge connections
            - 'edge_colors': NumPy array of edge colors (RGBA)
            - 'edge_importance': NumPy array of edge importance scores (for LOD)
            - 'is_simplified': Boolean indicating if the data was simplified
        """
        logger = logging.getLogger(__name__)
        
        # If we have a visibility mask, use it to filter the data
        if self.current_node_mask is not None and self.current_edge_mask is not None:
            # Get filtered data
            filtered_data = self.get_filtered_data_for_vispy()
            
            # Check if we need to simplify further
            if len(filtered_data['edge_connections']) <= max_visible_edges:
                logger.info(f"Using filtered data for Vispy: {len(filtered_data['node_positions'])} nodes, {len(filtered_data['edge_connections'])} edges")
                
                # Add edge importance scores (all 1.0 since we're not simplifying)
                filtered_data['edge_importance'] = np.ones(len(filtered_data['edge_connections']))
                filtered_data['is_simplified'] = False
                
                return filtered_data
        
        # If we have too many edges or no visibility mask, we need to simplify
        logger.info(f"Simplifying network for Vispy visualization")
        
        # Start with all nodes
        node_positions = self.node_positions
        
        # Handle node colors - might be None, a list, or a NumPy array
        if self.node_colors_rgba is None:
            # Create default colors if none exist
            node_colors = np.ones((len(node_positions), 4))
        elif isinstance(self.node_colors_rgba, np.ndarray):
            # If it's already a NumPy array, use it directly
            node_colors = self.node_colors_rgba
        else:
            # If it's a list or other format, convert to NumPy array
            try:
                node_colors = np.array(self.node_colors_rgba)
                if node_colors.shape[0] != len(node_positions):
                    # If the shape doesn't match, create default colors
                    logger.warning(f"Node colors shape mismatch: {node_colors.shape[0]} vs {len(node_positions)}")
                    node_colors = np.ones((len(node_positions), 4))
            except Exception as e:
                logger.error(f"Error converting node colors to NumPy array: {e}")
                # Default to white
                node_colors = np.ones((len(node_positions), 4))
                
        node_sizes = self.node_sizes
        
        # Calculate edge importance scores
        # We'll use these to determine which edges to show at different detail levels
        edge_importance = np.zeros(len(self.link_pairs))
        
        # 1. Interlayer edges are more important (they show the network structure)
        for i, (start_idx, end_idx) in enumerate(self.link_pairs):
            start_layer = start_idx // self.nodes_per_layer
            end_layer = end_idx // self.nodes_per_layer
            if start_layer != end_layer:
                edge_importance[i] += 2.0
            else:
                edge_importance[i] += 1.0
        
        # 2. Edges connected to high-degree nodes are more important
        node_degrees = np.zeros(len(node_positions))
        for start_idx, end_idx in self.link_pairs:
            node_degrees[start_idx] += 1
            node_degrees[end_idx] += 1
        
        # Normalize degrees to [0, 1]
        max_degree = np.max(node_degrees) if len(node_degrees) > 0 else 1
        normalized_degrees = node_degrees / max(max_degree, 1)
        
        # Add degree-based importance
        for i, (start_idx, end_idx) in enumerate(self.link_pairs):
            edge_importance[i] += (normalized_degrees[start_idx] + normalized_degrees[end_idx]) / 2
        
        # If we have too many edges, select the most important ones
        if len(self.link_pairs) > max_visible_edges:
            # Sort edges by importance (descending)
            sorted_indices = np.argsort(-edge_importance)
            
            # Select the top max_visible_edges
            selected_edge_indices = sorted_indices[:max_visible_edges]
            
            # Create simplified edge data
            edge_connections = np.array([self.link_pairs[i] for i in selected_edge_indices])
            edge_colors = np.array([self.link_colors_rgba[i] for i in selected_edge_indices])
            simplified_importance = edge_importance[selected_edge_indices]
            
            logger.info(f"Simplified network for Vispy: {len(node_positions)} nodes, {len(edge_connections)} edges (from {len(self.link_pairs)})")
            
            return {
                'node_positions': node_positions,
                'node_colors': node_colors,
                'node_sizes': node_sizes,
                'edge_connections': edge_connections,
                'edge_colors': edge_colors,
                'edge_importance': simplified_importance,
                'is_simplified': True
            }
        else:
            # No simplification needed
            return {
                'node_positions': node_positions,
                'node_colors': node_colors,
                'node_sizes': node_sizes,
                'edge_connections': np.array(self.link_pairs),
                'edge_colors': np.array(self.link_colors_rgba),
                'edge_importance': edge_importance,
                'is_simplified': False
            }
