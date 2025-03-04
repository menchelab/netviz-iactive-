import numpy as np
import logging
import networkx as nx
from scipy import sparse
from utils.color_utils import hex_to_rgba


class SparseNetworkDataManager:
    """
    Enhanced data manager using sparse matrices for efficient storage and operations
    on multilayer network data.
    """

    def __init__(self, data_dir=None):
        logger = logging.getLogger(__name__)
        logger.info("Initializing SparseNetworkDataManager")

        # Store data directory
        self._data_dir = data_dir

        # Core network structure
        self.n_nodes = 0  # Total unique base nodes
        self.n_layers = 0  # Number of layers
        self.layer_names = {}  # Mapping from layer index to name
        self.layer_indices = {}  # Mapping from layer name to index

        # Node information
        self.base_node_ids = []  # List of base node IDs
        self.node_id_to_index = {}  # Mapping from node ID to index
        self.base_positions = None  # Base x,y positions (n_nodes, 2)
        self.z_positions = None  # z-position for each layer

        # Layer membership - sparse matrix (n_nodes, n_layers)
        # 1 if node exists in layer, 0 otherwise
        self.layer_membership = None

        # Adjacency matrices - one sparse matrix per layer
        self.adjacency_matrices = []  # List of sparse matrices

        # Interlayer connections - sparse matrix (n_nodes*n_layers, n_nodes*n_layers)
        self.interlayer_connections = None

        # Node attributes
        self.node_attributes = {}  # name -> sparse matrix or array

        # Edge attributes
        self.edge_attributes = {}  # name -> list of sparse matrices (one per layer)

        # Visibility state
        self.visible_layers = None  # Boolean array (n_layers,)
        self.visible_clusters = None  # Set of visible cluster IDs
        self.visible_origins = None  # Set of visible origin IDs

        # Compatibility attributes (for existing code)
        self.node_positions = None
        self.link_pairs = None
        self.link_colors = None
        self.link_colors_rgba = None
        self.node_colors = None
        self.node_colors_rgba = None
        self.node_ids = None
        self.layers = None
        self.node_clusters = None
        self.unique_clusters = None
        self.node_origins = None
        self.unique_origins = None
        self.layer_colors = None
        self.layer_colors_rgba = None
        self.nodes_per_layer = None
        self.active_nodes = None
        self.node_sizes = None
        self.base_node_mapping = None
        self.current_node_mask = None
        self.current_edge_mask = None
        self.cluster_colors = {}

        # Cache for derived data
        self._cache = {}

    def _reset_cache(self):
        """Reset all cached calculations"""
        self._cache = {}
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
        node_clusters=None,
        unique_clusters=None,
        node_colors=None,
        node_origins=None,
        unique_origins=None,
        layer_colors=None,
    ):
        """Load network data into sparse matrix format"""
        logger = logging.getLogger(__name__)
        logger.info("Loading data into SparseNetworkDataManager")
        
        # Log information about input parameters
        logger.info(f"Number of nodes: {len(node_ids)}")
        logger.info(f"Number of layers: {len(layers)}")
        logger.info(f"Type of node_clusters: {type(node_clusters)}")
        if node_clusters is not None:
            if hasattr(node_clusters, '__len__'):
                logger.info(f"Length of node_clusters: {len(node_clusters)}")
            if isinstance(node_clusters, dict):
                logger.info(f"node_clusters keys sample: {list(node_clusters.keys())[:10]}")
            elif isinstance(node_clusters, (list, tuple, np.ndarray)):
                logger.info(f"node_clusters sample: {node_clusters[:min(10, len(node_clusters))]}")
        
        logger.info(f"unique_clusters: {unique_clusters}")

        # Store basic information
        self.n_layers = len(layers)
        self.layers = layers
        self.layer_names = {i: layer for i, layer in enumerate(layers)}
        self.layer_indices = {layer: i for i, layer in enumerate(layers)}

        # Process node IDs to extract base nodes
        base_nodes = set()
        for node_id in node_ids:
            base_node = node_id.split("_")[0] if "_" in node_id else node_id
            base_nodes.add(base_node)

        self.base_node_ids = sorted(list(base_nodes))
        self.n_nodes = len(self.base_node_ids)
        self.node_id_to_index = {node: i for i, node in enumerate(self.base_node_ids)}

        # Create layer membership matrix (sparse)
        rows, cols, data = [], [], []
        for i, node_id in enumerate(node_ids):
            base_node = node_id.split("_")[0] if "_" in node_id else node_id
            layer_name = node_id.split("_")[1] if "_" in node_id else layers[0]
            base_idx = self.node_id_to_index[base_node]
            layer_idx = self.layer_indices[layer_name]

            rows.append(base_idx)
            cols.append(layer_idx)
            data.append(1)

        self.layer_membership = sparse.csr_matrix(
            (data, (rows, cols)), shape=(self.n_nodes, self.n_layers)
        )

        # Store base positions (x,y only)
        self.base_positions = np.zeros((self.n_nodes, 2))
        for i, node_id in enumerate(node_ids):
            base_node = node_id.split("_")[0] if "_" in node_id else node_id
            base_idx = self.node_id_to_index[base_node]
            # Store only x,y coordinates
            self.base_positions[base_idx] = node_positions[i][:2]

        # Store z positions for layers
        self.z_positions = np.linspace(0, self.n_layers - 1, self.n_layers)

        # Create adjacency matrices (one per layer)
        self.adjacency_matrices = [
            sparse.lil_matrix((self.n_nodes, self.n_nodes))
            for _ in range(self.n_layers)
        ]

        # Create interlayer connections matrix
        self.interlayer_connections = sparse.lil_matrix(
            (self.n_nodes * self.n_layers, self.n_nodes * self.n_layers)
        )

        # Process links
        for i, (src, dst) in enumerate(link_pairs):
            src_id = node_ids[src]
            dst_id = node_ids[dst]

            src_base = src_id.split("_")[0] if "_" in src_id else src_id
            dst_base = dst_id.split("_")[0] if "_" in dst_id else dst_id

            src_layer = src_id.split("_")[1] if "_" in src_id else layers[0]
            dst_layer = dst_id.split("_")[1] if "_" in dst_id else layers[0]

            src_base_idx = self.node_id_to_index[src_base]
            dst_base_idx = self.node_id_to_index[dst_base]

            src_layer_idx = self.layer_indices[src_layer]
            dst_layer_idx = self.layer_indices[dst_layer]

            # If same layer, add to layer adjacency matrix
            if src_layer == dst_layer:
                self.adjacency_matrices[src_layer_idx][src_base_idx, dst_base_idx] = 1
                self.adjacency_matrices[src_layer_idx][dst_base_idx, src_base_idx] = (
                    1  # Undirected
                )
            else:
                # Add to interlayer connections
                src_idx = src_layer_idx * self.n_nodes + src_base_idx
                dst_idx = dst_layer_idx * self.n_nodes + dst_base_idx

                self.interlayer_connections[src_idx, dst_idx] = 1
                self.interlayer_connections[dst_idx, src_idx] = 1  # Undirected

        # Store node attributes (clusters, colors, origins)
        if node_clusters is not None:
            self._store_node_attribute("cluster", node_clusters, node_ids)
            self.node_clusters = node_clusters
            self.unique_clusters = unique_clusters

        if node_colors is not None:
            self._store_node_attribute("color", node_colors, node_ids)
            self.node_colors = node_colors

        if node_origins is not None:
            self._store_node_attribute("origin", node_origins, node_ids)
            self.node_origins = node_origins
            self.unique_origins = unique_origins

        # Store layer colors
        self.layer_colors = layer_colors or {}

        # Extract cluster colors from metadata
        self.cluster_colors = {}
        if unique_clusters:
            logger.info(f"Extracting cluster colors for {unique_clusters}")
            for cluster in unique_clusters:
                # Find a node with this cluster and get its color
                for i, node_id in enumerate(node_ids):
                    # Check if this node has the current cluster
                    if isinstance(node_clusters, dict):
                        node_cluster = node_clusters.get(node_id)
                    else:
                        node_cluster = node_clusters[i] if i < len(node_clusters) else None
                    
                    if (
                        node_cluster == cluster
                        and node_colors
                        and i < len(node_colors)
                    ):
                        self.cluster_colors[cluster] = node_colors[i]
                        logger.info(f"Found color for cluster {cluster}: {node_colors[i]}")
                        break

        # Initialize visibility state
        self.visible_layers = np.ones(self.n_layers, dtype=bool)
        self.visible_clusters = set(unique_clusters) if unique_clusters else set()
        self.visible_origins = set(unique_origins) if unique_origins else set()

        # For compatibility with existing code
        self.node_positions = node_positions
        self.link_pairs = link_pairs
        self.link_colors = link_colors
        self.node_ids = node_ids
        self.nodes_per_layer = len(node_ids) // len(layers)

        # Convert colors to RGBA if needed
        if link_colors:
            # Convert to numpy array for compatibility with visibility manager
            self.link_colors_rgba = np.array([hex_to_rgba(color) for color in link_colors])
        else:
            # Initialize with default colors if none provided
            self.link_colors_rgba = np.ones((len(link_pairs), 4)) * np.array([0.5, 0.5, 0.5, 1.0])  # Gray with full opacity
            
        if node_colors:
            # Convert to numpy array for compatibility with visibility manager
            self.node_colors_rgba = np.array([hex_to_rgba(color) for color in node_colors])
        else:
            # Initialize with default colors if none provided
            self.node_colors_rgba = np.ones((len(node_positions), 4))  # White with full opacity
            
        if layer_colors:
            self.layer_colors_rgba = {
                k: hex_to_rgba(v) for k, v in layer_colors.items()
            }
            
        # Initialize node sizes
        self.node_sizes = np.ones(len(node_positions)) * 3  # Default size
        
        # Determine which nodes actually exist in each layer based on edges
        self.active_nodes = np.zeros(len(node_positions), dtype=bool)
        for start_idx, end_idx in link_pairs:
            self.active_nodes[start_idx] = True
            self.active_nodes[end_idx] = True
            
        # Set larger size for active nodes
        self.node_sizes[self.active_nodes] = 9  # 3x larger for active nodes

        # Create base node mapping
        self.base_node_mapping = {}
        for i, node_id in enumerate(node_ids):
            base_node = node_id.split("_")[0] if "_" in node_id else node_id
            if base_node not in self.base_node_mapping:
                self.base_node_mapping[base_node] = []
            self.base_node_mapping[base_node].append(i)

        # Initialize current masks
        self.update_visibility()

        # Clear cache
        self._reset_cache()

    def _store_node_attribute(self, attr_name, attr_values, node_ids):
        """Store node attributes in an efficient format based on data type"""
        logger = logging.getLogger(__name__)
        logger.info(f"Storing node attribute: {attr_name}")
        logger.info(f"Type of attr_values: {type(attr_values)}")
        logger.info(f"Type of node_ids: {type(node_ids)}")
        logger.info(f"Length of node_ids: {len(node_ids)}")
        
        if hasattr(attr_values, '__len__'):
            logger.info(f"Length of attr_values: {len(attr_values)}")
        
        # Log a sample of the values
        if isinstance(attr_values, dict):
            logger.info(f"attr_values is a dictionary with keys: {list(attr_values.keys())[:10]}")
        elif isinstance(attr_values, (list, tuple, np.ndarray)):
            logger.info(f"First few attr_values: {attr_values[:min(10, len(attr_values))]}")
        
        # Determine if attribute is categorical or numerical
        if isinstance(attr_values, dict):
            sample_values = list(set(attr_values.values()))[:10]  # Take a sample
        else:
            sample_values = list(set(attr_values))[:10]  # Take a sample
            
        logger.info(f"Sample values: {sample_values}")
        
        is_categorical = all(
            not isinstance(v, (int, float)) or isinstance(v, bool)
            for v in sample_values
        )
        logger.info(f"Is categorical: {is_categorical}")

        if is_categorical:
            # For categorical attributes, store as a sparse matrix of indices
            if isinstance(attr_values, dict):
                unique_values = sorted(set(attr_values.values()))
            else:
                unique_values = sorted(set(attr_values))
                
            logger.info(f"Unique values: {unique_values}")
            value_to_idx = {val: i for i, val in enumerate(unique_values)}
            logger.info(f"value_to_idx mapping: {value_to_idx}")

            rows, cols, data = [], [], []
            for i, node_id in enumerate(node_ids):
                logger.info(f"Processing node {i}: {node_id}")
                base_node = node_id.split("_")[0] if "_" in node_id else node_id
                layer_name = (
                    node_id.split("_")[1] if "_" in node_id else self.layer_names[0]
                )

                base_idx = self.node_id_to_index[base_node]
                layer_idx = self.layer_indices[layer_name]

                # Get the attribute value for this node
                try:
                    if isinstance(attr_values, dict):
                        # For dictionary attributes, look up by node_id
                        current_value = attr_values.get(node_id)
                        logger.info(f"Dict lookup: node_id={node_id}, value={current_value}")
                    else:
                        # For list/array attributes, look up by index
                        current_value = attr_values[i]
                        logger.info(f"List lookup: i={i}, value={current_value}")
                    
                    # Skip if value is None or not found
                    if current_value is None:
                        logger.warning(f"No value found for node {node_id}, skipping")
                        continue
                    
                    # Convert value to Python native type to ensure consistent lookup
                    if isinstance(current_value, (np.integer, np.floating)):
                        current_value = current_value.item()  # Convert numpy type to Python native type
                    
                    logger.info(f"Final current_value: {current_value} (type: {type(current_value)})")
                    value_idx = value_to_idx[current_value]
                    logger.info(f"value_idx: {value_idx}")
                    
                    rows.append(base_idx)
                    cols.append(layer_idx)
                    data.append(value_idx)
                except Exception as e:
                    logger.error(f"Error processing node {node_id} at index {i}: {str(e)}")
                    logger.error(f"attr_values type: {type(attr_values)}")
                    if isinstance(attr_values, dict):
                        logger.error(f"attr_values keys sample: {list(attr_values.keys())[:20]}")
                        logger.error(f"Is node_id in keys: {node_id in attr_values}")
                    raise

            # Store as sparse matrix
            attr_matrix = sparse.csr_matrix(
                (data, (rows, cols)), shape=(self.n_nodes, self.n_layers)
            )

            self.node_attributes[attr_name] = {
                "type": "categorical",
                "values": unique_values,
                "matrix": attr_matrix,
            }
        else:
            # For numerical attributes, store as a sparse matrix of values
            rows, cols, data = [], [], []
            for i, node_id in enumerate(node_ids):
                logger.info(f"Processing numerical node {i}: {node_id}")
                base_node = node_id.split("_")[0] if "_" in node_id else node_id
                layer_name = (
                    node_id.split("_")[1] if "_" in node_id else self.layer_names[0]
                )

                base_idx = self.node_id_to_index[base_node]
                layer_idx = self.layer_indices[layer_name]

                # Get the attribute value for this node
                try:
                    if isinstance(attr_values, dict):
                        # For dictionary attributes, look up by node_id
                        current_value = attr_values.get(node_id)
                        logger.info(f"Dict lookup (numerical): node_id={node_id}, value={current_value}")
                    else:
                        # For list/array attributes, look up by index
                        current_value = attr_values[i]
                        logger.info(f"List lookup (numerical): i={i}, value={current_value}")
                    
                    # Skip if value is None or not found
                    if current_value is None:
                        logger.warning(f"No numerical value found for node {node_id}, skipping")
                        continue
                    
                    # Convert value to float
                    if isinstance(current_value, (np.integer, np.floating)):
                        current_value = current_value.item()  # Convert numpy type to Python native type
                    
                    rows.append(base_idx)
                    cols.append(layer_idx)
                    data.append(float(current_value))
                except Exception as e:
                    logger.error(f"Error processing numerical node {node_id} at index {i}: {str(e)}")
                    logger.error(f"attr_values type: {type(attr_values)}")
                    if isinstance(attr_values, dict):
                        logger.error(f"attr_values keys sample: {list(attr_values.keys())[:20]}")
                        logger.error(f"Is node_id in keys: {node_id in attr_values}")
                    continue  # Skip this node and continue with others

            # Store as sparse matrix
            attr_matrix = sparse.csr_matrix(
                (data, (rows, cols)), shape=(self.n_nodes, self.n_layers)
            )

            self.node_attributes[attr_name] = {
                "type": "numerical",
                "matrix": attr_matrix,
            }

    def update_visibility(
        self, visible_layers=None, visible_clusters=None, visible_origins=None
    ):
        """Update visibility state based on filters"""

        changed = False

        if visible_layers is not None and not np.array_equal(
            self.visible_layers, visible_layers
        ):
            self.visible_layers = visible_layers
            changed = True

        if visible_clusters is not None and self.visible_clusters != set(
            visible_clusters
        ):
            self.visible_clusters = set(visible_clusters)
            changed = True

        if visible_origins is not None and self.visible_origins != set(visible_origins):
            self.visible_origins = set(visible_origins)
            changed = True

        if changed or self.current_node_mask is None or self.current_edge_mask is None:
            # Clear relevant cache entries
            self._cache.pop("visible_node_mask", None)
            self._cache.pop("visible_edge_mask", None)
            self._cache.pop("visible_interlayer_edges", None)
            self._cache.pop("networkx_graph", None)

            # Update current masks
            self.current_node_mask = self._calculate_node_mask()
            self.current_edge_mask = self._calculate_edge_mask()

        return changed

    def _calculate_node_mask(self):
        """Calculate node mask based on current visibility settings"""
        logger = logging.getLogger(__name__)
        logger.info("Calculating node mask based on visibility settings")
        
        if "node_mask" in self._cache:
            logger.info("Using cached node mask")
            return self._cache["node_mask"]

        # Create a boolean mask for all nodes
        node_mask = np.zeros(len(self.node_ids), dtype=bool)

        # For each node, check if it should be visible
        for i, node_id in enumerate(self.node_ids):
            # Extract base node and layer
            base_node = node_id.split("_")[0] if "_" in node_id else node_id
            layer_name = node_id.split("_")[1] if "_" in node_id else self.layers[0]
            layer_idx = self.layer_indices[layer_name]

            # Check layer visibility
            if not self.visible_layers[layer_idx]:
                continue

            # Check cluster visibility if applicable
            if (
                self.visible_clusters
                and hasattr(self, "node_clusters")
                and self.node_clusters is not None
            ):
                # Get cluster for this node
                if isinstance(self.node_clusters, dict):
                    node_cluster = self.node_clusters.get(node_id)
                else:
                    node_cluster = self.node_clusters[i] if i < len(self.node_clusters) else None
                
                if node_cluster is None:
                    logger.warning(f"No cluster found for node {node_id}")
                    continue
                    
                if node_cluster not in self.visible_clusters:
                    continue

            # Check origin visibility if applicable
            if (
                self.visible_origins
                and hasattr(self, "node_origins")
                and self.node_origins is not None
            ):
                # Get origin for this node
                if isinstance(self.node_origins, dict):
                    node_origin = self.node_origins.get(node_id)
                else:
                    node_origin = self.node_origins[i] if i < len(self.node_origins) else None
                
                if node_origin is None:
                    logger.warning(f"No origin found for node {node_id}")
                    continue
                    
                if node_origin not in self.visible_origins:
                    continue

            # If we got here, the node is visible
            node_mask[i] = True

        self._cache["node_mask"] = node_mask
        return node_mask

    def _calculate_edge_mask(self):
        """Calculate edge mask based on current node visibility"""
        if "edge_mask" in self._cache:
            return self._cache["edge_mask"]

        # Get node mask
        node_mask = (
            self.current_node_mask
            if self.current_node_mask is not None
            else self._calculate_node_mask()
        )

        # Create a boolean mask for all edges
        edge_mask = np.zeros(len(self.link_pairs), dtype=bool)

        # For each edge, check if both endpoints are visible
        for i, (src, dst) in enumerate(self.link_pairs):
            if node_mask[src] and node_mask[dst]:
                edge_mask[i] = True

        self._cache["edge_mask"] = edge_mask
        return edge_mask

    def get_visualization_data(
        self, show_nodes=True, show_intralayer=True, show_interlayer=True
    ):
        """
        Extract data needed for visualization

        Returns:
        --------
        dict with keys:
            - node_positions: array of shape (n_visible_nodes, 3)
            - node_colors: array of shape (n_visible_nodes, 4)
            - node_sizes: array of shape (n_visible_nodes,)
            - intralayer_edges: array of shape (n_visible_intralayer_edges*2, 3)
            - intralayer_colors: array of shape (n_visible_intralayer_edges*2, 4)
            - interlayer_edges: array of shape (n_visible_interlayer_edges*2, 3)
            - interlayer_colors: array of shape (n_visible_interlayer_edges*2, 4)
        """
        # Get visible node mask
        node_mask = self.current_node_mask
        edge_mask = self.current_edge_mask

        # Prepare result dictionary
        result = {}

        if show_nodes:
            # Extract visible node positions and colors
            result["node_positions"] = self.node_positions[node_mask]
            result["node_colors"] = (
                self.node_colors_rgba[node_mask] if self.node_colors_rgba is not None else None
            )
            result["node_sizes"] = (
                self.node_sizes[node_mask] if self.node_sizes is not None else None
            )

        # Extract visible edges
        if show_intralayer or show_interlayer:
            visible_links = [
                self.link_pairs[i] for i, mask in enumerate(edge_mask) if mask
            ]
            visible_colors = (
                [self.link_colors_rgba[i] for i, mask in enumerate(edge_mask) if mask]
                if self.link_colors_rgba
                else None
            )

            # Separate intralayer and interlayer edges
            intralayer_edges = []
            intralayer_colors = []
            interlayer_edges = []
            interlayer_colors = []

            for i, (src, dst) in enumerate(visible_links):
                src_layer = src // self.nodes_per_layer
                dst_layer = dst // self.nodes_per_layer

                if src_layer == dst_layer and show_intralayer:
                    # Intralayer edge
                    intralayer_edges.append((src, dst))
                    if visible_colors:
                        intralayer_colors.append(visible_colors[i])
                elif src_layer != dst_layer and show_interlayer:
                    # Interlayer edge
                    interlayer_edges.append((src, dst))
                    if visible_colors:
                        interlayer_colors.append(visible_colors[i])

            result["intralayer_edges"] = intralayer_edges
            result["intralayer_colors"] = intralayer_colors
            result["interlayer_edges"] = interlayer_edges
            result["interlayer_colors"] = interlayer_colors

        return result

    def get_networkx_graph(self, visible_only=True):
        """
        Convert to NetworkX graph for analysis

        Parameters:
        -----------
        visible_only : bool
            If True, include only visible nodes and edges

        Returns:
        --------
        G : networkx.Graph
            NetworkX graph representation
        """
        cache_key = f"networkx_graph_{visible_only}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        G = nx.Graph()

        # Get node and edge masks
        node_mask = (
            self.current_node_mask
            if visible_only
            else np.ones(len(self.node_ids), dtype=bool)
        )
        edge_mask = (
            self.current_edge_mask
            if visible_only
            else np.ones(len(self.link_pairs), dtype=bool)
        )

        # Add nodes with attributes
        for i, node_id in enumerate(self.node_ids):
            if node_mask[i]:
                # Extract base node and layer
                base_node = node_id.split("_")[0] if "_" in node_id else node_id
                layer_name = node_id.split("_")[1] if "_" in node_id else self.layers[0]

                # Collect attributes
                attrs = {
                    "base_node": base_node,
                    "layer": layer_name,
                    "pos": self.node_positions[i],
                }

                # Add other attributes
                if hasattr(self, "node_clusters") and self.node_clusters is not None:
                    if isinstance(self.node_clusters, dict):
                        attrs["cluster"] = self.node_clusters.get(node_id)
                    else:
                        attrs["cluster"] = self.node_clusters[i] if i < len(self.node_clusters) else None

                if hasattr(self, "node_origins") and self.node_origins is not None:
                    if isinstance(self.node_origins, dict):
                        attrs["origin"] = self.node_origins.get(node_id)
                    else:
                        attrs["origin"] = self.node_origins[i] if i < len(self.node_origins) else None

                G.add_node(node_id, **attrs)

        # Add edges
        for i, (src, dst) in enumerate(self.link_pairs):
            if edge_mask[i]:
                src_id = self.node_ids[src]
                dst_id = self.node_ids[dst]

                # Determine if this is an intralayer or interlayer edge
                src_layer = src // self.nodes_per_layer
                dst_layer = dst // self.nodes_per_layer

                if src_layer == dst_layer:
                    edge_type = "intralayer"
                    layer = self.layers[src_layer]
                    G.add_edge(src_id, dst_id, type=edge_type, layer=layer)
                else:
                    edge_type = "interlayer"
                    src_layer_name = self.layers[src_layer]
                    dst_layer_name = self.layers[dst_layer]
                    G.add_edge(
                        src_id,
                        dst_id,
                        type=edge_type,
                        src_layer=src_layer_name,
                        dst_layer=dst_layer_name,
                    )

        # Cache the result
        self._cache[cache_key] = G

        return G

    def get_interlayer_edge_counts(self):
        """Get counts of interlayer edges for each base node"""
        if "interlayer_edge_counts" in self._cache:
            return self._cache["interlayer_edge_counts"]

        interlayer_edge_counts = {}

        # Get edge mask
        edge_mask = self.current_edge_mask

        for i, (src, dst) in enumerate(self.link_pairs):
            # Skip edges that aren't visible
            if not edge_mask[i]:
                continue

            # Skip intralayer edges
            src_layer = src // self.nodes_per_layer
            dst_layer = dst // self.nodes_per_layer
            if src_layer == dst_layer:
                continue

            # Count this edge for both source and destination nodes
            src_id = self.node_ids[src].split("_")[0]
            dst_id = self.node_ids[dst].split("_")[0]

            if src_id not in interlayer_edge_counts:
                interlayer_edge_counts[src_id] = 0
            if dst_id not in interlayer_edge_counts:
                interlayer_edge_counts[dst_id] = 0

            interlayer_edge_counts[src_id] += 1
            interlayer_edge_counts[dst_id] += 1

        self._cache["interlayer_edge_counts"] = interlayer_edge_counts
        return interlayer_edge_counts

    def get_layer_connections(self, filter_to_visible=True):
        """
        Get a matrix of connections between layers

        Parameters:
        -----------
        filter_to_visible : bool
            If True, only include visible layers

        Returns:
        --------
        layer_connections : numpy.ndarray
            Matrix of shape (n_layers, n_layers) where each cell contains
            the number of edges between layers
        """
        if "layer_connections" in self._cache and filter_to_visible == self._cache.get(
            "layer_connections_filtered", True
        ):
            return self._cache["layer_connections"]

        # Get visible layers
        if filter_to_visible:
            visible_layer_indices = np.where(self.visible_layers)[0]
            n_visible_layers = len(visible_layer_indices)
        else:
            visible_layer_indices = np.arange(self.n_layers)
            n_visible_layers = self.n_layers

        # Create matrix
        layer_connections = np.zeros((n_visible_layers, n_visible_layers), dtype=int)

        # Get edge mask
        edge_mask = (
            self.current_edge_mask
            if filter_to_visible
            else np.ones(len(self.link_pairs), dtype=bool)
        )

        # Fill matrix
        for i, (src, dst) in enumerate(self.link_pairs):
            if not edge_mask[i]:
                continue

            src_layer = src // self.nodes_per_layer
            dst_layer = dst // self.nodes_per_layer

            if filter_to_visible:
                # Convert to filtered indices
                if (
                    src_layer not in visible_layer_indices
                    or dst_layer not in visible_layer_indices
                ):
                    continue

                src_idx = np.where(visible_layer_indices == src_layer)[0][0]
                dst_idx = np.where(visible_layer_indices == dst_layer)[0][0]
            else:
                src_idx = src_layer
                dst_idx = dst_layer

            layer_connections[src_idx, dst_idx] += 1
            if src_idx != dst_idx:  # Don't double-count self-connections
                layer_connections[dst_idx, src_idx] += 1

        self._cache["layer_connections"] = layer_connections
        self._cache["layer_connections_filtered"] = filter_to_visible

        return layer_connections

    def count_active_nodes_for_base(self, base_node, reference_idx):
        """Count active nodes with this base ID across all layers"""
        if base_node not in self.base_node_mapping:
            return 0

        count = 0
        for idx in self.base_node_mapping[base_node]:
            if idx != reference_idx and self.current_node_mask[idx]:
                count += 1

        return count


# For backward compatibility
NetworkDataManager = SparseNetworkDataManager
