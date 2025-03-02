import logging
import networkx as nx
import numpy as np
from collections import defaultdict

# List of available community detection algorithms
AVAILABLE_COMMUNITY_ALGORITHMS = [
    "louvain",
    "leiden",
    "girvan_newman",
    "label_propagation",
    "greedy_modularity",
    "fluid_communities",
    "asyn_fluidc",
    "k_clique",
    "spectral_clustering"
]

def detect_communities(G, algorithm="louvain", resolution=1.0, k=None):
    """
    Detect communities in a graph using the specified algorithm.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to analyze
    algorithm : str
        The community detection algorithm to use
    resolution : float
        Resolution parameter for algorithms that support it (e.g., Louvain)
    k : int
        Number of communities for algorithms that require it (e.g., spectral)
        
    Returns:
    --------
    dict
        Mapping of node -> community ID
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Detecting communities using {algorithm} algorithm")
    
    if algorithm == "louvain":
        return detect_louvain_communities(G, resolution)
    elif algorithm == "leiden":
        return detect_leiden_communities(G, resolution)
    elif algorithm == "girvan_newman":
        return detect_girvan_newman_communities(G)
    elif algorithm == "label_propagation":
        return detect_label_propagation_communities(G)
    elif algorithm == "greedy_modularity":
        return detect_greedy_modularity_communities(G)
    elif algorithm == "fluid_communities" or algorithm == "asyn_fluidc":
        return detect_fluid_communities(G, k)
    elif algorithm == "k_clique":
        return detect_k_clique_communities(G)
    elif algorithm == "spectral_clustering":
        return detect_spectral_clustering(G, k)
    else:
        logger.warning(f"Unknown algorithm: {algorithm}, falling back to Louvain")
        return detect_louvain_communities(G, resolution)

def detect_louvain_communities(G, resolution=1.0):
    """
    Detect communities using the Louvain algorithm.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to analyze
    resolution : float
        Resolution parameter (higher values lead to smaller communities)
        
    Returns:
    --------
    dict
        Mapping of node -> community ID
    """
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, weight='weight', resolution=resolution)
        return partition
    except ImportError:
        logging.warning("python-louvain package not found, falling back to label propagation")
        return detect_label_propagation_communities(G)

def detect_leiden_communities(G, resolution=1.0):
    """
    Detect communities using the Leiden algorithm.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to analyze
    resolution : float
        Resolution parameter (higher values lead to smaller communities)
        
    Returns:
    --------
    dict
        Mapping of node -> community ID
    """
    try:
        import leidenalg
        import igraph as ig
        
        # Convert networkx graph to igraph
        edges = list(G.edges())
        weights = [G.get_edge_data(u, v).get('weight', 1.0) for u, v in edges]
        
        # Create igraph graph
        g_ig = ig.Graph(edges=edges, directed=G.is_directed())
        g_ig.es['weight'] = weights
        
        # Run Leiden algorithm
        partition = leidenalg.find_partition(
            g_ig, 
            leidenalg.ModularityVertexPartition, 
            weights='weight',
            resolution_parameter=resolution
        )
        
        # Convert result back to node -> community mapping
        node_list = list(G.nodes())
        community_dict = {node_list[i]: membership for i, membership in enumerate(partition.membership)}
        return community_dict
    except ImportError:
        logging.warning("leidenalg package not found, falling back to Louvain")
        return detect_louvain_communities(G, resolution)

def detect_girvan_newman_communities(G):
    """
    Detect communities using the Girvan-Newman algorithm.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to analyze
        
    Returns:
    --------
    dict
        Mapping of node -> community ID
    """
    # Limit to smaller graphs due to computational complexity
    if G.number_of_nodes() > 100:
        logging.warning("Graph too large for Girvan-Newman, falling back to label propagation")
        return detect_label_propagation_communities(G)
    
    comp = nx.community.girvan_newman(G)
    
    # Take the first level of communities
    communities = next(comp)
    
    # Convert to node -> community mapping
    community_dict = {}
    for i, community in enumerate(communities):
        for node in community:
            community_dict[node] = i
            
    return community_dict

def detect_label_propagation_communities(G):
    """
    Detect communities using label propagation.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to analyze
        
    Returns:
    --------
    dict
        Mapping of node -> community ID
    """
    communities = nx.community.label_propagation_communities(G)
    
    # Convert to node -> community mapping
    community_dict = {}
    for i, community in enumerate(communities):
        for node in community:
            community_dict[node] = i
            
    return community_dict

def detect_greedy_modularity_communities(G):
    """
    Detect communities using greedy modularity maximization.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to analyze
        
    Returns:
    --------
    dict
        Mapping of node -> community ID
    """
    communities = nx.community.greedy_modularity_communities(G, weight='weight')
    
    # Convert to node -> community mapping
    community_dict = {}
    for i, community in enumerate(communities):
        for node in community:
            community_dict[node] = i
            
    return community_dict

def detect_fluid_communities(G, k=None):
    """
    Detect communities using the fluid communities algorithm.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to analyze
    k : int
        Number of communities to find (required)
        
    Returns:
    --------
    dict
        Mapping of node -> community ID
    """
    if k is None:
        # Estimate a reasonable number of communities if not specified
        k = max(2, min(10, G.number_of_nodes() // 10))
    
    try:
        communities = nx.community.asyn_fluidc(G, k, max_iter=100, seed=42)
        
        # Convert to node -> community mapping
        community_dict = {}
        for i, community in enumerate(communities):
            for node in community:
                community_dict[node] = i
                
        return community_dict
    except Exception as e:
        logging.warning(f"Fluid communities algorithm failed: {e}, falling back to label propagation")
        return detect_label_propagation_communities(G)

def detect_k_clique_communities(G, k=3):
    """
    Detect communities using k-clique percolation.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to analyze
    k : int
        Size of the cliques to find
        
    Returns:
    --------
    dict
        Mapping of node -> community ID
    """
    try:
        communities = nx.community.k_clique_communities(G, k)
        
        # Convert to node -> community mapping
        community_dict = {}
        for i, community in enumerate(communities):
            for node in community:
                community_dict[node] = i
        
        # Handle nodes not in any community
        for node in G.nodes():
            if node not in community_dict:
                community_dict[node] = -1  # Assign to a special "no community" group
                
        return community_dict
    except Exception as e:
        logging.warning(f"k-clique communities algorithm failed: {e}, falling back to label propagation")
        return detect_label_propagation_communities(G)

def detect_spectral_clustering(G, k=None):
    """
    Detect communities using spectral clustering.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to analyze
    k : int
        Number of communities to find
        
    Returns:
    --------
    dict
        Mapping of node -> community ID
    """
    if k is None:
        # Estimate a reasonable number of communities if not specified
        k = max(2, min(10, G.number_of_nodes() // 10))
    
    try:
        # Get the adjacency matrix
        A = nx.to_numpy_array(G, weight='weight')
        
        # Compute the normalized Laplacian
        n = A.shape[0]
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        # Compute the k smallest eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        indices = np.argsort(eigenvalues)[1:k+1]  # Skip the first eigenvalue (0)
        k_smallest_eigenvectors = eigenvectors[:, indices]
        
        # Use k-means clustering on the eigenvectors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(k_smallest_eigenvectors)
        
        # Convert to node -> community mapping
        nodes = list(G.nodes())
        community_dict = {nodes[i]: label for i, label in enumerate(kmeans.labels_)}
        
        return community_dict
    except Exception as e:
        logging.warning(f"Spectral clustering failed: {e}, falling back to label propagation")
        return detect_label_propagation_communities(G)

def get_community_metrics(G, communities):
    """
    Calculate metrics for the detected communities.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to analyze
    communities : dict
        Mapping of node -> community ID
        
    Returns:
    --------
    dict
        Dictionary of community metrics
    """
    # Convert communities dict to list of sets
    community_to_nodes = defaultdict(set)
    for node, comm in communities.items():
        community_to_nodes[comm].add(node)
    community_sets = list(community_to_nodes.values())
    
    metrics = {}
    
    # Calculate modularity
    try:
        metrics['modularity'] = nx.community.modularity(G, community_sets)
    except:
        metrics['modularity'] = None
    
    # Calculate number of communities
    metrics['num_communities'] = len(community_to_nodes)
    
    # Calculate average community size
    metrics['avg_community_size'] = np.mean([len(c) for c in community_sets])
    
    # Calculate community size distribution
    size_distribution = [len(c) for c in community_sets]
    metrics['min_community_size'] = min(size_distribution) if size_distribution else 0
    metrics['max_community_size'] = max(size_distribution) if size_distribution else 0
    
    return metrics

def get_node_community_metrics(G, communities):
    """
    Calculate community-related metrics for each node.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to analyze
    communities : dict
        Mapping of node -> community ID
        
    Returns:
    --------
    dict
        Dictionary of node -> metrics
    """
    # Group nodes by community
    community_to_nodes = defaultdict(set)
    for node, comm in communities.items():
        community_to_nodes[comm].add(node)
    
    node_metrics = {}
    
    for node in G.nodes():
        if node not in communities:
            continue
            
        node_comm = communities[node]
        same_comm_nodes = community_to_nodes[node_comm]
        
        # Calculate internal and external connections
        internal_connections = 0
        external_connections = 0
        
        for neighbor in G.neighbors(node):
            if neighbor in communities:
                if communities[neighbor] == node_comm:
                    internal_connections += G[node][neighbor].get('weight', 1.0)
                else:
                    external_connections += G[node][neighbor].get('weight', 1.0)
        
        total_connections = internal_connections + external_connections
        
        metrics = {
            'community': node_comm,
            'community_size': len(same_comm_nodes),
            'internal_connections': internal_connections,
            'external_connections': external_connections,
            'total_connections': total_connections,
            'internal_ratio': internal_connections / total_connections if total_connections > 0 else 0,
            'external_ratio': external_connections / total_connections if total_connections > 0 else 0,
            'is_boundary_node': external_connections > 0
        }
        
        node_metrics[node] = metrics
    
    return node_metrics

def get_community_colors(partition, cmap_name="tab20"):
    """
    Generate colors for communities based on partition.
    
    Parameters:
    -----------
    partition : dict
        Dictionary mapping node IDs to community IDs
    cmap_name : str
        Name of the colormap to use
    
    Returns:
    --------
    dict
        Dictionary mapping community IDs to colors
    """
    import matplotlib.pyplot as plt
    
    communities = set(partition.values())
    cmap = plt.cm.get_cmap(cmap_name, max(20, len(communities)))
    
    return {comm: cmap(i % cmap.N) for i, comm in enumerate(sorted(communities))}

def draw_community_hulls(ax, G, pos, partition, alpha=0.2, linewidth=2):
    """
    Draw convex hulls around communities in the graph.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to draw on
    G : networkx.Graph
        The graph
    pos : dict
        Dictionary mapping node IDs to positions
    partition : dict
        Dictionary mapping node IDs to community IDs
    alpha : float
        Transparency of the hulls
    linewidth : float
        Width of the hull borders
    """
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    from scipy.spatial import ConvexHull
    
    # Group nodes by community
    communities = defaultdict(list)
    for node, comm in partition.items():
        if node in pos:  # Ensure the node has a position
            communities[comm].append(node)
    
    # Generate colors for communities
    community_colors = get_community_colors(partition)
    
    # Draw a convex hull for each community
    for comm, nodes in communities.items():
        if len(nodes) < 3:
            # Need at least 3 points for a convex hull
            continue
            
        # Get node positions for this community
        points = np.array([pos[node] for node in nodes])
        
        # Add some padding around points
        centroid = points.mean(axis=0)
        for i in range(len(points)):
            # Move points slightly away from centroid
            direction = points[i] - centroid
            norm = np.linalg.norm(direction)
            if norm > 0:
                points[i] = points[i] + 0.1 * direction / norm
        
        # Compute the convex hull
        try:
            hull = ConvexHull(points)
            
            # Get the hull vertices in order
            vertices = hull.vertices.tolist()
            vertices.append(vertices[0])  # Close the loop
            
            # Create a path for the hull
            path_vertices = [points[i] for i in vertices]
            path = Path(path_vertices)
            
            # Create a patch from the path
            patch = PathPatch(
                path, 
                facecolor=community_colors.get(comm, (0.8, 0.8, 0.8)), 
                edgecolor=community_colors.get(comm, (0.8, 0.8, 0.8)), 
                alpha=alpha,
                linewidth=linewidth
            )
            
            # Add the patch to the axis
            ax.add_patch(patch)
            
            # Add a label for the community
            centroid = points.mean(axis=0)
            ax.text(
                centroid[0], centroid[1],
                f"Community {comm}",
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'),
                zorder=5
            )
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to draw hull for community {comm}: {e}") 