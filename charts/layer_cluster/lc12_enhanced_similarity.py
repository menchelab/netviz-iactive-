import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy
from scipy.stats import entropy
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, linkage

def create_enhanced_cluster_similarity(
    ax,
    visible_links,
    node_ids,
    node_clusters,
    nodes_per_layer,
    layers,
    small_font,
    medium_font,
    visible_layer_indices=None,
    cluster_colors=None,
    metric="all",
    edge_type="all"
):
    """
    Create an enhanced similarity visualization with 9 different heatmaps showing
    cluster similarities using different metrics and approaches.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to draw the similarity matrix on
    visible_links : list of tuples
        List of (start_idx, end_idx) tuples representing visible links
    node_ids : list
        List of node IDs
    node_clusters : dict
        Dictionary mapping node IDs to cluster labels
    nodes_per_layer : int
        Number of nodes in each layer
    layers : list
        List of layer names
    small_font : dict
        Dictionary with font properties for small text
    medium_font : dict
        Dictionary with font properties for medium text
    visible_layer_indices : list, optional
        List of indices of visible layers
    cluster_colors : dict, optional
        Dictionary mapping cluster labels to colors
    metric : str
        Which similarity metric to use. Options: "all", "jaccard", "cosine", "overlap", 
        "connection", "layer_distribution", "hierarchical", "node_sharing", "path_based", "mutual_information"
    edge_type : str
        Which edge types to consider. Options: "all", "interlayer", "intralayer"
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating enhanced cluster similarity with nodes_per_layer={nodes_per_layer}")
    logger.info(f"Visible links: {len(visible_links)}, Node IDs: {len(node_ids)}, Clusters: {len(node_clusters)}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Visible layer indices: {visible_layer_indices}")
    
    # Handle font parameters correctly
    if isinstance(medium_font, dict):
        medium_fontsize = medium_font.get("fontsize", 12)
    else:
        medium_fontsize = medium_font
        
    if isinstance(small_font, dict):
        small_fontsize = small_font.get("fontsize", 10)
    else:
        small_fontsize = small_font
    
    logger.info(f"Using medium_fontsize={medium_fontsize}, small_fontsize={small_fontsize}")

    # Clear the axis
    ax.clear()
    ax.set_axis_off()  # Turn off the main axis
    
    # Create a figure with 3x3 grid layout
    fig = ax.figure
    fig.suptitle("Enhanced Cluster Similarity Analysis", fontsize=medium_fontsize+2)
    
    # Create 3x3 grid of subplots
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4, bottom=0.05, top=0.85, left=0.1, right=0.9)
    
    # Create 9 subplots for different similarity metrics
    ax1 = fig.add_subplot(gs[0, 0])  # Jaccard Similarity
    ax2 = fig.add_subplot(gs[0, 1])  # Cosine Similarity
    ax3 = fig.add_subplot(gs[0, 2])  # Overlap Coefficient
    ax4 = fig.add_subplot(gs[1, 0])  # Connection-based Similarity
    ax5 = fig.add_subplot(gs[1, 1])  # Layer Distribution Similarity
    ax6 = fig.add_subplot(gs[1, 2])  # Hierarchical Clustering
    ax7 = fig.add_subplot(gs[2, 0])  # Node Sharing Ratio
    ax8 = fig.add_subplot(gs[2, 1])  # Path-based Similarity
    ax9 = fig.add_subplot(gs[2, 2])  # Normalized Mutual Information

    # Filter visible layers if specified
    if visible_layer_indices is not None:
        visible_layers = set(visible_layer_indices)
        logger.info(f"Filtering similarity matrix to show only {len(visible_layers)} visible layers")
    else:
        visible_layers = set(range(len(layers)))
        logger.info("No layer filtering applied")

    # Get visible node indices
    visible_node_indices = set()
    for start_idx, end_idx in visible_links:
        visible_node_indices.add(start_idx)
        visible_node_indices.add(end_idx)
    
    logger.info(f"Found {len(visible_node_indices)} visible nodes")

    # Get nodes by cluster
    cluster_nodes = defaultdict(set)
    cluster_layer_nodes = defaultdict(lambda: defaultdict(set))
    
    # Track node connections for connection-based similarity
    node_connections = defaultdict(set)
    
    # Create a graph for path-based similarity
    G = nx.Graph()
    
    # Track which layers each node appears in
    node_layers = defaultdict(list)
    
    # Create duplicated nodes if using "all" metric or when specifically analyzing edge types
    use_duplicated_nodes = (metric == "all" or edge_type != "all")
    
    if use_duplicated_nodes:
        logging.info(f"Building network with duplicated nodes for metric={metric}, edge_type={edge_type}")
        
        # First, identify which layers each node appears in
        for node_idx in visible_node_indices:
            layer_idx = node_idx // nodes_per_layer
            if layer_idx in visible_layers:
                node_id = node_ids[node_idx]
                # Use base node ID (without layer suffix) to track unique nodes
                base_node_id = node_id.split('_')[0] if '_' in node_id else node_id
                node_layers[base_node_id].append(layer_idx)
        
        # Create duplicated nodes for each layer a node appears in
        duplicated_nodes = {}  # Maps original node ID to list of duplicated node IDs
        
        for base_node_id, layers_list in node_layers.items():
            duplicated_nodes[base_node_id] = []
            
            # Get the cluster for this node
            cluster = node_clusters.get(base_node_id, "Unknown")
            
            for layer_idx in layers_list:
                # Create a new node ID in the format <layer>_<node>
                new_node_id = f"{layer_idx}_{base_node_id}"
                duplicated_nodes[base_node_id].append(new_node_id)
                
                # Add the node to the graph with attributes
                G.add_node(new_node_id, 
                          original_id=base_node_id,
                          cluster=cluster, 
                          layer=layer_idx)
                
                # Add to cluster nodes and cluster-layer nodes
                cluster_nodes[cluster].add(base_node_id)
                cluster_layer_nodes[cluster][layer_idx].add(base_node_id)
        
        logging.info(f"Created {len(G.nodes)} duplicated nodes from {len(node_layers)} original nodes")
        
        # Store original network edges
        original_edges = set()
        for start_idx, end_idx in visible_links:
            if start_idx in visible_node_indices and end_idx in visible_node_indices:
                start_id = node_ids[start_idx]
                end_id = node_ids[end_idx]
                
                # Use base node IDs
                start_base_id = start_id.split('_')[0] if '_' in start_id else start_id
                end_base_id = end_id.split('_')[0] if '_' in end_id else end_id
                
                original_edges.add((start_base_id, end_base_id))
                original_edges.add((end_base_id, start_base_id))  # Add both directions
        
        # Add edges based on edge_type
        intralayer_edges = []
        interlayer_edges = []
        
        # Add intralayer edges if needed
        if edge_type == "all" or edge_type == "intralayer":
            for (source_id, target_id) in original_edges:
                # Find the layer(s) where both nodes exist
                common_layers = set(node_layers[source_id]) & set(node_layers[target_id])
                
                for layer_idx in common_layers:
                    # Create the duplicated node IDs
                    source_node = f"{layer_idx}_{source_id}"
                    target_node = f"{layer_idx}_{target_id}"
                    
                    # Add the intralayer edge
                    G.add_edge(source_node, target_node, edge_type="intralayer")
                    intralayer_edges.append((source_node, target_node))
                    
                    # Add connection to node_connections
                    node_connections[source_node].add(target_node)
                    node_connections[target_node].add(source_node)
        
        # Add interlayer edges if needed
        if edge_type == "all" or edge_type == "interlayer":
            for base_node_id, dup_nodes in duplicated_nodes.items():
                # Connect all duplicated versions of the same node across layers
                for i, source_node in enumerate(dup_nodes):
                    for target_node in dup_nodes[i+1:]:
                        # Add the interlayer edge
                        G.add_edge(source_node, target_node, edge_type="interlayer")
                        interlayer_edges.append((source_node, target_node))
                        
                        # Add connection to node_connections
                        node_connections[source_node].add(target_node)
                        node_connections[target_node].add(source_node)
        
        logging.info(f"Added {len(interlayer_edges)} interlayer edges and {len(intralayer_edges)} intralayer edges")
    
    else:
        # Original approach without duplicated nodes
        # Process nodes and connections
        for node_idx in visible_node_indices:
            # Calculate layer index using integer division
            layer_idx = node_idx // nodes_per_layer
            
            # Skip if layer is not visible
            if layer_idx not in visible_layers:
                continue
                
            node_id = node_ids[node_idx]
            cluster = node_clusters.get(node_id, "Unknown")
            
            # Use base node ID (without layer suffix) to track unique nodes
            base_node_id = node_id.split('_')[0] if '_' in node_id else node_id
            
            # Add to cluster nodes
            cluster_nodes[cluster].add(base_node_id)
            
            # Add to cluster-layer nodes
            cluster_layer_nodes[cluster][layer_idx].add(base_node_id)
            
            # Add node to graph
            G.add_node(node_idx, cluster=cluster, layer=layer_idx)
        
        # Process connections
        for start_idx, end_idx in visible_links:
            # Skip if either node is not in visible layers
            if start_idx not in visible_node_indices or end_idx not in visible_node_indices:
                continue
                
            # Get layer indices
            start_layer = start_idx // nodes_per_layer
            end_layer = end_idx // nodes_per_layer
            
            # Skip if either layer is not visible
            if start_layer not in visible_layers or end_layer not in visible_layers:
                continue
                
            # Get node IDs and clusters
            start_id = node_ids[start_idx]
            end_id = node_ids[end_idx]
            start_cluster = node_clusters.get(start_id, "Unknown")
            end_cluster = node_clusters.get(end_id, "Unknown")
            
            # Add connection to node_connections
            node_connections[start_idx].add(end_idx)
            node_connections[end_idx].add(start_idx)
            
            # Add edge to graph
            G.add_edge(start_idx, end_idx)
    
    # Get unique clusters
    unique_clusters = sorted(cluster_nodes.keys())
    n_clusters = len(unique_clusters)
    
    # Check if we have any data
    if not unique_clusters:
        logger.warning("No cluster data to display")
        for subplot_ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
            subplot_ax.text(
                0.5,
                0.5,
                "No cluster data to display",
                horizontalalignment="center",
                verticalalignment="center",
            )
            subplot_ax.axis("off")
        return

    try:
        # 1. Jaccard Similarity: |A ∩ B| / |A ∪ B|
        jaccard_matrix = np.zeros((n_clusters, n_clusters))
        for i, cluster1 in enumerate(unique_clusters):
            for j, cluster2 in enumerate(unique_clusters):
                if i == j:
                    jaccard_matrix[i, j] = 1.0  # Self-similarity is 1
                else:
                    intersection = len(cluster_nodes[cluster1] & cluster_nodes[cluster2])
                    union = len(cluster_nodes[cluster1] | cluster_nodes[cluster2])
                    jaccard_matrix[i, j] = intersection / union if union > 0 else 0
        
        # 2. Cosine Similarity: A·B / (||A|| ||B||)
        cosine_matrix = np.zeros((n_clusters, n_clusters))
        
        # Create feature vectors for each cluster based on node presence
        all_nodes = set()
        for nodes in cluster_nodes.values():
            all_nodes.update(nodes)
        
        node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Create binary feature vectors
        feature_vectors = {}
        for cluster, nodes in cluster_nodes.items():
            vector = np.zeros(len(all_nodes))
            for node in nodes:
                vector[node_to_idx[node]] = 1
            feature_vectors[cluster] = vector
        
        # Calculate cosine similarity
        for i, cluster1 in enumerate(unique_clusters):
            for j, cluster2 in enumerate(unique_clusters):
                if i == j:
                    cosine_matrix[i, j] = 1.0  # Self-similarity is 1
                else:
                    vec1 = feature_vectors[cluster1]
                    vec2 = feature_vectors[cluster2]
                    dot_product = np.dot(vec1, vec2)
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    cosine_matrix[i, j] = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
        
        # 3. Overlap Coefficient: |A ∩ B| / min(|A|, |B|)
        overlap_matrix = np.zeros((n_clusters, n_clusters))
        for i, cluster1 in enumerate(unique_clusters):
            for j, cluster2 in enumerate(unique_clusters):
                if i == j:
                    overlap_matrix[i, j] = 1.0  # Self-similarity is 1
                else:
                    intersection = len(cluster_nodes[cluster1] & cluster_nodes[cluster2])
                    min_size = min(len(cluster_nodes[cluster1]), len(cluster_nodes[cluster2]))
                    overlap_matrix[i, j] = intersection / min_size if min_size > 0 else 0
        
        # 4. Connection-based Similarity
        connection_matrix = np.zeros((n_clusters, n_clusters))
        
        # Count connections between clusters
        cluster_connections = defaultdict(lambda: defaultdict(int))
        
        # Use the appropriate graph based on whether we're using duplicated nodes
        if use_duplicated_nodes:
            # For duplicated nodes, count connections between clusters using the graph
            for u, v in G.edges():
                u_cluster = G.nodes[u]['cluster']
                v_cluster = G.nodes[v]['cluster']
                cluster_connections[u_cluster][v_cluster] += 1
                cluster_connections[v_cluster][u_cluster] += 1
        else:
            # Original approach
            for node_idx in visible_node_indices:
                if node_idx not in node_connections:
                    continue
                    
                node_id = node_ids[node_idx]
                node_cluster = node_clusters.get(node_id, "Unknown")
                
                for neighbor_idx in node_connections[node_idx]:
                    neighbor_id = node_ids[neighbor_idx]
                    neighbor_cluster = node_clusters.get(neighbor_id, "Unknown")
                    
                    cluster_connections[node_cluster][neighbor_cluster] += 1
        
        # Calculate connection-based similarity
        for i, cluster1 in enumerate(unique_clusters):
            for j, cluster2 in enumerate(unique_clusters):
                if i == j:
                    connection_matrix[i, j] = 1.0  # Self-similarity is 1
                else:
                    # Normalize by the maximum possible connections
                    max_connections = len(cluster_nodes[cluster1]) * len(cluster_nodes[cluster2])
                    connections = cluster_connections[cluster1][cluster2]
                    connection_matrix[i, j] = connections / max_connections if max_connections > 0 else 0
        
        # 5. Layer Distribution Similarity
        layer_dist_matrix = np.zeros((n_clusters, n_clusters))
        
        # Create layer distribution vectors
        layer_vectors = {}
        for cluster in unique_clusters:
            vector = np.zeros(len(visible_layers))
            total_nodes = sum(len(nodes) for layer, nodes in cluster_layer_nodes[cluster].items())
            
            for i, layer_idx in enumerate(sorted(visible_layers)):
                if layer_idx in cluster_layer_nodes[cluster]:
                    vector[i] = len(cluster_layer_nodes[cluster][layer_idx]) / total_nodes if total_nodes > 0 else 0
            
            layer_vectors[cluster] = vector
        
        # Calculate cosine similarity between layer distributions
        for i, cluster1 in enumerate(unique_clusters):
            for j, cluster2 in enumerate(unique_clusters):
                if i == j:
                    layer_dist_matrix[i, j] = 1.0  # Self-similarity is 1
                else:
                    vec1 = layer_vectors[cluster1]
                    vec2 = layer_vectors[cluster2]
                    dot_product = np.dot(vec1, vec2)
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    layer_dist_matrix[i, j] = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
        
        # 6. Hierarchical Clustering
        # Create a distance matrix based on Jaccard distance
        distance_matrix = 1 - jaccard_matrix
        # Convert to condensed form for linkage
        condensed_dist = squareform(distance_matrix)
        # Perform hierarchical clustering
        Z = linkage(condensed_dist, method='average')
        
        # 7. Node Sharing Ratio
        sharing_matrix = np.zeros((n_clusters, n_clusters))
        for i, cluster1 in enumerate(unique_clusters):
            for j, cluster2 in enumerate(unique_clusters):
                if i == j:
                    sharing_matrix[i, j] = 1.0  # Self-similarity is 1
                else:
                    # Calculate the ratio of shared nodes to total nodes
                    shared_nodes = cluster_nodes[cluster1] & cluster_nodes[cluster2]
                    total_nodes = cluster_nodes[cluster1] | cluster_nodes[cluster2]
                    sharing_matrix[i, j] = len(shared_nodes) / len(total_nodes) if total_nodes else 0
        
        # 8. Path-based Similarity
        path_matrix = np.zeros((n_clusters, n_clusters))
        
        # Calculate average shortest path length between clusters
        for i, cluster1 in enumerate(unique_clusters):
            for j, cluster2 in enumerate(unique_clusters):
                if i == j:
                    path_matrix[i, j] = 1.0  # Self-similarity is 1
                else:
                    # Get nodes in each cluster
                    if use_duplicated_nodes:
                        # For duplicated nodes, get nodes by cluster attribute
                        cluster1_nodes = [n for n, attr in G.nodes(data=True) if attr.get('cluster') == cluster1]
                        cluster2_nodes = [n for n, attr in G.nodes(data=True) if attr.get('cluster') == cluster2]
                    else:
                        # Original approach
                        cluster1_nodes = [idx for idx in visible_node_indices 
                                        if node_clusters.get(node_ids[idx], "Unknown") == cluster1]
                        cluster2_nodes = [idx for idx in visible_node_indices 
                                        if node_clusters.get(node_ids[idx], "Unknown") == cluster2]
                    
                    # Calculate shortest paths
                    path_lengths = []
                    for n1 in cluster1_nodes[:10]:  # Limit to 10 nodes for efficiency
                        for n2 in cluster2_nodes[:10]:
                            try:
                                path_length = nx.shortest_path_length(G, n1, n2)
                                path_lengths.append(path_length)
                            except nx.NetworkXNoPath:
                                # No path exists
                                pass
                    
                    # Inverse of average path length (shorter paths = higher similarity)
                    if path_lengths:
                        avg_path_length = sum(path_lengths) / len(path_lengths)
                        path_matrix[i, j] = 1.0 / max(avg_path_length, 1.0)
                    else:
                        path_matrix[i, j] = 0
        
        # 9. Normalized Mutual Information
        nmi_matrix = np.zeros((n_clusters, n_clusters))
        
        # Create cluster assignment vectors
        cluster_assignments = {}
        for cluster in unique_clusters:
            assignments = []
            for node in all_nodes:
                if node in cluster_nodes[cluster]:
                    assignments.append(1)
                else:
                    assignments.append(0)
            cluster_assignments[cluster] = np.array(assignments)
        
        # Calculate NMI
        for i, cluster1 in enumerate(unique_clusters):
            for j, cluster2 in enumerate(unique_clusters):
                if i == j:
                    nmi_matrix[i, j] = 1.0  # Self-similarity is 1
                else:
                    # Calculate entropy for each cluster
                    p1 = cluster_assignments[cluster1]
                    p2 = cluster_assignments[cluster2]
                    
                    # Calculate entropy H(X)
                    p1_counts = np.bincount(p1, minlength=2)
                    p1_probs = p1_counts / len(p1)
                    h1 = entropy(p1_probs, base=2)
                    
                    # Calculate entropy H(Y)
                    p2_counts = np.bincount(p2, minlength=2)
                    p2_probs = p2_counts / len(p2)
                    h2 = entropy(p2_probs, base=2)
                    
                    # Calculate joint entropy H(X,Y)
                    joint_counts = np.zeros((2, 2))
                    for k in range(len(p1)):
                        joint_counts[p1[k], p2[k]] += 1
                    joint_probs = joint_counts / len(p1)
                    joint_probs_flat = joint_probs.flatten()
                    joint_probs_flat = joint_probs_flat[joint_probs_flat > 0]  # Remove zeros
                    h_joint = -np.sum(joint_probs_flat * np.log2(joint_probs_flat))
                    
                    # Calculate mutual information I(X;Y) = H(X) + H(Y) - H(X,Y)
                    mi = h1 + h2 - h_joint
                    
                    # Normalize by max(H(X), H(Y))
                    nmi = mi / max(h1, h2) if max(h1, h2) > 0 else 0
                    nmi_matrix[i, j] = nmi
        
        # Create a colormap for the heatmaps
        cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#f7fbff", "#08306b"])
        
        # Set the title to indicate the network building approach
        network_type = ""
        if use_duplicated_nodes:
            if edge_type == "all":
                network_type = "with Duplicated Nodes (All Edges)"
            elif edge_type == "intralayer":
                network_type = "with Duplicated Nodes (Intralayer Edges Only)"
            elif edge_type == "interlayer":
                network_type = "with Duplicated Nodes (Interlayer Edges Only)"
        
        fig.suptitle(f"Enhanced Cluster Similarity Analysis {network_type}", fontsize=medium_fontsize+2)
        
        # Plot all similarity matrices
        
        # 1. Jaccard Similarity
        plot_similarity_matrix(ax1, jaccard_matrix, unique_clusters, "Jaccard Similarity", 
                              small_fontsize, cmap, show_values=True)
        
        # 2. Cosine Similarity
        plot_similarity_matrix(ax2, cosine_matrix, unique_clusters, "Cosine Similarity", 
                              small_fontsize, cmap, show_values=True)
        
        # 3. Overlap Coefficient
        plot_similarity_matrix(ax3, overlap_matrix, unique_clusters, "Overlap Coefficient", 
                              small_fontsize, cmap, show_values=True)
        
        # 4. Connection-based Similarity
        plot_similarity_matrix(ax4, connection_matrix, unique_clusters, "Connection Similarity", 
                              small_fontsize, cmap, show_values=True)
        
        # 5. Layer Distribution Similarity
        plot_similarity_matrix(ax5, layer_dist_matrix, unique_clusters, "Layer Distribution", 
                              small_fontsize, cmap, show_values=True)
        
        # 6. Hierarchical Clustering
        ax6.set_title("Hierarchical Clustering", fontsize=small_fontsize+1)
        dendrogram = hierarchy.dendrogram(Z, ax=ax6, labels=unique_clusters, 
                                         orientation='right', leaf_font_size=small_fontsize-1)
        ax6.set_xlabel("Distance", fontsize=small_fontsize)
        ax6.set_ylabel("Clusters", fontsize=small_fontsize)
        
        # 7. Node Sharing Ratio
        plot_similarity_matrix(ax7, sharing_matrix, unique_clusters, "Node Sharing Ratio", 
                              small_fontsize, cmap, show_values=True)
        
        # 8. Path-based Similarity
        plot_similarity_matrix(ax8, path_matrix, unique_clusters, "Path-based Similarity", 
                              small_fontsize, cmap, show_values=True)
        
        # 9. Normalized Mutual Information
        plot_similarity_matrix(ax9, nmi_matrix, unique_clusters, "Mutual Information", 
                              small_fontsize, cmap, show_values=True)
        
        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        logger.info(f"Created enhanced similarity visualization with {n_clusters} clusters")
        
    except Exception as e:
        logger.error(f"Error creating enhanced similarity visualization: {str(e)}")
        for subplot_ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
            subplot_ax.clear()
            subplot_ax.text(
                0.5,
                0.5,
                f"Error: {str(e)}",
                horizontalalignment="center",
                verticalalignment="center",
            )
            subplot_ax.axis("off")

def plot_similarity_matrix(ax, matrix, labels, title, fontsize, cmap, show_values=True):
    """Helper function to plot a similarity matrix"""
    # Plot the matrix
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1)
    
    # Add title
    ax.set_title(title, fontsize=fontsize+1)
    
    # Add labels
    n = len(labels)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=fontsize-2)
    ax.set_yticklabels(labels, fontsize=fontsize-2)
    
    # Add grid lines
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
    
    # Add text annotations if requested
    if show_values:
        for i in range(n):
            for j in range(n):
                # Only show significant similarities
                if matrix[i, j] >= 0.1:
                    text_color = 'white' if matrix[i, j] > 0.5 else 'black'
                    ax.text(j, i, f"{matrix[i, j]:.2f}", 
                            ha="center", va="center", color=text_color, fontsize=fontsize-2)
    
    return im 