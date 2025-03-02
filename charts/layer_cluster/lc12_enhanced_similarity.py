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
                    intersection = len(cluster_nodes[cluster1].intersection(cluster_nodes[cluster2]))
                    union = len(cluster_nodes[cluster1].union(cluster_nodes[cluster2]))
                    jaccard_matrix[i, j] = intersection / union if union > 0 else 0.0
        
        # 2. Cosine Similarity based on layer distribution
        # Create feature vectors for each cluster based on layer distribution
        layer_vectors = np.zeros((n_clusters, len(visible_layers)))
        for i, cluster in enumerate(unique_clusters):
            for j, layer_idx in enumerate(sorted(visible_layers)):
                layer_vectors[i, j] = len(cluster_layer_nodes[cluster][layer_idx])
        
        # Normalize vectors
        row_sums = layer_vectors.sum(axis=1, keepdims=True)
        layer_vectors_norm = np.divide(layer_vectors, row_sums, out=np.zeros_like(layer_vectors), where=row_sums!=0)
        
        # Calculate cosine similarity
        cosine_matrix = cosine_similarity(layer_vectors_norm)
        
        # 3. Overlap Coefficient: |A ∩ B| / min(|A|, |B|)
        overlap_matrix = np.zeros((n_clusters, n_clusters))
        for i, cluster1 in enumerate(unique_clusters):
            for j, cluster2 in enumerate(unique_clusters):
                if i == j:
                    overlap_matrix[i, j] = 1.0  # Self-similarity is 1
                else:
                    intersection = len(cluster_nodes[cluster1].intersection(cluster_nodes[cluster2]))
                    min_size = min(len(cluster_nodes[cluster1]), len(cluster_nodes[cluster2]))
                    overlap_matrix[i, j] = intersection / min_size if min_size > 0 else 0.0
        
        # 4. Connection-based Similarity
        # Count connections between clusters
        connection_matrix = np.zeros((n_clusters, n_clusters))
        max_connections = 1  # To avoid division by zero
        
        for start_idx in visible_node_indices:
            if start_idx not in node_connections:
                continue
                
            start_id = node_ids[start_idx]
            start_cluster = node_clusters.get(start_id, "Unknown")
            if start_cluster not in unique_clusters:
                continue
                
            i = unique_clusters.index(start_cluster)
            
            for end_idx in node_connections[start_idx]:
                end_id = node_ids[end_idx]
                end_cluster = node_clusters.get(end_id, "Unknown")
                if end_cluster not in unique_clusters:
                    continue
                    
                j = unique_clusters.index(end_cluster)
                connection_matrix[i, j] += 1
                max_connections = max(max_connections, connection_matrix[i, j])
        
        # Normalize connection matrix
        connection_matrix = connection_matrix / max_connections
        
        # 5. Layer Distribution Similarity
        # Calculate similarity based on layer distribution patterns
        layer_dist_matrix = np.zeros((n_clusters, n_clusters))
        
        for i, cluster1 in enumerate(unique_clusters):
            for j, cluster2 in enumerate(unique_clusters):
                if i == j:
                    layer_dist_matrix[i, j] = 1.0  # Self-similarity is 1
                else:
                    # Calculate similarity based on layer distribution patterns
                    similarity = 0
                    total_layers = 0
                    
                    for layer_idx in sorted(visible_layers):
                        nodes1 = len(cluster_layer_nodes[cluster1][layer_idx])
                        nodes2 = len(cluster_layer_nodes[cluster2][layer_idx])
                        
                        if nodes1 > 0 or nodes2 > 0:
                            total_layers += 1
                            # Calculate ratio of smaller to larger (or 1 if equal)
                            if nodes1 == 0 or nodes2 == 0:
                                ratio = 0  # No overlap in this layer
                            else:
                                ratio = min(nodes1, nodes2) / max(nodes1, nodes2)
                            similarity += ratio
                    
                    layer_dist_matrix[i, j] = similarity / total_layers if total_layers > 0 else 0.0
        
        # 6. Hierarchical Clustering
        # Use Jaccard distance for hierarchical clustering
        jaccard_distance = 1 - jaccard_matrix
        np.fill_diagonal(jaccard_distance, 0)  # Ensure diagonal is 0
        
        # Create a hierarchical clustering
        Z = hierarchy.linkage(squareform(jaccard_distance), method='average')
        
        # 7. Node Sharing Ratio
        # Calculate the ratio of shared nodes to total nodes
        sharing_matrix = np.zeros((n_clusters, n_clusters))
        
        for i, cluster1 in enumerate(unique_clusters):
            for j, cluster2 in enumerate(unique_clusters):
                if i == j:
                    sharing_matrix[i, j] = 1.0  # Self-similarity is 1
                else:
                    intersection = len(cluster_nodes[cluster1].intersection(cluster_nodes[cluster2]))
                    total_nodes = len(cluster_nodes[cluster1]) + len(cluster_nodes[cluster2])
                    sharing_matrix[i, j] = 2 * intersection / total_nodes if total_nodes > 0 else 0.0
        
        # 8. Path-based Similarity
        # Calculate similarity based on shortest paths in the graph
        path_matrix = np.zeros((n_clusters, n_clusters))
        
        # Group nodes by cluster
        cluster_node_indices = defaultdict(list)
        for node_idx in visible_node_indices:
            node_id = node_ids[node_idx]
            cluster = node_clusters.get(node_id, "Unknown")
            if cluster in unique_clusters:
                cluster_node_indices[cluster].append(node_idx)
        
        # Calculate average shortest path length between clusters
        max_path_length = 1  # Default max path length
        
        for i, cluster1 in enumerate(unique_clusters):
            for j, cluster2 in enumerate(unique_clusters):
                if i == j:
                    path_matrix[i, j] = 1.0  # Self-similarity is 1
                else:
                    # Calculate average shortest path length between nodes in different clusters
                    path_lengths = []
                    
                    for node1 in cluster_node_indices[cluster1]:
                        for node2 in cluster_node_indices[cluster2]:
                            try:
                                # Calculate shortest path length
                                path_length = nx.shortest_path_length(G, node1, node2)
                                path_lengths.append(path_length)
                                max_path_length = max(max_path_length, path_length)
                            except nx.NetworkXNoPath:
                                # No path exists
                                pass
                    
                    # Calculate average path length
                    if path_lengths:
                        avg_path_length = sum(path_lengths) / len(path_lengths)
                        # Invert and normalize: shorter paths = higher similarity
                        path_matrix[i, j] = 1 / avg_path_length
                    else:
                        path_matrix[i, j] = 0.0
        
        # Normalize path matrix
        if np.max(path_matrix) > 0:
            path_matrix = path_matrix / np.max(path_matrix)
        
        # 9. Normalized Mutual Information
        # Calculate similarity based on mutual information of layer distributions
        nmi_matrix = np.zeros((n_clusters, n_clusters))
        
        for i, cluster1 in enumerate(unique_clusters):
            for j, cluster2 in enumerate(unique_clusters):
                if i == j:
                    nmi_matrix[i, j] = 1.0  # Self-similarity is 1
                else:
                    # Calculate mutual information based on layer distributions
                    mutual_info = 0
                    entropy1 = 0
                    entropy2 = 0
                    
                    # Get total nodes in each cluster
                    total1 = sum(len(nodes) for nodes in cluster_layer_nodes[cluster1].values())
                    total2 = sum(len(nodes) for nodes in cluster_layer_nodes[cluster2].values())
                    
                    if total1 > 0 and total2 > 0:
                        # Calculate entropies and mutual information
                        for layer_idx in sorted(visible_layers):
                            p1 = len(cluster_layer_nodes[cluster1][layer_idx]) / total1 if total1 > 0 else 0
                            p2 = len(cluster_layer_nodes[cluster2][layer_idx]) / total2 if total2 > 0 else 0
                            
                            # Calculate entropies
                            if p1 > 0:
                                entropy1 -= p1 * np.log2(p1)
                            if p2 > 0:
                                entropy2 -= p2 * np.log2(p2)
                            
                            # Calculate joint probability (simplified)
                            if p1 > 0 and p2 > 0:
                                # Assuming independence for simplicity
                                joint_p = p1 * p2
                                mutual_info += joint_p * np.log2(joint_p / (p1 * p2))
                        
                        # Normalize mutual information
                        if entropy1 > 0 and entropy2 > 0:
                            nmi_matrix[i, j] = 2 * mutual_info / (entropy1 + entropy2)
        
        # Create a custom colormap from white to blue
        cmap = LinearSegmentedColormap.from_list('WhiteToBlue', ['#FFFFFF', '#0343DF'])
        
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