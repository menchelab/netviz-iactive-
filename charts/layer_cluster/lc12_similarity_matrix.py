import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_cluster_similarity_matrix(
    ax,
    visible_links,
    node_ids,
    node_clusters,
    nodes_per_layer,  # This is now an integer
    layers,
    small_font,
    medium_font,
    visible_layer_indices=None,
    cluster_colors=None,
):
    """
    Create a similarity matrix showing how similar clusters are across all layers.
    Similarity is measured by the Jaccard index of node sets.
    
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
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating cluster similarity matrix with nodes_per_layer={nodes_per_layer}")
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
    ax.set_title("Cluster Similarity Matrix", fontsize=medium_fontsize)

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
        cluster_nodes[cluster].add(base_node_id)
    
    logger.info(f"Collected nodes for {len(cluster_nodes)} clusters")

    # Get unique clusters
    unique_clusters = sorted(cluster_nodes.keys())
    
    # Check if we have any data
    if not unique_clusters:
        logger.warning("No cluster data to display")
        ax.text(
            0.5,
            0.5,
            "No cluster data to display",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off")
        return

    try:
        # Calculate Jaccard similarity between clusters
        n_clusters = len(unique_clusters)
        similarity_matrix = np.zeros((n_clusters, n_clusters))
        
        logger.info(f"Calculating Jaccard similarity for {n_clusters} clusters")
        
        for i, cluster1 in enumerate(unique_clusters):
            for j, cluster2 in enumerate(unique_clusters):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # Self-similarity is 1
                else:
                    # Jaccard similarity: |A ∩ B| / |A ∪ B|
                    intersection = len(cluster_nodes[cluster1].intersection(cluster_nodes[cluster2]))
                    union = len(cluster_nodes[cluster1].union(cluster_nodes[cluster2]))
                    
                    if union > 0:
                        similarity_matrix[i, j] = intersection / union
                    else:
                        similarity_matrix[i, j] = 0.0
        
        logger.info(f"Created similarity matrix of size {similarity_matrix.shape}")
        logger.info(f"Similarity range: min={np.min(similarity_matrix)}, max={np.max(similarity_matrix)}")

        # Create a custom colormap from white to blue
        cmap = LinearSegmentedColormap.from_list('WhiteToBlue', ['#FFFFFF', '#0343DF'])
        
        # Plot the similarity matrix
        im = ax.imshow(similarity_matrix, cmap=cmap, vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Jaccard Similarity', fontsize=small_fontsize)
        cbar.ax.tick_params(labelsize=small_fontsize)
        
        # Add labels
        ax.set_xticks(np.arange(n_clusters))
        ax.set_yticks(np.arange(n_clusters))
        ax.set_xticklabels(unique_clusters, rotation=45, ha='right', fontsize=small_fontsize)
        ax.set_yticklabels(unique_clusters, fontsize=small_fontsize)
        
        # Add grid lines
        ax.set_xticks(np.arange(-.5, n_clusters, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n_clusters, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
        
        # Add text annotations
        for i in range(n_clusters):
            for j in range(n_clusters):
                # Only show significant similarities
                if similarity_matrix[i, j] >= 0.1:
                    text_color = 'white' if similarity_matrix[i, j] > 0.5 else 'black'
                    ax.text(j, i, f"{similarity_matrix[i, j]:.2f}", 
                            ha="center", va="center", color=text_color, fontsize=small_fontsize)
        
        # Add axis labels
        ax.set_xlabel("Clusters", fontsize=small_fontsize)
        ax.set_ylabel("Clusters", fontsize=small_fontsize)
        
        # Add a title with more information
        ax.set_title(f"Cluster Similarity Matrix\n({n_clusters} clusters)", fontsize=medium_fontsize)
        
        logger.info(f"Created similarity matrix with {n_clusters} clusters")
        
    except Exception as e:
        logger.error(f"Error creating similarity matrix: {str(e)}")
        ax.clear()
        ax.text(
            0.5,
            0.5,
            f"Error creating similarity matrix: {str(e)}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off") 