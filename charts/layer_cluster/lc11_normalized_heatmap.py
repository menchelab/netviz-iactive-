import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_layer_cluster_normalized_heatmap(
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
    Create a normalized heatmap showing the distribution of clusters across layers.
    Each row (cluster) is normalized to sum to 1, showing the relative distribution.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to draw the heatmap on
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
    logger.info(f"Creating normalized layer-cluster heatmap with nodes_per_layer={nodes_per_layer}")
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
    ax.set_title("Normalized Layer-Cluster Distribution", fontsize=medium_fontsize)

    # Filter visible layers if specified
    if visible_layer_indices is not None:
        visible_layers = set(visible_layer_indices)
        logger.info(f"Filtering normalized heatmap to show only {len(visible_layers)} visible layers")
    else:
        visible_layers = set(range(len(layers)))
        logger.info("No layer filtering applied")

    # Get visible node indices
    visible_node_indices = set()
    for start_idx, end_idx in visible_links:
        visible_node_indices.add(start_idx)
        visible_node_indices.add(end_idx)
    
    logger.info(f"Found {len(visible_node_indices)} visible nodes")

    # Count nodes by cluster and layer
    cluster_layer_counts = defaultdict(lambda: defaultdict(int))
    
    for node_idx in visible_node_indices:
        # Calculate layer index using integer division
        layer_idx = node_idx // nodes_per_layer
        
        # Skip if layer is not visible
        if layer_idx not in visible_layers:
            continue
            
        node_id = node_ids[node_idx]
        cluster = node_clusters.get(node_id, "Unknown")
        
        cluster_layer_counts[cluster][layer_idx] += 1
    
    logger.info(f"Counted nodes by cluster and layer: {dict(cluster_layer_counts)}")

    # Check if we have any data
    if not cluster_layer_counts:
        logger.warning("No cluster-layer data to display")
        ax.text(
            0.5,
            0.5,
            "No cluster-layer data to display",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off")
        return

    try:
        # Get unique layers and clusters
        unique_clusters = sorted(cluster_layer_counts.keys())
        unique_layer_indices = sorted(visible_layers)
        unique_layers = [layers[i] for i in unique_layer_indices]
        
        logger.info(f"Found {len(unique_clusters)} unique clusters and {len(unique_layers)} unique layers")
        
        # Create a matrix for the heatmap
        count_matrix = np.zeros((len(unique_clusters), len(unique_layers)))
        
        # Fill the matrix with node counts
        for i, cluster in enumerate(unique_clusters):
            for j, layer_idx in enumerate(unique_layer_indices):
                count_matrix[i, j] = cluster_layer_counts[cluster][layer_idx]
        
        logger.info(f"Created count matrix with shape {count_matrix.shape}")
        
        # Normalize each row (cluster) to sum to 1
        normalized_matrix = np.zeros_like(count_matrix, dtype=float)
        for i in range(len(unique_clusters)):
            row_sum = np.sum(count_matrix[i, :])
            if row_sum > 0:
                normalized_matrix[i, :] = count_matrix[i, :] / row_sum
        
        logger.info(f"Created normalized matrix with shape {normalized_matrix.shape}")
        
        # Create a custom colormap from white to green
        cmap = LinearSegmentedColormap.from_list('WhiteToGreen', ['#FFFFFF', '#03DF43'])
        
        # Plot the heatmap
        im = ax.imshow(normalized_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Proportion of cluster nodes in layer', fontsize=small_fontsize)
        cbar.ax.tick_params(labelsize=small_fontsize)
        
        # Add labels
        ax.set_xticks(np.arange(len(unique_layers)))
        ax.set_yticks(np.arange(len(unique_clusters)))
        ax.set_xticklabels(unique_layers, rotation=45, ha='right', fontsize=small_fontsize)
        ax.set_yticklabels(unique_clusters, fontsize=small_fontsize)
        
        # Add grid lines
        ax.set_xticks(np.arange(-.5, len(unique_layers), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(unique_clusters), 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
        
        # Add text annotations for percentages
        for i in range(len(unique_clusters)):
            for j in range(len(unique_layers)):
                value = normalized_matrix[i, j]
                if value > 0:
                    # Show as percentage
                    text = f"{value:.0%}"
                    # Use white text for dark cells, black for light cells
                    text_color = 'white' if value > 0.5 else 'black'
                    ax.text(j, i, text, 
                            ha="center", va="center", 
                            color=text_color, fontsize=small_fontsize)
        
        # Add axis labels
        ax.set_xlabel("Layers", fontsize=small_fontsize)
        ax.set_ylabel("Clusters", fontsize=small_fontsize)
        
        # Add a title with more information
        ax.set_title(f"Normalized Layer-Cluster Distribution\n({len(unique_clusters)} clusters, {len(unique_layers)} layers)", 
                    fontsize=medium_fontsize)
        
        logger.info(f"Created normalized heatmap with {len(unique_clusters)} clusters and {len(unique_layers)} layers")
        
    except Exception as e:
        logger.error(f"Error creating normalized heatmap: {str(e)}")
        ax.clear()
        ax.text(
            0.5,
            0.5,
            f"Error creating normalized heatmap: {str(e)}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off") 