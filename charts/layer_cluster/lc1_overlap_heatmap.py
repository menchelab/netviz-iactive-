import numpy as np
import matplotlib.pyplot as plt
import logging
from collections import defaultdict
from ..layer_cluster.utils import get_visible_nodes

def create_layer_cluster_overlap_heatmap(
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
):
    """
    Create a heatmap showing the overlap between layers and clusters.
    Each cell represents the number of nodes from a specific cluster in a specific layer.
    
    Args:
        ax: Matplotlib axis to draw on
        visible_links: List of (start_idx, end_idx) tuples representing visible links
        node_ids: List of node IDs
        node_clusters: Dictionary mapping node IDs to cluster IDs
        nodes_per_layer: Integer representing the number of nodes in each layer
        layers: List of layer names
        small_font: Dictionary with font properties for small text
        medium_font: Dictionary with font properties for medium text
        visible_layer_indices: List of indices for visible layers (optional)
        cluster_colors: Dictionary mapping cluster IDs to colors (optional)
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating layer-cluster overlap heatmap")

    # Clear the axis
    ax.clear()
    ax.set_title("Layer-Cluster Overlap", **medium_font)

    # Filter to only show visible layers if specified
    visible_layers = None
    if visible_layer_indices is not None and len(visible_layer_indices) > 0:
        visible_layers = set(visible_layer_indices)
        logger.info(
            f"Filtering heatmap to show only {len(visible_layers)} visible layers"
        )

    # Get visible nodes
    visible_node_indices = get_visible_nodes(visible_links)
    logger.info(f"Found {len(visible_node_indices)} visible nodes")

    # Count nodes by cluster and layer
    cluster_layer_counts = defaultdict(lambda: defaultdict(int))
    
    # Track which clusters are actually visible in the data
    visible_clusters = set()
    
    for node_idx in visible_node_indices:
        # Calculate layer index based on node index and nodes per layer
        layer_idx = node_idx // nodes_per_layer
            
        # Skip if layer is not visible
        if visible_layers is not None and layer_idx not in visible_layers:
            continue
            
        node_id = node_ids[node_idx]
        cluster = node_clusters.get(node_id, "Unknown")
        
        # Add this cluster to our set of visible clusters
        visible_clusters.add(cluster)
        
        cluster_layer_counts[cluster][layer_idx] += 1
        
    logger.info(f"Found {len(visible_clusters)} visible clusters")

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

    # Get unique layers and clusters
    if visible_layers is not None:
        unique_layer_indices = sorted(visible_layers)
        unique_layers = [layers[i] for i in unique_layer_indices]
    else:
        unique_layer_indices = list(range(len(layers)))
        unique_layers = layers
        
    # Sort clusters to ensure consistent ordering, but only include visible ones
    unique_clusters = sorted(visible_clusters)
    logger.info(f"Creating heatmap with {len(unique_clusters)} clusters and {len(unique_layers)} layers")
    
    # Create a colormap for clusters if not provided
    if cluster_colors is None:
        colormap = plt.cm.tab20
        cluster_colors = {cluster: colormap(i % 20) for i, cluster in enumerate(unique_clusters)}

    try:
        # Create a matrix for the heatmap
        overlap_matrix = np.zeros((len(unique_clusters), len(unique_layers)))
        
        # Fill the matrix with node counts
        for i, cluster in enumerate(unique_clusters):
            for j, layer_idx in enumerate(unique_layer_indices):
                overlap_matrix[i, j] = cluster_layer_counts[cluster][layer_idx]
        
        # Create the heatmap
        im = ax.imshow(overlap_matrix, cmap="viridis", interpolation="nearest", aspect="auto")
        
        # Add colorbar - Use the figure that contains the axis
        fig = ax.get_figure()
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Number of nodes", rotation=-90, va="bottom", **small_font)
        cbar.ax.tick_params(labelsize=small_font["fontsize"])
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(unique_layers)))
        ax.set_yticks(np.arange(len(unique_clusters)))
        
        # Make x-axis labels (layers) more readable
        ax.set_xticklabels(unique_layers, rotation=45, ha="right", **small_font)
        
        # Make y-axis labels (clusters) more readable
        ax.set_yticklabels(unique_clusters, **small_font)
        
        # Add grid lines
        ax.set_xticks(np.arange(-.5, len(unique_layers), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(unique_clusters), 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
        
        # Add text annotations in each cell with larger font
        for i in range(len(unique_clusters)):
            for j in range(len(unique_layers)):
                count = overlap_matrix[i, j]
                if count > 0:
                    text_color = "white" if count > overlap_matrix.max() / 2 else "black"
                    # Use a slightly larger font for the counts
                    count_font = {"fontsize": small_font["fontsize"] + 1}
                    ax.text(j, i, int(count), ha="center", va="center", 
                            color=text_color, fontweight='bold', **count_font)
        
        # Add clear axis labels
        ax.set_xlabel("Layers", fontweight='bold', **medium_font)
        ax.set_ylabel("Clusters", fontweight='bold', **medium_font)
        
        # Add a title that clearly explains the visualization
        ax.set_title(f"Node Count by Cluster ({len(unique_clusters)}) and Layer ({len(unique_layers)})", 
                    fontweight='bold', **medium_font)
        
        logger.info(
            f"Created layer-cluster overlap heatmap with {len(unique_layers)} layers and {len(unique_clusters)} clusters"
        )
        
    except Exception as e:
        logger.error(f"Error creating layer-cluster overlap heatmap: {str(e)}")
        ax.clear()
        ax.text(
            0.5,
            0.5,
            f"Error creating heatmap: {str(e)}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off") 