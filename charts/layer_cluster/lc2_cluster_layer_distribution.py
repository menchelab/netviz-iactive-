import numpy as np
import matplotlib.pyplot as plt
import logging
from collections import defaultdict
from ..layer_cluster.utils import get_visible_nodes

def create_cluster_layer_distribution(
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
    Create a stacked bar chart showing the distribution of clusters across layers.
    Each bar represents a layer, and the segments represent different clusters.
    
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
    logger.info("Creating cluster-layer distribution chart")

    # Clear the axis
    ax.clear()
    ax.set_title("Cluster Distribution Across Layers", **medium_font)

    # Filter to only show visible layers if specified
    visible_layers = None
    if visible_layer_indices is not None and len(visible_layer_indices) > 0:
        visible_layers = set(visible_layer_indices)
        logger.info(
            f"Filtering chart to show only {len(visible_layers)} visible layers"
        )

    # Get visible nodes
    visible_node_indices = get_visible_nodes(visible_links)
    logger.info(f"Found {len(visible_node_indices)} visible nodes")

    # Count nodes by cluster and layer
    cluster_layer_counts = defaultdict(lambda: defaultdict(int))
    
    for node_idx in visible_node_indices:
        # Calculate layer index based on node index and nodes per layer
        layer_idx = node_idx // nodes_per_layer
            
        # Skip if layer is not visible
        if visible_layers is not None and layer_idx not in visible_layers:
            continue
            
        node_id = node_ids[node_idx]
        cluster = node_clusters.get(node_id, "Unknown")
        
        cluster_layer_counts[cluster][layer_idx] += 1
        
    logger.info(f"Counted nodes by cluster and layer")

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
        unique_layers = [layers[i] for i in unique_layer_indices if i < len(layers)]
    else:
        unique_layer_indices = list(range(len(layers)))
        unique_layers = layers
        
    unique_clusters = sorted(cluster_layer_counts.keys())
    logger.info(f"Found {len(unique_clusters)} unique clusters and {len(unique_layers)} unique layers")
    
    # Create a colormap for clusters if not provided
    if cluster_colors is None:
        colormap = plt.cm.tab20
        cluster_colors = {cluster: colormap(i % 20) for i, cluster in enumerate(unique_clusters)}

    try:
        # Prepare data for stacked bar chart
        data = np.zeros((len(unique_layers), len(unique_clusters)))
        
        for i, layer_idx in enumerate(unique_layer_indices):
            if layer_idx >= len(layers):
                continue
            layer = layers[layer_idx]
            for j, cluster in enumerate(unique_clusters):
                data[i, j] = cluster_layer_counts[cluster][layer_idx]
        
        # Create the stacked bar chart
        bottom = np.zeros(len(unique_layers))
        
        for i, cluster in enumerate(unique_clusters):
            ax.bar(
                range(len(unique_layers)), 
                data[:, i], 
                bottom=bottom,
                label=cluster,
                color=cluster_colors.get(cluster, "gray"),
                edgecolor="white",
                linewidth=0.5
            )
            bottom += data[:, i]
        
        # Set ticks and labels
        ax.set_xticks(range(len(unique_layers)))
        ax.set_xticklabels(unique_layers, rotation=45, ha="right", **small_font)
        ax.tick_params(axis="y", labelsize=small_font["fontsize"])
        
        # Add legend
        if len(unique_clusters) <= 10:
            ax.legend(fontsize=small_font["fontsize"], loc="upper right")
        else:
            # For many clusters, place legend outside
            ax.legend(
                fontsize=small_font["fontsize"],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=min(5, len(unique_clusters))
            )
        
        ax.set_xlabel("Layers", **medium_font)
        ax.set_ylabel("Number of nodes", **medium_font)
        
        # Add grid lines
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Add value labels on top of each bar
        for i, v in enumerate(bottom):
            if v > 0:
                ax.text(i, v + 1, str(int(v)), ha='center', va='bottom', fontsize=small_font["fontsize"])
        
        # Add a title that clearly explains the visualization
        ax.set_title(f"Cluster Distribution Across {len(unique_layers)} Layers", 
                    fontweight='bold', **medium_font)
        
        logger.info(
            f"Created cluster-layer distribution chart with {len(unique_layers)} layers and {len(unique_clusters)} clusters"
        )
        
    except Exception as e:
        logger.error(f"Error creating cluster-layer distribution chart: {str(e)}")
        ax.clear()
        ax.text(
            0.5,
            0.5,
            f"Error creating chart: {str(e)}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off") 