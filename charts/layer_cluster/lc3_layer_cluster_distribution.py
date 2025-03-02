import numpy as np
import matplotlib.pyplot as plt
import logging
from collections import defaultdict
from ..layer_cluster.utils import get_visible_nodes

def create_layer_cluster_distribution(
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
    Create a stacked bar chart showing the distribution of layers across clusters.
    Each bar represents a cluster, and the segments represent different layers.
    This is the inverse of the cluster_layer_distribution chart.
    
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
    logger.info("Creating layer-cluster distribution chart")

    # Clear the axis
    ax.clear()
    ax.set_title("Layer Distribution Across Clusters", **medium_font)

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
    
    # Create a colormap for layers if not provided
    layer_colors = {}
    for i, layer in enumerate(unique_layers):
        layer_colors[layer] = plt.cm.tab10(i % 10)

    try:
        # Calculate total nodes per cluster for sorting
        cluster_totals = defaultdict(int)
        for cluster, layer_dict in cluster_layer_counts.items():
            for layer_idx, count in layer_dict.items():
                if visible_layers is not None and layer_idx not in visible_layers:
                    continue
                cluster_totals[cluster] += count
        
        # Sort clusters by total nodes (descending)
        sorted_clusters = sorted(unique_clusters, key=lambda c: cluster_totals[c], reverse=True)
        
        # Limit to top 15 clusters if there are too many
        if len(sorted_clusters) > 15:
            sorted_clusters = sorted_clusters[:15]
            logger.info(f"Limiting chart to top 15 clusters by node count")
        
        # Prepare data for stacked bar chart
        data = np.zeros((len(unique_layers), len(sorted_clusters)))
        
        for i, layer_idx in enumerate(unique_layer_indices):
            if layer_idx >= len(layers):
                continue
            layer = layers[layer_idx]
            for j, cluster in enumerate(sorted_clusters):
                data[i, j] = cluster_layer_counts[cluster][layer_idx]
        
        # Create the stacked bar chart
        bottom = np.zeros(len(sorted_clusters))
        
        for i, layer_idx in enumerate(unique_layer_indices):
            if layer_idx >= len(layers):
                continue
            layer = layers[layer_idx]
            ax.bar(
                range(len(sorted_clusters)), 
                data[i], 
                bottom=bottom,
                label=layer,
                color=layer_colors.get(layer, "gray"),
                edgecolor="white",
                linewidth=0.5
            )
            bottom += data[i]
        
        # Set ticks and labels
        ax.set_xticks(range(len(sorted_clusters)))
        ax.set_xticklabels(sorted_clusters, rotation=45, ha="right", **small_font)
        ax.tick_params(axis="y", labelsize=small_font["fontsize"])
        
        # Add legend
        if len(unique_layers) <= 10:
            ax.legend(fontsize=small_font["fontsize"], loc="upper right")
        else:
            # For many layers, place legend outside
            ax.legend(
                fontsize=small_font["fontsize"],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=min(5, len(unique_layers))
            )
        
        ax.set_xlabel("Clusters", **medium_font)
        ax.set_ylabel("Number of nodes", **medium_font)
        
        # Add grid lines
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Add value labels on top of each bar
        for i, v in enumerate(bottom):
            if v > 0:
                ax.text(i, v + 1, str(int(v)), ha='center', va='bottom', fontsize=small_font["fontsize"])
        
        # Add a title that clearly explains the visualization
        ax.set_title(f"Layer Distribution Across {len(sorted_clusters)} Clusters", 
                    fontweight='bold', **medium_font)
        
        logger.info(
            f"Created layer-cluster distribution chart with {len(unique_layers)} layers and {len(sorted_clusters)} clusters"
        )
        
    except Exception as e:
        logger.error(f"Error creating layer-cluster distribution chart: {str(e)}")
        ax.clear()
        ax.text(
            0.5,
            0.5,
            f"Error creating chart: {str(e)}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off") 