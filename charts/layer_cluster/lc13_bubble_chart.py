import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def create_layer_cluster_bubble_chart(
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
    Create a bubble chart showing the distribution of nodes across layers and clusters.
    Bubble size represents the number of nodes in each layer-cluster combination.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to draw the bubble chart on
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
    logger.info(f"Creating layer-cluster bubble chart with nodes_per_layer={nodes_per_layer}")
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
    ax.set_title("Layer-Cluster Bubble Chart", fontsize=medium_fontsize)

    # Filter visible layers if specified
    if visible_layer_indices is not None:
        visible_layers = set(visible_layer_indices)
        logger.info(f"Filtering bubble chart to show only {len(visible_layers)} visible layers")
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
        
        # Create data for the bubble chart
        x_data = []  # Layer indices
        y_data = []  # Cluster indices
        sizes = []   # Bubble sizes
        colors = []  # Bubble colors
        
        # Map layers and clusters to indices for plotting
        layer_to_idx = {layer_idx: i for i, layer_idx in enumerate(unique_layer_indices)}
        cluster_to_idx = {cluster: i for i, cluster in enumerate(unique_clusters)}
        
        # Fill the data arrays
        for cluster, layer_dict in cluster_layer_counts.items():
            for layer_idx, count in layer_dict.items():
                if count > 0:
                    x_data.append(layer_to_idx[layer_idx])
                    y_data.append(cluster_to_idx[cluster])
                    sizes.append(count)
                    
                    # Use cluster color if available
                    if cluster_colors and cluster in cluster_colors:
                        colors.append(cluster_colors[cluster])
                    else:
                        # Generate a color based on cluster name hash
                        colors.append(plt.cm.tab20(hash(str(cluster)) % 20))
        
        logger.info(f"Created data arrays with {len(x_data)} points")
        
        # Normalize bubble sizes
        min_size = 50
        max_size = 1000
        if sizes:
            min_val = min(sizes)
            max_val = max(sizes)
            logger.info(f"Size range: min={min_val}, max={max_val}")
            
            if min_val != max_val:
                normalized_sizes = [min_size + (max_size - min_size) * (size - min_val) / (max_val - min_val) 
                                  for size in sizes]
            else:
                normalized_sizes = [min_size + (max_size - min_size) / 2 for _ in sizes]
        else:
            normalized_sizes = []
            logger.warning("No sizes data available")
        
        # Create the bubble chart
        scatter = ax.scatter(
            x_data, 
            y_data, 
            s=normalized_sizes, 
            c=colors, 
            alpha=0.7,
            edgecolors='black',
            linewidths=1
        )
        
        # Set axis limits with some padding
        ax.set_xlim(-0.5, len(unique_layers) - 0.5)
        ax.set_ylim(-0.5, len(unique_clusters) - 0.5)
        
        # Set ticks and labels
        ax.set_xticks(range(len(unique_layers)))
        ax.set_yticks(range(len(unique_clusters)))
        ax.set_xticklabels(unique_layers, rotation=45, ha='right', fontsize=small_fontsize)
        ax.set_yticklabels(unique_clusters, fontsize=small_fontsize)
        
        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add text annotations for bubble sizes
        for x, y, size in zip(x_data, y_data, sizes):
            if size > max_val * 0.1:  # Only label significant bubbles
                ax.text(x, y, str(size), 
                        ha='center', va='center', 
                        fontsize=small_fontsize, 
                        fontweight='bold')
        
        # Add axis labels
        ax.set_xlabel("Layers", fontsize=small_fontsize)
        ax.set_ylabel("Clusters", fontsize=small_fontsize)
        
        # Add a title with more information
        ax.set_title(f"Layer-Cluster Bubble Chart\n({len(unique_clusters)} clusters, {len(unique_layers)} layers)", 
                    fontsize=medium_fontsize)
        
        # Add a legend for bubble sizes
        if sizes:
            # Create legend handles for different bubble sizes
            legend_sizes = [min(sizes), (min(sizes) + max(sizes)) / 2, max(sizes)]
            legend_labels = [str(int(size)) for size in legend_sizes]
            
            # Create legend handles
            legend_handles = []
            for size in legend_sizes:
                # Normalize size for legend
                norm_size = min_size + (max_size - min_size) * (size - min_val) / (max_val - min_val) if min_val != max_val else (min_size + max_size) / 2
                handle = plt.Line2D(
                    [0], [0], 
                    marker='o', 
                    color='w', 
                    markerfacecolor='gray',
                    markeredgecolor='black',
                    markersize=np.sqrt(norm_size) / 5,  # Scale down for legend
                    linestyle='none'
                )
                legend_handles.append(handle)
            
            # Add the legend
            ax.legend(
                legend_handles, 
                legend_labels, 
                title="Node Count", 
                loc="upper right", 
                fontsize=small_fontsize,
                title_fontsize=small_fontsize
            )
        
        logger.info(f"Created bubble chart with {len(unique_clusters)} clusters and {len(unique_layers)} layers")
        
    except Exception as e:
        logger.error(f"Error creating bubble chart: {str(e)}")
        ax.clear()
        ax.text(
            0.5,
            0.5,
            f"Error creating bubble chart: {str(e)}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off") 