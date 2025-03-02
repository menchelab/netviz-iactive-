import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path

def create_layer_cluster_sankey(
    ax,
    visible_links,
    node_ids,
    node_clusters,
    nodes_per_layer,  # This is now an integer
    layers,
    small_font,
    medium_font,
    visible_layer_indices=None,
):
    """
    Create a Sankey diagram showing the flow of nodes between layers and clusters.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to draw the Sankey diagram on
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
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating layer-cluster Sankey diagram with nodes_per_layer={nodes_per_layer}")
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
    ax.set_title("Layer-Cluster Flow\nConnections between layers and clusters", fontsize=medium_fontsize)

    # Filter to only show visible layers if specified
    visible_layers = None
    if visible_layer_indices is not None and len(visible_layer_indices) > 0:
        visible_layers = set(visible_layer_indices)
        logger.info(
            f"Filtering Sankey to show only {len(visible_layers)} visible layers"
        )

    # Get visible nodes
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
        if visible_layers is not None and layer_idx not in visible_layers:
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

    # Get unique layers and clusters
    if visible_layers is not None:
        unique_layer_indices = sorted(visible_layers)
        unique_layers = [layers[i] for i in unique_layer_indices]
    else:
        unique_layer_indices = list(range(len(layers)))
        unique_layers = layers
        
    unique_clusters = sorted(cluster_layer_counts.keys())
    
    logger.info(f"Found {len(unique_layers)} unique layers and {len(unique_clusters)} unique clusters")
    
    # Create a colormap for clusters
    colormap = plt.cm.tab20
    cluster_colors = {cluster: colormap(i % 20) for i, cluster in enumerate(unique_clusters)}

    try:
        # Prepare data for Sankey diagram
        # For simplicity, we'll create a flow from layers to clusters
        
        # Calculate total nodes per layer and per cluster
        layer_totals = defaultdict(int)
        cluster_totals = defaultdict(int)
        
        for cluster, layer_dict in cluster_layer_counts.items():
            for layer_idx, count in layer_dict.items():
                if visible_layers is not None and layer_idx not in visible_layers:
                    continue
                layer = layers[layer_idx]
                layer_totals[layer] += count
                cluster_totals[cluster] += count
        
        logger.info(f"Layer totals: {dict(layer_totals)}")
        logger.info(f"Cluster totals: {dict(cluster_totals)}")
        
        # Create flows from layers to clusters
        flows = []
        for cluster, layer_dict in cluster_layer_counts.items():
            for layer_idx, count in layer_dict.items():
                if visible_layers is not None and layer_idx not in visible_layers:
                    continue
                layer = layers[layer_idx]
                flows.append((layer, cluster, count))
        
        logger.info(f"Created {len(flows)} flows between layers and clusters")
        
        # Sort layers and clusters by total nodes
        sorted_layers = sorted(layer_totals.keys(), key=lambda x: layer_totals[x], reverse=True)
        sorted_clusters = sorted(cluster_totals.keys(), key=lambda x: cluster_totals[x], reverse=True)
        
        # Limit to top 5 layers and clusters if there are too many
        if len(sorted_layers) > 5:
            sorted_layers = sorted_layers[:5]
            logger.info(f"Limiting Sankey diagram to top 5 layers")
        
        if len(sorted_clusters) > 5:
            sorted_clusters = sorted_clusters[:5]
            logger.info(f"Limiting Sankey diagram to top 5 clusters")
        
        # Filter flows to only include top layers and clusters
        filtered_flows = [
            (layer, cluster, count) 
            for layer, cluster, count in flows 
            if layer in sorted_layers and cluster in sorted_clusters
        ]
        
        logger.info(f"Filtered to {len(filtered_flows)} flows between top layers and clusters")
        
        # Extract flow counts for width calculation
        flow_counts = [count for _, _, count in filtered_flows]
        
        # Draw the Sankey diagram
        ax.clear()
        ax.set_title("Layer-Cluster Flow\nConnections between layers and clusters", fontsize=medium_fontsize)
        
        # Draw the layer blocks (left side)
        layer_y_positions = {}
        layer_heights = {}
        current_y = 0.9
        
        for layer_name, count in layer_totals.items():
            height = 0.8 * (count / max(1, sum(layer_totals.values())))
            rect = plt.Rectangle((0.1, current_y - height), 0.2, height, 
                               facecolor='lightblue', edgecolor='black', alpha=0.8)
            ax.add_patch(rect)
            
            # Add layer label with count
            ax.text(0.2, current_y - height/2, f"{layer_name}\n({count} nodes)", 
                   ha='center', va='center', fontsize=small_fontsize, fontweight='bold')
            
            layer_y_positions[layer_name] = current_y - height/2
            layer_heights[layer_name] = height
            current_y -= height - 0.01  # Small gap between blocks
        
        # Draw the cluster blocks (right side)
        cluster_y_positions = {}
        cluster_heights = {}
        current_y = 0.9
        
        for cluster, count in cluster_totals.items():
            height = 0.8 * (count / max(1, sum(cluster_totals.values())))
            
            # Use cluster color if available
            color = cluster_colors.get(cluster, 'lightgray')
            
            rect = plt.Rectangle((0.7, current_y - height), 0.2, height, 
                               facecolor=color, edgecolor='black', alpha=0.8)
            ax.add_patch(rect)
            
            # Add cluster label with count
            ax.text(0.8, current_y - height/2, f"{cluster}\n({count} nodes)", 
                   ha='center', va='center', fontsize=small_fontsize, fontweight='bold')
            
            cluster_y_positions[cluster] = current_y - height/2
            cluster_heights[cluster] = height
            current_y -= height - 0.01  # Small gap between blocks
        
        # Draw the flows between layers and clusters
        for (layer_name, cluster, count) in filtered_flows:
            if layer_name in layer_y_positions and cluster in cluster_y_positions:
                # Calculate flow width based on count
                flow_width = 0.8 * (count / max(1, max(flow_counts)))
                
                # Calculate control points for the Bezier curve
                start_x = 0.3
                start_y = layer_y_positions[layer_name]
                end_x = 0.7
                end_y = cluster_y_positions[cluster]
                
                # Create a Bezier path
                verts = [
                    (start_x, start_y),
                    (start_x + 0.1, start_y),
                    (end_x - 0.1, end_y),
                    (end_x, end_y)
                ]
                codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                path = Path(verts, codes)
                
                # Get color based on cluster
                color = cluster_colors.get(cluster, 'lightgray')
                alpha = 0.6
                
                # Draw the flow
                patch = mpatches.PathPatch(path, facecolor='none', edgecolor=color, 
                                         linewidth=flow_width*20, alpha=alpha)
                ax.add_patch(patch)
                
                # Add flow count label in the middle of the flow
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2
                
                # Add a small white background to make the text more readable
                ax.text(mid_x, mid_y, f"{count}", ha='center', va='center', 
                       fontsize=small_fontsize, fontweight='bold', color='black',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Add a legend for clusters
        legend_elements = []
        for cluster in unique_clusters[:min(5, len(unique_clusters))]:
            legend_elements.append(
                plt.Line2D([0], [0], 
                          marker='s', 
                          color='w',
                          markerfacecolor=cluster_colors.get(cluster, 'lightgray'),
                          markersize=10,
                          label=f"Cluster {cluster}")
            )
        
        # Add the legend
        ax.legend(
            handles=legend_elements,
            loc='upper center',
            fontsize=small_fontsize,
            title="Clusters",
            title_fontsize=small_fontsize
        )
        
        # Add directional arrow to indicate flow direction
        ax.annotate('', xy=(0.65, 0.95), xytext=(0.35, 0.95),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))
        ax.text(0.5, 0.97, "Flow Direction", ha='center', va='center', fontsize=small_fontsize)
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        logger.info(f"Successfully created Sankey diagram with {len(sorted_layers)} layers and {len(sorted_clusters)} clusters")
        
    except Exception as e:
        logger.error(f"Error creating Sankey diagram: {str(e)}")
        ax.clear()
        ax.text(
            0.5,
            0.5,
            f"Error creating Sankey diagram: {str(e)}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off") 