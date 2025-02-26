import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.path import Path
import logging
from collections import defaultdict

def create_cluster_alluvial(ax, visible_links, node_ids, node_clusters, nodes_per_layer, 
                           layers, small_font, medium_font, visible_layer_indices=None):
    """
    Create an alluvial diagram showing how clusters flow between layers.
    Each cluster is shown as a stream flowing through each layer.
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating cluster alluvial diagram")
    
    # Clear the axis
    ax.clear()
    ax.set_title('Cluster Flow Through Layers', **medium_font)
    
    # Filter to only show visible layers if specified
    visible_layers = None
    if visible_layer_indices is not None and len(visible_layer_indices) > 0:
        visible_layers = set(visible_layer_indices)
        logger.info(f"Filtering alluvial diagram to show only {len(visible_layers)} visible layers")
    
    # Count nodes per cluster per layer
    cluster_layer_counts = defaultdict(lambda: defaultdict(int))
    
    # Get visible nodes
    visible_node_indices = set()
    for start_idx, end_idx in visible_links:
        visible_node_indices.add(start_idx)
        visible_node_indices.add(end_idx)
    
    # Count nodes by cluster and layer
    for node_idx in visible_node_indices:
        layer_idx = node_idx // nodes_per_layer
        
        # Skip if layer is not visible
        if visible_layers is not None and layer_idx not in visible_layers:
            continue
            
        node_id = node_ids[node_idx]
        cluster = node_clusters.get(node_id, "Unknown")
        
        cluster_layer_counts[cluster][layer_idx] += 1
    
    # Check if we have any data
    if not cluster_layer_counts:
        ax.text(0.5, 0.5, 'No cluster data to display',
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
        return
    
    # Get unique layers and clusters
    if visible_layers is not None:
        unique_layer_indices = sorted(visible_layers)
        unique_layers = [layers[i] for i in unique_layer_indices]
    else:
        unique_layer_indices = list(range(len(layers)))
        unique_layers = layers
        
    unique_clusters = sorted(cluster_layer_counts.keys())
    
    # Create a colormap for clusters
    colormap = plt.cm.tab20
    cluster_colors = {cluster: colormap(i % 20) for i, cluster in enumerate(unique_clusters)}
    
    try:
        # Set up the plot
        ax.set_xlim(0, len(unique_layer_indices) - 0.5)
        ax.set_ylim(0, 1)
        
        # Helper function to create a curved path between points
        def curved_path(x1, y1, x2, y2, width1, width2):
            # Control points for the Bezier curve
            control_x = (x1 + x2) / 2
            
            # Create the path for the top curve
            verts_top = [
                (x1, y1),                      # Start point
                (control_x, y1),               # Control point 1
                (control_x, y2),               # Control point 2
                (x2, y2),                      # End point
            ]
            
            # Create the path for the bottom curve
            verts_bottom = [
                (x2, y2 + width2),             # End point (bottom)
                (control_x, y2 + width2),      # Control point 2
                (control_x, y1 + width1),      # Control point 1
                (x1, y1 + width1),             # Start point (bottom)
            ]
            
            # Combine the paths
            verts = verts_top + verts_bottom + [(x1, y1)]  # Close the path
            codes = [Path.MOVETO] + [Path.CURVE4] * 3 + [Path.LINETO] + [Path.CURVE4] * 3 + [Path.CLOSEPOLY]
            
            return Path(verts, codes)
        
        # Calculate the maximum count for any cluster in any layer
        max_count = 0
        for cluster, layer_counts in cluster_layer_counts.items():
            for layer_idx, count in layer_counts.items():
                max_count = max(max_count, count)
        
        # Calculate the vertical positions for each cluster in each layer
        positions = {}  # {layer_idx: {cluster: (y_start, height)}}
        
        for layer_idx in unique_layer_indices:
            positions[layer_idx] = {}
            
            # Get all clusters in this layer
            layer_clusters = []
            for cluster in unique_clusters:
                count = cluster_layer_counts[cluster].get(layer_idx, 0)
                if count > 0:
                    layer_clusters.append((cluster, count))
            
            # Sort clusters by count (largest first)
            layer_clusters.sort(key=lambda x: x[1], reverse=True)
            
            # Calculate positions
            y_pos = 0
            for cluster, count in layer_clusters:
                height = count / max_count * 0.8 if max_count > 0 else 0
                positions[layer_idx][cluster] = (y_pos, height)
                y_pos += height + 0.02  # Add a small gap between clusters
        
        # Draw the alluvial diagram
        for i in range(len(unique_layer_indices) - 1):
            src_layer_idx = unique_layer_indices[i]
            tgt_layer_idx = unique_layer_indices[i + 1]
            
            for cluster in unique_clusters:
                # Check if cluster exists in both layers
                if cluster in positions[src_layer_idx] and cluster in positions[tgt_layer_idx]:
                    src_y, src_height = positions[src_layer_idx][cluster]
                    tgt_y, tgt_height = positions[tgt_layer_idx][cluster]
                    
                    # Create curved path for the flow
                    path = curved_path(i, src_y, i+1, tgt_y, src_height, tgt_height)
                    patch = mpatches.PathPatch(
                        path, facecolor=cluster_colors.get(cluster, 'gray'), 
                        alpha=0.7, edgecolor='none'
                    )
                    ax.add_patch(patch)
                    
                    # Add label if flow is large enough
                    if src_height > 0.05 and tgt_height > 0.05:
                        # Position label at the middle of the curve
                        label_x = i + 0.5
                        label_y = (src_y + src_height/2 + tgt_y + tgt_height/2) / 2
                        ax.text(label_x, label_y, cluster, 
                               ha='center', va='center', 
                               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                               **small_font)
        
        # Add layer labels
        for i, layer in enumerate(unique_layers):
            ax.text(i, 1.02, layer, ha='center', va='bottom', rotation=45, **small_font)
        
        # Add legend for clusters
        legend_elements = []
        for cluster in unique_clusters:
            legend_elements.append(
                mpatches.Patch(
                    facecolor=cluster_colors.get(cluster, 'gray'),
                    alpha=0.7,
                    label=cluster
                )
            )
        
        # Position the legend at the bottom
        ax.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.1),
            ncol=min(4, len(unique_clusters)),
            frameon=False,
            fontsize=small_font['fontsize']
        )
        
        # Remove axes
        ax.axis('off')
        
        logger.info(f"Created alluvial diagram with {len(unique_layers)} layers and {len(unique_clusters)} clusters")
        
    except Exception as e:
        logger.error(f"Error creating alluvial diagram: {str(e)}")
        ax.clear()
        ax.text(0.5, 0.5, f'Error creating alluvial diagram: {str(e)}',
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off') 