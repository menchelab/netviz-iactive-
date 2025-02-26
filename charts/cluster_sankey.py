import numpy as np
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
import logging
from collections import defaultdict
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.path import Path

def create_cluster_sankey_chart(ax, visible_links, node_ids, node_clusters, nodes_per_layer, 
                               layers, small_font, medium_font):
    """
    Create a Sankey diagram showing interlayer connections per cluster.
    Left side shows source layers, right side shows target layers,
    with flows representing connections between layers colored by cluster.
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating cluster Sankey chart")
    
    # Clear the axis
    ax.clear()
    ax.set_title('Cluster Connections Between Layers', **medium_font)
    
    # Initialize data structures to track connections
    # Format: {(source_layer_idx, target_layer_idx): {cluster: count}}
    layer_connections = defaultdict(lambda: defaultdict(int))
    
    # Count connections between layers by cluster
    for start_idx, end_idx in visible_links:
        # Skip self-connections
        if start_idx == end_idx:
            continue
            
        # Get layer indices
        start_layer_idx = start_idx // nodes_per_layer
        end_layer_idx = end_idx // nodes_per_layer
        
        # Skip connections within the same layer
        if start_layer_idx == end_layer_idx:
            continue
            
        # Get node IDs
        start_node_id = node_ids[start_idx]
        end_node_id = node_ids[end_idx]
        
        # Get cluster (use source node's cluster)
        cluster = node_clusters.get(start_node_id, "Unknown")
        
        # Add to layer connections
        layer_connections[(start_layer_idx, end_layer_idx)][cluster] += 1
    
    # Check if we have any connections
    if not layer_connections:
        ax.text(0.5, 0.5, 'No interlayer connections to display',
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
        return
    
    # Create a simpler representation for visualization
    # We'll use a direct approach instead of matplotlib's Sankey
    
    # Get unique layers and clusters
    unique_layers = sorted(set([layers[i] for i in range(len(layers))]))
    unique_clusters = set()
    for cluster_dict in layer_connections.values():
        unique_clusters.update(cluster_dict.keys())
    unique_clusters = sorted(unique_clusters)
    
    # Create a colormap for clusters
    colormap = plt.cm.tab20
    cluster_colors = {cluster: colormap(i % 20) for i, cluster in enumerate(unique_clusters)}
    
    try:
        # Create a matrix of connections: source_layer -> target_layer -> cluster -> count
        connection_matrix = {}
        for (src_idx, tgt_idx), cluster_dict in layer_connections.items():
            src_layer = layers[src_idx]
            tgt_layer = layers[tgt_idx]
            
            if src_layer not in connection_matrix:
                connection_matrix[src_layer] = {}
            
            if tgt_layer not in connection_matrix[src_layer]:
                connection_matrix[src_layer][tgt_layer] = {}
            
            for cluster, count in cluster_dict.items():
                connection_matrix[src_layer][tgt_layer][cluster] = count
        
        # Calculate total connections for scaling
        max_connections = 0
        for src_layer, targets in connection_matrix.items():
            for tgt_layer, clusters in targets.items():
                layer_total = sum(clusters.values())
                max_connections = max(max_connections, layer_total)
        
        # Set up the plot
        ax.set_xlim(0, 10)
        ax.set_ylim(0, len(unique_layers) * 2 + 2)  # Add space for legend
        
        # Draw source layers on the left
        src_y_positions = {}
        for i, layer in enumerate(unique_layers):
            y_pos = i * 2 + 1
            src_y_positions[layer] = y_pos
            ax.text(1, y_pos, layer, ha='right', va='center', **small_font)
        
        # Draw target layers on the right
        tgt_y_positions = {}
        for i, layer in enumerate(unique_layers):
            y_pos = i * 2 + 1
            tgt_y_positions[layer] = y_pos
            ax.text(9, y_pos, layer, ha='left', va='center', **small_font)
        
        # Helper function to create a curved path between points
        def curved_path(start_x, start_y, end_x, end_y, width_start, width_end):
            # Control points for the Bezier curve
            control_x = (start_x + end_x) / 2
            
            # Create the path for the top curve
            verts_top = [
                (start_x, start_y),                      # Start point
                (control_x, start_y),                    # Control point 1
                (control_x, end_y),                      # Control point 2
                (end_x, end_y),                          # End point
            ]
            
            # Create the path for the bottom curve
            verts_bottom = [
                (end_x, end_y + width_end),              # End point (bottom)
                (control_x, end_y + width_end),          # Control point 2
                (control_x, start_y + width_start),      # Control point 1
                (start_x, start_y + width_start),        # Start point (bottom)
            ]
            
            # Combine the paths
            verts = verts_top + verts_bottom
            codes = [Path.MOVETO] + [Path.CURVE4] * 3 + [Path.LINETO] + [Path.CURVE4] * 3
            
            return Path(verts, codes)
        
        # Draw connections
        for src_layer, targets in connection_matrix.items():
            src_y = src_y_positions[src_layer]
            
            for tgt_layer, clusters in targets.items():
                tgt_y = tgt_y_positions[tgt_layer]
                
                # Calculate total width for this layer pair
                total_width = sum(clusters.values())
                if total_width == 0:
                    continue
                
                # Scale width for visualization
                scale_factor = 0.8 / max_connections if max_connections > 0 else 0
                
                # Track vertical position for stacking flows
                current_src_offset = -0.4 * (total_width * scale_factor)
                current_tgt_offset = -0.4 * (total_width * scale_factor)
                
                # Draw each cluster's flow
                for cluster, count in clusters.items():
                    if count == 0:
                        continue
                    
                    # Calculate width
                    width = count * scale_factor
                    
                    # Calculate positions
                    src_y1 = src_y + current_src_offset
                    src_y2 = src_y + current_src_offset + width
                    tgt_y1 = tgt_y + current_tgt_offset
                    tgt_y2 = tgt_y + current_tgt_offset + width
                    
                    # Create curved path for the flow
                    path = curved_path(2, src_y1, 8, tgt_y1, width, width)
                    patch = mpatches.PathPatch(
                        path, facecolor=cluster_colors.get(cluster, 'gray'), 
                        alpha=0.7, edgecolor='none'
                    )
                    ax.add_patch(patch)
                    
                    # Add label if flow is large enough
                    if width > 0.1:
                        # Position label at the middle of the curve
                        label_x = 5
                        label_y = (src_y1 + src_y2 + tgt_y1 + tgt_y2) / 4
                        ax.text(label_x, label_y, f"{cluster}: {count}", 
                                ha='center', va='center', 
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                                **small_font)
                    
                    # Update offsets
                    current_src_offset += width
                    current_tgt_offset += width
        
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
        legend_y = 0.02  # Position at the bottom
        ax.legend(
            handles=legend_elements,
            loc='lower center',
            bbox_to_anchor=(0.5, legend_y),
            ncol=min(5, len(unique_clusters)),  # Adjust number of columns based on cluster count
            frameon=False,
            fontsize=small_font['fontsize']
        )
        
        # Remove axes
        ax.axis('off')
        
        logger.info(f"Created custom Sankey diagram with {len(connection_matrix)} source layers and {len(unique_clusters)} clusters")
        
    except Exception as e:
        logger.error(f"Error creating Sankey diagram: {str(e)}")
        ax.clear()
        ax.text(0.5, 0.5, f'Error creating Sankey diagram: {str(e)}',
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off') 