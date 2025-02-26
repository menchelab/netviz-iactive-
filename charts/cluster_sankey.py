import numpy as np
import matplotlib.pyplot as plt
import logging
from collections import defaultdict
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.path import Path

def create_cluster_sankey_chart(ax, visible_links, node_ids, node_clusters, nodes_per_layer, 
                               layers, small_font, medium_font, visible_layer_indices=None):
    """
    Create a Sankey diagram showing interlayer connections per cluster.
    Left side shows source layers, right side shows target layers,
    with flows representing connections between layers colored by cluster.
    
    Only shows connections between visible layers if visible_layer_indices is provided.
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating cluster Sankey chart")
    
    # Clear the axis
    ax.clear()
    ax.set_title('Cluster Connections Between Layers', **medium_font)
    
    # Filter to only show visible layers if specified
    visible_layers = None
    if visible_layer_indices is not None and len(visible_layer_indices) > 0:
        visible_layers = set(visible_layer_indices)
        logger.info(f"Filtering Sankey diagram to show only {len(visible_layers)} visible layers")
    
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
            
        # Skip connections involving hidden layers
        if visible_layers is not None and (start_layer_idx not in visible_layers or end_layer_idx not in visible_layers):
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
    
    # Get unique layers and clusters
    if visible_layers is not None:
        # Only include visible layers
        unique_layers = sorted([layers[i] for i in visible_layers])
    else:
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
        
        # Calculate total outgoing and incoming connections per layer
        outgoing_connections = defaultdict(int)
        incoming_connections = defaultdict(int)
        
        for src_layer, targets in connection_matrix.items():
            for tgt_layer, clusters in targets.items():
                total = sum(clusters.values())
                outgoing_connections[src_layer] += total
                incoming_connections[tgt_layer] += total
        
        # Calculate maximum connections for scaling
        max_connections = max(
            max(outgoing_connections.values(), default=0),
            max(incoming_connections.values(), default=0)
        )
        
        # Set up the plot
        ax.set_xlim(0, 10)
        ax.set_ylim(0, len(unique_layers) * 3 + 2)  # More space between layers + legend
        
        # Calculate vertical space needed for each layer
        layer_height = 2.0
        layer_spacing = 1.0
        
        # Calculate the scale factor for connection heights
        # Use a fixed unit height per connection to ensure consistent stacking
        unit_height = 0.05  # Height per connection
        
        # Draw source layers on the left with proper spacing
        src_y_positions = {}
        src_y_start_positions = {}  # Starting y-position for connections
        
        for i, layer in enumerate(unique_layers):
            y_pos = i * (layer_height + layer_spacing) + layer_height/2
            src_y_positions[layer] = y_pos
            
            # Calculate the total height needed for connections
            total_outgoing = outgoing_connections[layer]
            connection_height = total_outgoing * unit_height
            
            # Center the connections around the layer position
            src_y_start_positions[layer] = y_pos - connection_height/2
            
            ax.text(1, y_pos, f"{layer} ({total_outgoing})", ha='right', va='center', **small_font)
        
        # Draw target layers on the right with proper spacing
        tgt_y_positions = {}
        tgt_y_start_positions = {}  # Starting y-position for connections
        
        for i, layer in enumerate(unique_layers):
            y_pos = i * (layer_height + layer_spacing) + layer_height/2
            tgt_y_positions[layer] = y_pos
            
            # Calculate the total height needed for connections
            total_incoming = incoming_connections[layer]
            connection_height = total_incoming * unit_height
            
            # Center the connections around the layer position
            tgt_y_start_positions[layer] = y_pos - connection_height/2
            
            ax.text(9, y_pos, f"{layer} ({total_incoming})", ha='left', va='center', **small_font)
        
        # Helper function to create a curved path between points
        def curved_path(start_x, start_y, end_x, end_y, height):
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
                (end_x, end_y + height),                 # End point (bottom)
                (control_x, end_y + height),             # Control point 2
                (control_x, start_y + height),           # Control point 1
                (start_x, start_y + height),             # Start point (bottom)
            ]
            
            # Combine the paths
            verts = verts_top + verts_bottom + [(start_x, start_y)]  # Close the path
            codes = [Path.MOVETO] + [Path.CURVE4] * 3 + [Path.LINETO] + [Path.CURVE4] * 3 + [Path.CLOSEPOLY]
            
            return Path(verts, codes)
        
        # Track the current position for each layer's connections
        src_current_positions = {layer: pos for layer, pos in src_y_start_positions.items()}
        tgt_current_positions = {layer: pos for layer, pos in tgt_y_start_positions.items()}
        
        # Draw connections
        for src_layer, targets in connection_matrix.items():
            for tgt_layer, clusters in targets.items():
                # Draw each cluster's flow
                for cluster, count in sorted(clusters.items(), key=lambda x: x[0]):
                    if count == 0:
                        continue
                    
                    # Calculate the height for this flow
                    flow_height = count * unit_height
                    
                    # Get the current positions
                    src_y = src_current_positions[src_layer]
                    tgt_y = tgt_current_positions[tgt_layer]
                    
                    # Create curved path for the flow
                    path = curved_path(2, src_y, 8, tgt_y, flow_height)
                    patch = mpatches.PathPatch(
                        path, facecolor=cluster_colors.get(cluster, 'gray'), 
                        alpha=0.7, edgecolor='none'
                    )
                    ax.add_patch(patch)
                    
                    # Add label if flow is large enough
                    if flow_height > 0.1:
                        # Position label at the middle of the curve
                        label_x = 5
                        label_y = (src_y + flow_height/2 + tgt_y + flow_height/2) / 2
                        ax.text(label_x, label_y, f"{cluster}: {count}", 
                                ha='center', va='center', 
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                                **small_font)
                    
                    # Update the current positions
                    src_current_positions[src_layer] += flow_height
                    tgt_current_positions[tgt_layer] += flow_height
        
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