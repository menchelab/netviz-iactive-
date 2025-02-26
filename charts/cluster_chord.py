import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.path import Path
import logging
from collections import defaultdict
import math

def create_cluster_chord_diagram(ax, visible_links, node_ids, node_clusters, nodes_per_layer, 
                                layers, small_font, medium_font, visible_layer_indices=None):
    """
    Create a chord diagram showing interlayer connections per cluster.
    Layers are arranged in a circle, with arcs connecting them colored by cluster.
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating cluster chord diagram")
    
    # Clear the axis
    ax.clear()
    ax.set_title('Cluster Connections Between Layers', **medium_font)
    
    # Filter to only show visible layers if specified
    visible_layers = None
    if visible_layer_indices is not None and len(visible_layer_indices) > 0:
        visible_layers = set(visible_layer_indices)
        logger.info(f"Filtering chord diagram to show only {len(visible_layers)} visible layers")
    
    # Initialize data structures to track connections
    layer_connections = defaultdict(lambda: defaultdict(int))
    
    # Count connections between layers by cluster
    for start_idx, end_idx in visible_links:
        if start_idx == end_idx:
            continue
            
        start_layer_idx = start_idx // nodes_per_layer
        end_layer_idx = end_idx // nodes_per_layer
        
        if start_layer_idx == end_layer_idx:
            continue
            
        if visible_layers is not None and (start_layer_idx not in visible_layers or end_layer_idx not in visible_layers):
            continue
            
        start_node_id = node_ids[start_idx]
        cluster = node_clusters.get(start_node_id, "Unknown")
        
        layer_connections[(start_layer_idx, end_layer_idx)][cluster] += 1
    
    # Check if we have any connections
    if not layer_connections:
        ax.text(0.5, 0.5, 'No interlayer connections to display',
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
        return
    
    # Get unique layers and clusters
    if visible_layers is not None:
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
        # Create a matrix of connections
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
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        
        # Calculate positions for layers around the circle
        radius = 1.0
        layer_positions = {}
        layer_angles = {}
        
        for i, layer in enumerate(unique_layers):
            angle = 2 * math.pi * i / len(unique_layers)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            layer_positions[layer] = (x, y)
            layer_angles[layer] = angle
            
            # Draw layer label
            label_x = 1.1 * x
            label_y = 1.1 * y
            ax.text(label_x, label_y, layer, ha='center', va='center', 
                   rotation=angle * 180 / math.pi - 90 if -math.pi/2 <= angle <= math.pi/2 else angle * 180 / math.pi + 90,
                   **small_font)
        
        # Helper function to create a curved arc between points
        def curved_arc(start_angle, end_angle, inner_radius, outer_radius, cluster_color):
            # Number of points to create a smooth arc
            n_points = 50
            
            # Create the inner arc
            inner_thetas = np.linspace(start_angle, end_angle, n_points)
            inner_xs = inner_radius * np.cos(inner_thetas)
            inner_ys = inner_radius * np.sin(inner_thetas)
            
            # Create the outer arc (in reverse)
            outer_thetas = np.linspace(end_angle, start_angle, n_points)
            outer_xs = outer_radius * np.cos(outer_thetas)
            outer_ys = outer_radius * np.sin(outer_thetas)
            
            # Combine to form a closed path
            xs = np.concatenate([inner_xs, outer_xs, [inner_xs[0]]])
            ys = np.concatenate([inner_ys, outer_ys, [inner_ys[0]]])
            
            # Create the patch
            patch = mpatches.Polygon(np.column_stack([xs, ys]), closed=True, 
                                    facecolor=cluster_color, alpha=0.7, edgecolor='none')
            return patch
        
        # Draw connections
        for src_layer, targets in connection_matrix.items():
            src_angle = layer_angles[src_layer]
            
            for tgt_layer, clusters in targets.items():
                tgt_angle = layer_angles[tgt_layer]
                
                # Calculate total connections for this layer pair
                total_connections = sum(clusters.values())
                
                # Skip if no connections
                if total_connections == 0:
                    continue
                
                # Calculate the width of the arc at the base
                base_width = 0.1 * (total_connections / max_connections) if max_connections > 0 else 0
                
                # Calculate the starting positions for each cluster's arc
                src_start = src_angle - base_width/2
                tgt_start = tgt_angle - base_width/2
                
                # Draw each cluster's arc
                for cluster, count in sorted(clusters.items(), key=lambda x: x[0]):
                    if count == 0:
                        continue
                    
                    # Calculate the proportion of this cluster
                    proportion = count / total_connections
                    
                    # Calculate the width for this cluster's arc
                    cluster_width = base_width * proportion
                    
                    # Calculate the angles for this cluster's arc
                    src_end = src_start + cluster_width
                    tgt_end = tgt_start + cluster_width
                    
                    # Create the arc
                    inner_radius = 0.9
                    outer_radius = 1.0
                    
                    # Determine the direction of the arc
                    arc_src_angle = src_angle
                    arc_tgt_angle = tgt_angle
                    if abs(tgt_angle - src_angle) > math.pi:
                        # Take the shorter path around the circle
                        if src_angle < tgt_angle:
                            arc_src_angle += 2 * math.pi
                        else:
                            arc_tgt_angle += 2 * math.pi
                    
                    # Create and add the patch
                    patch = curved_arc(src_start, src_end, inner_radius, outer_radius, 
                                      cluster_colors.get(cluster, 'gray'))
                    ax.add_patch(patch)
                    
                    # Draw the connection between the arcs
                    control_radius = 0.5  # Radius for the control points
                    src_mid = (src_start + src_end) / 2
                    tgt_mid = (tgt_start + tgt_end) / 2
                    
                    # Calculate points for the Bezier curve
                    src_x, src_y = inner_radius * math.cos(src_mid), inner_radius * math.sin(src_mid)
                    tgt_x, tgt_y = inner_radius * math.cos(tgt_mid), inner_radius * math.sin(tgt_mid)
                    
                    # Control points
                    control_x = control_radius * math.cos((src_mid + tgt_mid) / 2)
                    control_y = control_radius * math.sin((src_mid + tgt_mid) / 2)
                    
                    # Create the path
                    verts = [
                        (src_x, src_y),
                        (control_x, control_y),
                        (tgt_x, tgt_y)
                    ]
                    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                    path = Path(verts, codes)
                    
                    # Add the path with the cluster color
                    width = max(1, 5 * (count / max_connections))  # Scale width by connection count
                    ax.add_patch(mpatches.PathPatch(path, facecolor='none', 
                                                   edgecolor=cluster_colors.get(cluster, 'gray'),
                                                   linewidth=width, alpha=0.7))
                    
                    # Add label if arc is large enough
                    if cluster_width > 0.05:
                        # Position label at the middle of the arc
                        mid_angle = (src_start + src_end) / 2
                        mid_radius = (inner_radius + outer_radius) / 2
                        label_x = mid_radius * math.cos(mid_angle)
                        label_y = mid_radius * math.sin(mid_angle)
                        
                        ax.text(label_x, label_y, f"{cluster}: {count}", 
                               ha='center', va='center', 
                               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                               **small_font)
                    
                    # Update the starting positions
                    src_start = src_end
                    tgt_start = tgt_end
        
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
            loc='center',
            bbox_to_anchor=(0.5, 0),
            ncol=min(4, len(unique_clusters)),
            frameon=False,
            fontsize=small_font['fontsize']
        )
        
        # Remove axes
        ax.axis('off')
        
        logger.info(f"Created chord diagram with {len(connection_matrix)} layers and {len(unique_clusters)} clusters")
        
    except Exception as e:
        logger.error(f"Error creating chord diagram: {str(e)}")
        ax.clear()
        ax.text(0.5, 0.5, f'Error creating chord diagram: {str(e)}',
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off') 