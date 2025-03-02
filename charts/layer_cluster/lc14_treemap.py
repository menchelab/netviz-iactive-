import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import squarify
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib.colors as mcolors

def create_layer_cluster_treemap(
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
    count_type="nodes",  # New parameter: 'nodes' or 'intralayer_edges'
):
    """
    Create a treemap visualization showing the distribution of nodes or intralayer edges across layers and clusters.
    Rectangle size represents the number of nodes or intralayer edges in each layer-cluster combination.
    
    Parameters:
    -----------
    count_type : str
        Type of counting to perform: 'nodes' (default) or 'intralayer_edges'
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating layer-cluster treemap with nodes_per_layer={nodes_per_layer}, count_type={count_type}")
    
    # Handle font parameters correctly
    if isinstance(medium_font, dict):
        medium_fontsize = medium_font.get("fontsize", 12)
    else:
        medium_fontsize = medium_font
        
    if isinstance(small_font, dict):
        small_fontsize = small_font.get("fontsize", 10)
    else:
        small_fontsize = small_font
    
    try:
        # Clear the axis
        ax.clear()
        
        # Set title based on count type
        if count_type == "nodes":
            ax.set_title("Layer-Cluster Node Distribution Treemap", fontsize=medium_fontsize)
        else:
            ax.set_title("Layer-Cluster Intralayer Edge Distribution Treemap", fontsize=medium_fontsize)
        
        # Filter to only visible layers
        visible_layer_indices = visible_layer_indices or list(range(len(layers)))
        
        # Create a mapping of node_id to layer
        node_to_layer = {}
        for layer_idx in visible_layer_indices:
            if layer_idx < len(layers):
                # Handle different formats of nodes_per_layer
                if isinstance(nodes_per_layer, dict):
                    # If nodes_per_layer is a dictionary mapping layer_idx -> list of nodes
                    for node_id in nodes_per_layer.get(layer_idx, []):
                        if node_id in node_ids:
                            node_to_layer[node_id] = layer_idx
                elif isinstance(nodes_per_layer, int):
                    # If nodes_per_layer is an integer (number of nodes per layer)
                    for i, node_id in enumerate(node_ids):
                        node_layer = i // nodes_per_layer
                        if node_layer == layer_idx:
                            node_to_layer[node_id] = layer_idx
        
        cluster_layer_counts = {}
        
        if count_type == "nodes":
            # Count nodes by cluster and layer
            for node_id, cluster in node_clusters.items():
                if node_id in node_ids and node_id in node_to_layer:
                    layer_idx = node_to_layer[node_id]
                    if cluster not in cluster_layer_counts:
                        cluster_layer_counts[cluster] = {}
                    if layer_idx not in cluster_layer_counts[cluster]:
                        cluster_layer_counts[cluster][layer_idx] = 0
                    cluster_layer_counts[cluster][layer_idx] += 1
        else:
            # Count intralayer edges by cluster and layer
            # First, create a mapping of node_id to cluster
            node_to_cluster = {node_id: cluster for node_id, cluster in node_clusters.items() if node_id in node_ids}
            
            # Count intralayer edges
            for source_id, target_id in visible_links:
                # Check if both nodes are in the same layer
                if (source_id in node_to_layer and 
                    target_id in node_to_layer and 
                    node_to_layer[source_id] == node_to_layer[target_id]):
                    
                    layer_idx = node_to_layer[source_id]
                    
                    # Check if both nodes are in the same cluster
                    if (source_id in node_to_cluster and 
                        target_id in node_to_cluster and 
                        node_to_cluster[source_id] == node_to_cluster[target_id]):
                        
                        cluster = node_to_cluster[source_id]
                        
                        # Increment the count for this cluster-layer combination
                        if cluster not in cluster_layer_counts:
                            cluster_layer_counts[cluster] = {}
                        if layer_idx not in cluster_layer_counts[cluster]:
                            cluster_layer_counts[cluster][layer_idx] = 0
                        cluster_layer_counts[cluster][layer_idx] += 1
        
        # Check if we have any data
        if not cluster_layer_counts:
            ax.text(0.5, 0.5, "No data to display", ha="center", va="center")
            ax.axis("off")
            return
        
        # Prepare data for treemap
        labels = []
        sizes = []
        colors = []
        
        # Create a colormap for layers
        layer_cmap = plt.cm.viridis
        num_layers = len(layers)
        
        # Create a list of cluster colors
        cluster_colors_list = []
        for cluster in sorted(set(node_clusters.values())):
            # Default to a gray color if cluster color not provided
            cluster_colors_list.append(cluster_colors.get(cluster, 'gray'))
        
        # Create rectangles for each cluster-layer combination
        for cluster, layer_dict in sorted(cluster_layer_counts.items()):
            for layer_idx, count in sorted(layer_dict.items()):
                if layer_idx < len(layers):
                    layer_name = layers[layer_idx]
                    # Create label with appropriate count description
                    if count_type == "nodes":
                        label = f"C{cluster}-{layer_name}\n({count} nodes)"
                    else:
                        label = f"C{cluster}-{layer_name}\n({count} edges)"
                    labels.append(label)
                    
                    # Add size (must be float)
                    sizes.append(float(count))
                    
                    # Create blended color between cluster and layer
                    cluster_color = cluster_colors.get(cluster, 'gray')
                    layer_color = layer_cmap(layer_idx / max(1, num_layers - 1))
                    
                    # Convert colors to RGB arrays for blending
                    if isinstance(cluster_color, str):
                        cluster_rgb = np.array(mcolors.to_rgb(cluster_color))
                    else:
                        cluster_rgb = np.array(cluster_color[:3])  # Take RGB part
                        
                    layer_rgb = np.array(layer_color[:3])  # Take RGB part
                    
                    # Blend colors (70% cluster, 30% layer)
                    blended_color = 0.7 * cluster_rgb + 0.3 * layer_rgb
                    
                    # Ensure color values are within [0, 1]
                    blended_color = np.clip(blended_color, 0, 1)
                    
                    colors.append(blended_color)
        
        # Log size values for debugging
        logger.info(f"Treemap sizes: {sizes}")
        
        # Convert sizes to a list of floats
        sizes_list = [float(s) for s in sizes]
        
        # Normalize rectangle sizes
        if sum(sizes_list) > 0:
            norm_sizes = squarify.normalize_sizes(sizes_list, 1000, 1000)
            
            # Create treemap layout
            rects = squarify.squarify(norm_sizes, 0, 0, 1000, 1000)
            
            # Plot rectangles
            for i, rect in enumerate(rects):
                x = rect['x'] / 1000
                y = rect['y'] / 1000
                dx = rect['dx'] / 1000
                dy = rect['dy'] / 1000
                
                ax.add_patch(
                    plt.Rectangle(
                        (x, y), dx, dy,
                        facecolor=colors[i],
                        edgecolor='white',
                        linewidth=1,
                        alpha=0.8
                    )
                )
                
                # Add text if rectangle is large enough
                if dx > 0.05 and dy > 0.05:
                    ax.text(
                        x + dx/2, y + dy/2, labels[i],
                        ha='center', va='center',
                        fontsize=small_fontsize,
                        color='black',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1)
                    )
        
        # Set axis limits and remove ticks
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Create legend for clusters
        legend_elements = []
        for cluster in sorted(set(node_clusters.values())):
            color = cluster_colors.get(cluster, 'gray')
            legend_elements.append(
                Line2D([0], [0], marker='s', color='w', markerfacecolor=color, 
                       markersize=10, label=f"Cluster {str(cluster)}")
            )
        
        # Add legend
        ax.legend(handles=legend_elements, loc='upper right', 
                  title="Clusters", fontsize=small_fontsize-1)
        
        count_type_str = "nodes" if count_type == "nodes" else "intralayer edges"
        logger.info(f"Successfully created treemap with {len(labels)} rectangles showing {count_type_str}")
        
    except Exception as e:
        logger.error(f"Error creating treemap: {str(e)}")
        logger.exception(e)
        ax.text(0.5, 0.5, 
                f"Error creating treemap: {str(e)}", 
                ha="center", va="center")
        ax.axis("off") 