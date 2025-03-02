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
):
    """
    Create a treemap visualization showing the distribution of nodes across layers and clusters.
    Rectangle size represents the number of nodes in each layer-cluster combination.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating layer-cluster treemap with nodes_per_layer={nodes_per_layer}")
    
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
        ax.set_title("Layer-Cluster Distribution Treemap", fontsize=medium_fontsize)
        
        # Filter to only visible layers
        visible_layer_indices = visible_layer_indices or list(range(len(layers)))
        
        # Count nodes by cluster and layer
        cluster_layer_counts = {}
        for node_id, cluster in node_clusters.items():
            if node_id in node_ids:
                for layer_idx in visible_layer_indices:
                    if layer_idx < len(layers):
                        # Handle different formats of nodes_per_layer
                        node_in_layer = False
                        if isinstance(nodes_per_layer, dict):
                            # If nodes_per_layer is a dictionary mapping layer_idx -> list of nodes
                            node_in_layer = node_id in nodes_per_layer.get(layer_idx, [])
                        elif isinstance(nodes_per_layer, int):
                            # If nodes_per_layer is an integer (number of nodes per layer)
                            # Calculate the layer for this node based on its index in node_ids
                            try:
                                node_idx = node_ids.index(node_id)
                                node_layer = node_idx // nodes_per_layer
                                node_in_layer = node_layer == layer_idx
                            except ValueError:
                                # Node ID not found in node_ids list
                                node_in_layer = False
                        
                        if node_in_layer:
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
                    # Create label
                    label = f"C{cluster}-{layer_name}\n({count})"
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
        
        logger.info(f"Successfully created treemap with {len(labels)} rectangles")
        
    except Exception as e:
        logger.error(f"Error creating treemap: {str(e)}")
        logger.exception(e)
        ax.text(0.5, 0.5, 
                f"Error creating treemap: {str(e)}", 
                ha="center", va="center")
        ax.axis("off") 