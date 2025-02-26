import numpy as np

def create_layer_connectivity_chart(ax, visible_links, nodes_per_layer, layers, small_font, medium_font):
    """Create layer connectivity heatmap"""
    # Calculate layer connections
    layer_connections = np.zeros((len(layers), len(layers)))
    
    for start_idx, end_idx in visible_links:
        start_layer = start_idx // nodes_per_layer
        end_layer = end_idx // nodes_per_layer
        
        if start_layer != end_layer:  # Only count inter-layer connections
            layer_connections[start_layer, end_layer] += 1
            layer_connections[end_layer, start_layer] += 1  # Symmetric
    
    # Plot layer connectivity heatmap
    im = ax.imshow(layer_connections, cmap='viridis')
    ax.set_title('Layer Connectivity', **medium_font)
    ax.set_xticks(range(len(layers)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, **small_font)
    ax.set_yticklabels(layers, **small_font)
    
    return im, layer_connections 