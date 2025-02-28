import numpy as np


def create_layer_connectivity_chart(
    ax, visible_links, nodes_per_layer, layers, small_font, medium_font, layer_index_map=None
):
    """Create layer connectivity heatmap"""
    # Calculate layer connections
    layer_connections = np.zeros((len(layers), len(layers)))

    # If no layer_index_map provided, create identity mapping
    if layer_index_map is None:
        layer_index_map = {i: i for i in range(len(layers))}

    for start_idx, end_idx in visible_links:
        # Get original layer indices
        orig_start_layer = start_idx // nodes_per_layer
        orig_end_layer = end_idx // nodes_per_layer

        # Skip if either layer is not in our mapping (filtered out)
        if orig_start_layer not in layer_index_map or orig_end_layer not in layer_index_map:
            continue

        # Map to filtered indices
        start_layer = layer_index_map[orig_start_layer]
        end_layer = layer_index_map[orig_end_layer]

        if start_layer != end_layer:  # Only count inter-layer connections
            layer_connections[start_layer, end_layer] += 1
            layer_connections[end_layer, start_layer] += 1  # Symmetric

    # Plot layer connectivity heatmap
    im = ax.imshow(layer_connections, cmap="viridis")
    ax.set_title("Layer Connectivity", **medium_font)
    ax.set_xticks(range(len(layers)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, **small_font)
    ax.set_yticklabels(layers, **small_font)

    return im, layer_connections
