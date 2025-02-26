import numpy as np

def create_layer_activity_chart(ax, visible_links, nodes_per_layer, layers, small_font, medium_font):
    """Create chart showing edge counts and interlayer connections per layer"""
    edges_per_layer_count = [0] * len(layers)
    interlayer_connections_per_layer = [0] * len(layers)

    # Count edges per layer (intralayer only)
    for start_idx, end_idx in visible_links:
        start_layer = start_idx // nodes_per_layer
        end_layer = end_idx // nodes_per_layer
        if start_layer == end_layer:
            edges_per_layer_count[start_layer] += 1
        else:
            # Count interlayer connections for both layers involved
            interlayer_connections_per_layer[start_layer] += 1
            interlayer_connections_per_layer[end_layer] += 1

    # Create bar chart with two y-axes
    x = np.arange(len(layers))
    width = 0.7  # Make bars wider since we only have one set on primary axis

    # Primary axis for intralayer edges
    ax1 = ax
    bars1 = ax1.bar(x, edges_per_layer_count, width, label='Intralayer Edges', color='skyblue')
    ax1.set_xlabel('Layers', **small_font)
    ax1.set_ylabel('Intralayer Edge Count', **small_font)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers, rotation=90, **small_font)
    ax1.tick_params(axis='y', labelsize=6)

    # Secondary axis for interlayer connections
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x, interlayer_connections_per_layer, width*0.7, label='Interlayer Connections', 
                   color='red', alpha=0.6)  # Slightly narrower and transparent
    ax2.set_ylabel('Interlayer Connection Count', color='red', **small_font)
    ax2.tick_params(axis='y', labelcolor='red', labelsize=6)

    # Add legend with both datasets
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', prop={'size': 6})

    ax.set_title('Layer Activity', **medium_font) 