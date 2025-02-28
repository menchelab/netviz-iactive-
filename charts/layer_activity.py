import numpy as np


def create_layer_activity_chart(
    ax, visible_links, nodes_per_layer, layers, small_font, medium_font
):
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

    # Create bar chart with two datasets side by side
    x = np.arange(len(layers))
    width = 0.35  # Width of each bar, narrower to fit side by side

    # Primary axis for both types of data
    ax1 = ax
    bars1 = ax1.bar(
        x - width / 2,
        edges_per_layer_count,
        width,
        label="Intralayer Edges",
        color="skyblue",
    )
    bars2 = ax1.bar(
        x + width / 2,
        interlayer_connections_per_layer,
        width,
        label="Interlayer Connections",
        color="red",
        alpha=0.8,
    )

    ax1.set_xlabel("Layers", **small_font)
    ax1.set_ylabel("Edge Count", **small_font)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers, rotation=90, **small_font)
    ax1.tick_params(axis="y", labelsize=6)
    ax1.legend(loc="upper right", prop={"size": 6})

    ax.set_title("Layer Activity", **medium_font)
