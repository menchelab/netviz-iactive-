import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import logging
from collections import defaultdict


def create_cluster_heatmap(
    ax,
    visible_links,
    node_ids,
    node_clusters,
    nodes_per_layer,
    layers,
    small_font,
    medium_font,
    visible_layer_indices=None,
):
    """
    Create a heatmap showing interlayer connections with pie charts in each cell
    showing the distribution of clusters for those connections.
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating cluster heatmap")

    # Clear the axis
    ax.clear()
    ax.set_title("Cluster Distribution Between Layers", **medium_font)

    # Filter to only show visible layers if specified
    visible_layers = None
    if visible_layer_indices is not None and len(visible_layer_indices) > 0:
        visible_layers = set(visible_layer_indices)
        logger.info(
            f"Filtering heatmap to show only {len(visible_layers)} visible layers"
        )

    # Initialize data structures to track connections
    layer_connections = {}
    layer_cluster_connections = defaultdict(lambda: defaultdict(int))

    # Count connections between layers by cluster
    for start_idx, end_idx in visible_links:
        if start_idx == end_idx:
            continue

        start_layer_idx = start_idx // nodes_per_layer
        end_layer_idx = end_idx // nodes_per_layer

        if start_layer_idx == end_layer_idx:
            continue

        if visible_layers is not None and (
            start_layer_idx not in visible_layers or end_layer_idx not in visible_layers
        ):
            continue

        start_node_id = node_ids[start_idx]
        cluster = node_clusters.get(start_node_id, "Unknown")

        # Ensure consistent ordering (smaller layer index first)
        if start_layer_idx > end_layer_idx:
            start_layer_idx, end_layer_idx = end_layer_idx, start_layer_idx

        # Create the key for layer pair
        layer_pair = (start_layer_idx, end_layer_idx)

        # Increment total connections
        if layer_pair not in layer_connections:
            layer_connections[layer_pair] = 0
        layer_connections[layer_pair] += 1

        # Increment cluster-specific connections
        layer_cluster_connections[layer_pair][cluster] += 1

    # Check if we have any connections
    if not layer_connections:
        ax.text(
            0.5,
            0.5,
            "No interlayer connections to display",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off")
        return

    # Get unique layers and clusters
    if visible_layers is not None:
        unique_layer_indices = sorted(visible_layers)
        unique_layers = [layers[i] for i in unique_layer_indices]
    else:
        unique_layer_indices = list(range(len(layers)))
        unique_layers = layers

    unique_clusters = set()
    for layer_pair, cluster_dict in layer_cluster_connections.items():
        unique_clusters.update(cluster_dict.keys())
    unique_clusters = sorted(unique_clusters)

    # Create a colormap for clusters
    colormap = plt.cm.tab20
    cluster_colors = {
        cluster: colormap(i % 20) for i, cluster in enumerate(unique_clusters)
    }

    try:
        # Create a matrix for the heatmap
        n_layers = len(unique_layers)
        connection_matrix = np.zeros((n_layers, n_layers))

        # Fill the matrix with connection counts
        for (src_idx, tgt_idx), count in layer_connections.items():
            if visible_layers is not None:
                if src_idx not in visible_layers or tgt_idx not in visible_layers:
                    continue
                src_pos = list(visible_layers).index(src_idx)
                tgt_pos = list(visible_layers).index(tgt_idx)
            else:
                src_pos = src_idx
                tgt_pos = tgt_idx

            connection_matrix[src_pos, tgt_pos] = count

        # Create the heatmap
        im = ax.imshow(connection_matrix, cmap="viridis", interpolation="nearest")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel(
            "Number of connections", rotation=-90, va="bottom", **small_font
        )
        cbar.ax.tick_params(labelsize=small_font["fontsize"])

        # Set ticks and labels
        ax.set_xticks(np.arange(n_layers))
        ax.set_yticks(np.arange(n_layers))
        ax.set_xticklabels(unique_layers, rotation=45, ha="right", **small_font)
        ax.set_yticklabels(unique_layers, **small_font)

        # Add pie charts to cells with connections
        for i in range(n_layers):
            for j in range(n_layers):
                if connection_matrix[i, j] > 0:
                    # Get the layer indices
                    if visible_layers is not None:
                        src_idx = list(visible_layers)[i]
                        tgt_idx = list(visible_layers)[j]
                    else:
                        src_idx = i
                        tgt_idx = j

                    # Ensure consistent ordering
                    if src_idx > tgt_idx:
                        src_idx, tgt_idx = tgt_idx, src_idx

                    # Get cluster distribution
                    cluster_counts = layer_cluster_connections[(src_idx, tgt_idx)]

                    # Create pie chart data
                    sizes = [
                        cluster_counts[cluster]
                        for cluster in unique_clusters
                        if cluster in cluster_counts
                    ]
                    colors = [
                        cluster_colors[cluster]
                        for cluster in unique_clusters
                        if cluster in cluster_counts
                    ]

                    # Skip if no data
                    if not sizes:
                        continue

                    # Create a small pie chart
                    size = 0.3  # Size of the pie chart relative to the cell (reduced from 0.4)

                    # Ensure the pie chart is centered in the cell
                    ax.pie(
                        sizes,
                        colors=colors,
                        radius=size,
                        center=(j, i),  # Note: x, y are swapped in imshow
                        wedgeprops=dict(width=size * 0.8, edgecolor="w", linewidth=0.5),
                    )

        # Add legend for clusters
        legend_elements = []
        for cluster in unique_clusters:
            legend_elements.append(
                mpatches.Patch(
                    facecolor=cluster_colors.get(cluster, "gray"),
                    alpha=0.7,
                    label=cluster,
                )
            )

        # Position the legend outside the plot area
        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(4, len(unique_clusters)),
            frameon=False,
            fontsize=small_font["fontsize"],
        )

        # Adjust layout to make room for the legend
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)

        # Set aspect ratio to be equal
        ax.set_aspect("equal")

        logger.info(
            f"Created cluster heatmap with {n_layers} layers and {len(unique_clusters)} clusters"
        )

    except Exception as e:
        logger.error(f"Error creating cluster heatmap: {str(e)}")
        ax.clear()
        ax.text(
            0.5,
            0.5,
            f"Error creating cluster heatmap: {str(e)}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off")
