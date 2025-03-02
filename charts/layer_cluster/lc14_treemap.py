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
    count_type="nodes",  # Options: 'nodes', 'intralayer_edges', or 'interlayer_edges'
):
    """
    Create a treemap visualization showing the distribution of nodes or edges across layers and clusters.
    Rectangle size represents the number of nodes or edges in each layer-cluster combination.

    Parameters:
    -----------
    count_type : str
        Type of counting to perform:
        - 'nodes' (default): Count nodes in each layer-cluster combination
        - 'intralayer_edges': Count edges between nodes in the same layer and cluster
        - 'interlayer_edges': Count edges between nodes in different layers but same cluster
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"Creating layer-cluster treemap with nodes_per_layer={nodes_per_layer}, count_type={count_type}"
    )
    logger.info(f"Number of visible links: {len(visible_links)}")
    logger.info(f"Number of node_ids: {len(node_ids)}")
    logger.info(f"Number of node_clusters: {len(node_clusters)}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Visible layer indices: {visible_layer_indices}")

    # Sample some links for debugging
    if visible_links and len(visible_links) > 0:
        sample_size = min(5, len(visible_links))
        logger.info(f"Sample of visible_links: {visible_links[:sample_size]}")

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
            ax.set_title(
                "Layer-Cluster Node Distribution Treemap", fontsize=medium_fontsize
            )
        elif count_type == "intralayer_edges":
            ax.set_title(
                "Layer-Cluster Intralayer Edge Distribution Treemap",
                fontsize=medium_fontsize,
            )
        elif count_type == "interlayer_edges":
            ax.set_title(
                "Layer-Cluster Interlayer Edge Distribution Treemap",
                fontsize=medium_fontsize,
            )
        else:
            ax.set_title("Layer-Cluster Distribution Treemap", fontsize=medium_fontsize)

        # Filter to only visible layers
        visible_layer_indices = visible_layer_indices or list(range(len(layers)))
        logger.info(f"Using visible layer indices: {visible_layer_indices}")

        # Check if visible_links contains indices or node IDs
        # If the first item in visible_links contains values that are integers and less than len(node_ids),
        # assume they are indices, otherwise assume they are node IDs
        links_are_indices = False
        if visible_links and len(visible_links) > 0:
            first_link = visible_links[0]
            logger.info(
                f"First link: {first_link}, types: {type(first_link[0])}, {type(first_link[1])}"
            )

            # Check if the values are integers (including numpy integer types)
            is_int_type_0 = isinstance(first_link[0], (int, np.integer))
            is_int_type_1 = isinstance(first_link[1], (int, np.integer))
            
            # Check if the values are within range of node_ids
            in_range_0 = first_link[0] < len(node_ids) if is_int_type_0 else False
            in_range_1 = first_link[1] < len(node_ids) if is_int_type_1 else False

            if is_int_type_0 and is_int_type_1 and in_range_0 and in_range_1:
                links_are_indices = True
                logger.info("Detected index-based visible_links")
            else:
                logger.info("Detected ID-based visible_links")

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

        # If node_to_layer is empty, try an alternative approach
        if not node_to_layer and isinstance(nodes_per_layer, int):
            logger.warning(
                "node_to_layer mapping is empty, trying alternative approach"
            )
            # Try to infer layer from node ID format (if it contains layer information)
            for node_id in node_ids:
                # Check if node_id has a format like "layer_nodeid" or "nodeid_layer"
                parts = node_id.split("_")
                if len(parts) > 1:
                    # Try to extract layer information from the node ID
                    for part in parts:
                        try:
                            # See if any part can be converted to an integer
                            layer_idx = int(part)
                            if layer_idx in visible_layer_indices:
                                node_to_layer[node_id] = layer_idx
                                break
                        except ValueError:
                            continue

        # If node_to_layer is still empty, try another approach based on node indices
        if not node_to_layer and isinstance(nodes_per_layer, int) and links_are_indices:
            logger.warning(
                "node_to_layer mapping is still empty, trying index-based approach"
            )
            # Assign layers based on node indices
            for i, node_id in enumerate(node_ids):
                layer_idx = i // nodes_per_layer
                if layer_idx in visible_layer_indices:
                    node_to_layer[node_id] = layer_idx

            # If we're using index-based links, also create a direct mapping from index to layer
            node_idx_to_layer = {}
            for i in range(len(node_ids)):
                layer_idx = i // nodes_per_layer
                if layer_idx in visible_layer_indices:
                    node_idx_to_layer[i] = layer_idx

            logger.info(
                f"Created node_idx_to_layer mapping with {len(node_idx_to_layer)} entries"
            )

            # Add a special case for interlayer edges with index-based links
            if count_type == "interlayer_edges":
                logger.info(
                    "Using special case for interlayer edges with index-based links"
                )

                # Count interlayer edges directly using node indices
                interlayer_count = 0
                cluster_layer_counts = {}

                for source_idx, target_idx in visible_links:
                    # Convert numpy integers to Python integers if needed
                    if isinstance(source_idx, np.integer):
                        source_idx = int(source_idx)
                    if isinstance(target_idx, np.integer):
                        target_idx = int(target_idx)
                        
                    # Check if indices are valid
                    if source_idx >= len(node_ids) or target_idx >= len(node_ids):
                        continue

                    # Get layers directly from indices
                    source_layer = source_idx // nodes_per_layer
                    target_layer = target_idx // nodes_per_layer

                    # Check if nodes are in different layers
                    if (
                        source_layer != target_layer
                        and source_layer in visible_layer_indices
                        and target_layer in visible_layer_indices
                    ):
                        # Get node IDs and clusters
                        source_id = node_ids[source_idx]
                        target_id = node_ids[target_idx]

                        source_cluster = node_clusters.get(source_id)
                        target_cluster = node_clusters.get(target_id)

                        # Check if both nodes are in the same cluster
                        if (
                            source_cluster is not None
                            and source_cluster == target_cluster
                        ):
                            cluster = source_cluster
                            interlayer_count += 1

                            # For interlayer edges, we count them for both layers involved
                            for layer_idx in [source_layer, target_layer]:
                                if cluster not in cluster_layer_counts:
                                    cluster_layer_counts[cluster] = {}
                                if layer_idx not in cluster_layer_counts[cluster]:
                                    cluster_layer_counts[cluster][layer_idx] = 0
                                # Count as 0.5 for each layer to avoid double counting
                                cluster_layer_counts[cluster][layer_idx] += 0.5

                logger.info(f"Special case found {interlayer_count} interlayer edges")
                logger.info(
                    f"Special case cluster layer counts: {cluster_layer_counts}"
                )

                # If we found interlayer edges, return early with these counts
                if interlayer_count > 0:
                    # Skip the regular processing
                    logger.info("Using special case results for treemap")

                    # Prepare data for treemap
                    labels = []
                    sizes = []
                    colors = []

                    # Create a colormap for layers
                    layer_cmap = plt.cm.viridis
                    num_layers = len(layers)

                    # Create rectangles for each cluster-layer combination
                    for cluster, layer_dict in sorted(cluster_layer_counts.items()):
                        for layer_idx, count in sorted(layer_dict.items()):
                            if layer_idx < len(layers):
                                layer_name = layers[layer_idx]
                                # Create label with appropriate count description
                                count_str = (
                                    str(int(count))
                                    if count.is_integer()
                                    else f"{count:.1f}"
                                )
                                label = f"C{cluster}-{layer_name}\n({count_str} inter edges)"
                                labels.append(label)

                                # Add size (must be float)
                                sizes.append(float(count))

                                # Create blended color between cluster and layer
                                cluster_color = cluster_colors.get(cluster, "gray")
                                layer_color = layer_cmap(
                                    layer_idx / max(1, num_layers - 1)
                                )

                                # Convert colors to RGB arrays for blending
                                if isinstance(cluster_color, str):
                                    cluster_rgb = np.array(
                                        mcolors.to_rgb(cluster_color)
                                    )
                                else:
                                    cluster_rgb = np.array(
                                        cluster_color[:3]
                                    )  # Take RGB part

                                layer_rgb = np.array(layer_color[:3])  # Take RGB part

                                # Blend colors (70% cluster, 30% layer)
                                blended_color = 0.7 * cluster_rgb + 0.3 * layer_rgb

                                # Ensure color values are within [0, 1]
                                blended_color = np.clip(blended_color, 0, 1)

                                colors.append(blended_color)

                    # Log size values for debugging
                    logger.info(f"Special case treemap sizes: {sizes}")
                    logger.info(f"Special case treemap labels: {labels}")

                    # Convert sizes to a list of floats
                    sizes_list = [float(s) for s in sizes]

                    # Normalize rectangle sizes
                    if sum(sizes_list) > 0:
                        norm_sizes = squarify.normalize_sizes(sizes_list, 1000, 1000)

                        # Create treemap layout
                        rects = squarify.squarify(norm_sizes, 0, 0, 1000, 1000)

                        # Plot rectangles
                        for i, rect in enumerate(rects):
                            x = rect["x"] / 1000
                            y = rect["y"] / 1000
                            dx = rect["dx"] / 1000
                            dy = rect["dy"] / 1000

                            ax.add_patch(
                                plt.Rectangle(
                                    (x, y),
                                    dx,
                                    dy,
                                    facecolor=colors[i],
                                    edgecolor="white",
                                    linewidth=1,
                                    alpha=0.8,
                                )
                            )

                            # Add text if rectangle is large enough
                            if dx > 0.05 and dy > 0.05:
                                ax.text(
                                    x + dx / 2,
                                    y + dy / 2,
                                    labels[i],
                                    ha="center",
                                    va="center",
                                    fontsize=small_fontsize,
                                    color="black",
                                    bbox=dict(
                                        facecolor="white",
                                        alpha=0.5,
                                        edgecolor="none",
                                        pad=1,
                                    ),
                                )

                        # Set axis limits and remove ticks
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.set_xticks([])
                        ax.set_yticks([])

                        # Create legend for clusters
                        legend_elements = []
                        for cluster in sorted(set(node_clusters.values())):
                            color = cluster_colors.get(cluster, "gray")
                            legend_elements.append(
                                Line2D(
                                    [0],
                                    [0],
                                    marker="s",
                                    color="w",
                                    markerfacecolor=color,
                                    markersize=10,
                                    label=f"Cluster {str(cluster)}",
                                )
                            )

                        # Add legend
                        ax.legend(
                            handles=legend_elements,
                            loc="upper right",
                            title="Clusters",
                            fontsize=small_fontsize - 1,
                        )

                        # Add explanation
                        ax.text(
                            0.5,
                            0.01,
                            "Rectangle size represents the number of edges between nodes in different layers but same cluster",
                            ha="center",
                            va="bottom",
                            fontsize=small_fontsize - 1,
                            style="italic",
                            color="dimgray",
                            transform=ax.transAxes,
                        )

                        logger.info(
                            f"Successfully created special case treemap with {len(labels)} rectangles"
                        )
                        return

        logger.info(f"Final node_to_layer mapping has {len(node_to_layer)} entries")
        # Log a sample of the node_to_layer mapping
        if node_to_layer:
            sample_entries = list(node_to_layer.items())[:5]
            logger.info(f"Sample of node_to_layer mapping: {sample_entries}")

        # Create a mapping from node index to node ID (for handling index-based visible_links)
        node_idx_to_id = {i: node_id for i, node_id in enumerate(node_ids)}
        logger.info(
            f"Created node_idx_to_id mapping with {len(node_idx_to_id)} entries"
        )

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
        elif count_type == "intralayer_edges":
            # Count intralayer edges by cluster and layer
            # First, create a mapping of node_id to cluster
            node_to_cluster = {
                node_id: cluster
                for node_id, cluster in node_clusters.items()
                if node_id in node_ids
            }
            logger.info(
                f"Created node_to_cluster mapping with {len(node_to_cluster)} entries"
            )

            # Count intralayer edges
            intralayer_count = 0
            for link in visible_links:
                # Handle both index-based and ID-based links
                if links_are_indices:
                    source_idx, target_idx = link
                    # Convert numpy integers to Python integers if needed
                    if isinstance(source_idx, np.integer):
                        source_idx = int(source_idx)
                    if isinstance(target_idx, np.integer):
                        target_idx = int(target_idx)
                        
                    source_id = node_idx_to_id[source_idx]
                    target_id = node_idx_to_id[target_idx]
                else:
                    source_id, target_id = link

                # Check if both nodes are in the same layer
                if (
                    source_id in node_to_layer
                    and target_id in node_to_layer
                    and node_to_layer[source_id] == node_to_layer[target_id]
                ):
                    layer_idx = node_to_layer[source_id]

                    # Check if both nodes are in the same cluster
                    if (
                        source_id in node_to_cluster
                        and target_id in node_to_cluster
                        and node_to_cluster[source_id] == node_to_cluster[target_id]
                    ):
                        cluster = node_to_cluster[source_id]
                        intralayer_count += 1

                        # Increment the count for this cluster-layer combination
                        if cluster not in cluster_layer_counts:
                            cluster_layer_counts[cluster] = {}
                        if layer_idx not in cluster_layer_counts[cluster]:
                            cluster_layer_counts[cluster][layer_idx] = 0
                        cluster_layer_counts[cluster][layer_idx] += 1

            logger.info(f"Found {intralayer_count} intralayer edges")
        elif count_type == "interlayer_edges":
            # Count interlayer edges by cluster
            # First, create a mapping of node_id to cluster
            node_to_cluster = {
                node_id: cluster
                for node_id, cluster in node_clusters.items()
                if node_id in node_ids
            }
            logger.info(
                f"Created node_to_cluster mapping with {len(node_to_cluster)} entries"
            )

            # Log a sample of the node_to_cluster mapping
            sample_entries = list(node_to_cluster.items())[:5]
            logger.info(f"Sample of node_to_cluster mapping: {sample_entries}")

            # Log a few sample links and their layer calculations
            logger.info("Sample links and their layer calculations:")
            for i, link in enumerate(visible_links[:3]):
                if links_are_indices:
                    source_idx, target_idx = link
                    # Convert numpy integers to Python integers if needed
                    if isinstance(source_idx, np.integer):
                        source_idx = int(source_idx)
                    if isinstance(target_idx, np.integer):
                        target_idx = int(target_idx)
                        
                    source_id = node_idx_to_id[source_idx]
                    target_id = node_idx_to_id[target_idx]
                    
                    source_layer = source_idx // nodes_per_layer if isinstance(nodes_per_layer, int) else "unknown"
                    target_layer = target_idx // nodes_per_layer if isinstance(nodes_per_layer, int) else "unknown"
                    
                    logger.info(f"  Link {i}: {source_idx} -> {target_idx}")
                    logger.info(f"    Source ID: {source_id}, Target ID: {target_id}")
                    logger.info(f"    Source layer: {source_layer}, Target layer: {target_layer}")
                    logger.info(f"    Different layers: {source_layer != target_layer}")
                    
                    if source_id in node_to_cluster and target_id in node_to_cluster:
                        source_cluster = node_to_cluster[source_id]
                        target_cluster = node_to_cluster[target_id]
                        logger.info(f"    Source cluster: {source_cluster}, Target cluster: {target_cluster}")
                        logger.info(f"    Same cluster: {source_cluster == target_cluster}")
            
            # Log some sample node IDs and their layers for debugging
            logger.info("Sample node IDs and their layers:")
            sample_nodes = list(node_to_layer.items())[:5]
            for node_id, layer in sample_nodes:
                logger.info(f"  Node {node_id} is in layer {layer}")
                
            # Log some sample node IDs and their clusters for debugging
            logger.info("Sample node IDs and their clusters:")
            sample_clusters = list(node_to_cluster.items())[:5]
            for node_id, cluster in sample_clusters:
                logger.info(f"  Node {node_id} is in cluster {cluster}")
                
            # Count interlayer edges
            interlayer_count = 0
            different_layers_count = 0
            same_cluster_count = 0
                
            for i, link in enumerate(visible_links):
                # Handle both index-based and ID-based links
                if links_are_indices:
                    source_idx, target_idx = link
                    # Convert numpy integers to Python integers if needed
                    if isinstance(source_idx, np.integer):
                        source_idx = int(source_idx)
                    if isinstance(target_idx, np.integer):
                        target_idx = int(target_idx)
                        
                    try:
                        source_id = node_idx_to_id[source_idx]
                        target_id = node_idx_to_id[target_idx]
                    except KeyError as e:
                        logger.error(f"KeyError at link {i}: {link}, error: {e}")
                        continue
                else:
                    source_id, target_id = link

                # Debug every 1000th link
                if i % 1000 == 0:
                    logger.debug(f"Processing link {i}: {source_id} -> {target_id}")
                    logger.debug(
                        f"  source_id in node_to_layer: {source_id in node_to_layer}"
                    )
                    logger.debug(
                        f"  target_id in node_to_layer: {target_id in node_to_layer}"
                    )
                    if source_id in node_to_layer and target_id in node_to_layer:
                        logger.debug(f"  source layer: {node_to_layer[source_id]}")
                        logger.debug(f"  target layer: {node_to_layer[target_id]}")
                        logger.debug(
                            f"  different layers: {node_to_layer[source_id] != node_to_layer[target_id]}"
                        )

                    logger.debug(
                        f"  source_id in node_to_cluster: {source_id in node_to_cluster}"
                    )
                    logger.debug(
                        f"  target_id in node_to_cluster: {target_id in node_to_cluster}"
                    )
                    if source_id in node_to_cluster and target_id in node_to_cluster:
                        logger.debug(f"  source cluster: {node_to_cluster[source_id]}")
                        logger.debug(f"  target cluster: {node_to_cluster[target_id]}")
                        logger.debug(
                            f"  same cluster: {node_to_cluster[source_id] == node_to_cluster[target_id]}"
                        )

                # Check if nodes are in different layers
                if (
                    source_id in node_to_layer
                    and target_id in node_to_layer
                    and node_to_layer[source_id] != node_to_layer[target_id]
                ):
                    different_layers_count += 1
                    source_layer = node_to_layer[source_id]
                    target_layer = node_to_layer[target_id]

                    # Check if both nodes are in the same cluster
                    if (
                        source_id in node_to_cluster
                        and target_id in node_to_cluster
                        and node_to_cluster[source_id] == node_to_cluster[target_id]
                    ):
                        same_cluster_count += 1
                        cluster = node_to_cluster[source_id]
                        interlayer_count += 1

                        # For interlayer edges, we count them for both layers involved
                        # This gives a better representation of which layers are connected
                        for layer_idx in [source_layer, target_layer]:
                            if cluster not in cluster_layer_counts:
                                cluster_layer_counts[cluster] = {}
                            if layer_idx not in cluster_layer_counts[cluster]:
                                cluster_layer_counts[cluster][layer_idx] = 0
                            # Count as 0.5 for each layer to avoid double counting
                            cluster_layer_counts[cluster][layer_idx] += 0.5

            logger.info(
                f"Found {different_layers_count} links between different layers"
            )
            logger.info(f"Found {same_cluster_count} links between same clusters")
            logger.info(
                f"Found {interlayer_count} interlayer edges (different layers, same cluster)"
            )
            logger.info(f"Cluster layer counts: {cluster_layer_counts}")

        # Check if we have any data
        if not cluster_layer_counts:
            logger.warning("No data to display in treemap")
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
            cluster_colors_list.append(cluster_colors.get(cluster, "gray"))

        # Create rectangles for each cluster-layer combination
        for cluster, layer_dict in sorted(cluster_layer_counts.items()):
            for layer_idx, count in sorted(layer_dict.items()):
                if layer_idx < len(layers):
                    layer_name = layers[layer_idx]
                    # Create label with appropriate count description
                    if count_type == "nodes":
                        label = f"C{cluster}-{layer_name}\n({count} nodes)"
                    elif count_type == "intralayer_edges":
                        label = f"C{cluster}-{layer_name}\n({int(count)} intra edges)"
                    elif count_type == "interlayer_edges":
                        # Convert to integer if it's a whole number, otherwise show one decimal place
                        count_str = (
                            str(int(count)) if count.is_integer() else f"{count:.1f}"
                        )
                        label = f"C{cluster}-{layer_name}\n({count_str} inter edges)"
                    labels.append(label)

                    # Add size (must be float)
                    sizes.append(float(count))

                    # Create blended color between cluster and layer
                    cluster_color = cluster_colors.get(cluster, "gray")
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
                x = rect["x"] / 1000
                y = rect["y"] / 1000
                dx = rect["dx"] / 1000
                dy = rect["dy"] / 1000

                ax.add_patch(
                    plt.Rectangle(
                        (x, y),
                        dx,
                        dy,
                        facecolor=colors[i],
                        edgecolor="white",
                        linewidth=1,
                        alpha=0.8,
                    )
                )

                # Add text if rectangle is large enough
                if dx > 0.05 and dy > 0.05:
                    ax.text(
                        x + dx / 2,
                        y + dy / 2,
                        labels[i],
                        ha="center",
                        va="center",
                        fontsize=small_fontsize,
                        color="black",
                        bbox=dict(
                            facecolor="white", alpha=0.5, edgecolor="none", pad=1
                        ),
                    )

        # Set axis limits and remove ticks
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        # Create legend for clusters
        legend_elements = []
        for cluster in sorted(set(node_clusters.values())):
            color = cluster_colors.get(cluster, "gray")
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                    label=f"Cluster {str(cluster)}",
                )
            )

        # Add legend
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            title="Clusters",
            fontsize=small_fontsize - 1,
        )

        # Add explanation of count type
        count_type_explanations = {
            "nodes": "Rectangle size represents the number of nodes in each layer-cluster combination",
            "intralayer_edges": "Rectangle size represents the number of edges between nodes in the same layer and cluster",
            "interlayer_edges": "Rectangle size represents the number of edges between nodes in different layers but same cluster",
        }

        explanation = count_type_explanations.get(count_type, "")
        if explanation:
            ax.text(
                0.5,
                0.01,
                explanation,
                ha="center",
                va="bottom",
                fontsize=small_fontsize - 1,
                style="italic",
                color="dimgray",
                transform=ax.transAxes,
            )

        count_type_str = (
            "nodes"
            if count_type == "nodes"
            else "intralayer edges"
            if count_type == "intralayer_edges"
            else "interlayer edges"
        )
        logger.info(
            f"Successfully created treemap with {len(labels)} rectangles showing {count_type_str}"
        )

    except Exception as e:
        logger.error(f"Error creating treemap: {str(e)}")
        logger.exception(e)
        ax.text(0.5, 0.5, f"Error creating treemap: {str(e)}", ha="center", va="center")
        ax.axis("off")
