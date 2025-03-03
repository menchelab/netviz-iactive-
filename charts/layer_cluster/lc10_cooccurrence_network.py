import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm


def create_cluster_cooccurrence_network(
    ax,
    visible_links,
    node_ids,
    node_clusters,
    nodes_per_layer,  # This is now an integer
    layers,
    small_font,
    medium_font,
    visible_layer_indices=None,
    cluster_colors=None,
):
    """
    Create a network visualization showing how clusters co-occur across layers.
    Nodes represent clusters, and edges represent co-occurrence in the same layer.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to draw the network on
    visible_links : list of tuples
        List of (start_idx, end_idx) tuples representing visible links
    node_ids : list
        List of node IDs
    node_clusters : dict
        Dictionary mapping node IDs to cluster labels
    nodes_per_layer : int
        Number of nodes in each layer
    layers : list
        List of layer names
    small_font : dict
        Dictionary with font properties for small text
    medium_font : dict
        Dictionary with font properties for medium text
    visible_layer_indices : list, optional
        List of indices of visible layers
    cluster_colors : dict, optional
        Dictionary mapping cluster labels to colors
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"Creating cluster co-occurrence network with nodes_per_layer={nodes_per_layer}"
    )
    logger.info(
        f"Visible links: {len(visible_links)}, Node IDs: {len(node_ids)}, Clusters: {len(node_clusters)}"
    )
    logger.info(f"Layers: {layers}")
    logger.info(f"Visible layer indices: {visible_layer_indices}")

    # Handle font parameters correctly
    if isinstance(medium_font, dict):
        medium_fontsize = medium_font.get("fontsize", 12)
    else:
        medium_fontsize = medium_font

    if isinstance(small_font, dict):
        small_fontsize = small_font.get("fontsize", 10)
    else:
        small_fontsize = small_font

    logger.info(
        f"Using medium_fontsize={medium_fontsize}, small_fontsize={small_fontsize}"
    )

    # Clear the axis
    ax.clear()
    ax.set_title("Cluster Co-occurrence Network", fontsize=medium_fontsize)

    # Filter visible layers if specified
    if visible_layer_indices is not None:
        visible_layers = set(visible_layer_indices)
        logger.info(
            f"Filtering co-occurrence network to show only {len(visible_layers)} visible layers"
        )
    else:
        visible_layers = set(range(len(layers)))
        logger.info("No layer filtering applied")

    # Get visible node indices
    visible_node_indices = set()
    for start_idx, end_idx in visible_links:
        visible_node_indices.add(start_idx)
        visible_node_indices.add(end_idx)

    logger.info(f"Found {len(visible_node_indices)} visible nodes")

    # Track clusters present in each layer
    clusters_by_layer = defaultdict(set)

    # Track all unique clusters
    all_clusters = set()

    # Process each visible node
    for node_idx in visible_node_indices:
        # Calculate layer index using integer division
        layer_idx = node_idx // nodes_per_layer

        # Skip if layer is not visible
        if layer_idx not in visible_layers:
            continue

        node_id = node_ids[node_idx]
        cluster = node_clusters.get(node_id, "Unknown")

        # Add to tracking
        clusters_by_layer[layer_idx].add(cluster)
        all_clusters.add(cluster)

    logger.info(
        f"Found {len(all_clusters)} unique clusters across {len(clusters_by_layer)} layers"
    )

    # Check if we have any data
    if not all_clusters:
        logger.warning("No cluster co-occurrence data to display")
        ax.text(
            0.5,
            0.5,
            "No cluster co-occurrence data to display",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off")
        return

    try:
        # Create a graph to represent co-occurrence
        G = nx.Graph()

        # Add nodes for each cluster
        for cluster in all_clusters:
            G.add_node(cluster)

        # Count co-occurrences between clusters
        cooccurrence_counts = defaultdict(int)

        # For each layer, count co-occurrences
        for layer_idx, clusters in clusters_by_layer.items():
            # Skip layers with fewer than 2 clusters
            if len(clusters) < 2:
                continue

            # Count co-occurrences for all pairs of clusters in this layer
            for cluster1 in clusters:
                for cluster2 in clusters:
                    if cluster1 < cluster2:  # Avoid double counting
                        cooccurrence_counts[(cluster1, cluster2)] += 1

        logger.info(f"Found {len(cooccurrence_counts)} cluster co-occurrence pairs")

        # Add edges for co-occurrences
        for (cluster1, cluster2), count in cooccurrence_counts.items():
            G.add_edge(cluster1, cluster2, weight=count, width=np.sqrt(count) * 2)

        # Check if the graph has any edges
        if not G.edges():
            logger.warning("No cluster co-occurrences found")
            ax.text(
                0.5,
                0.5,
                "No cluster co-occurrences found",
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax.axis("off")
            return

        # Create a layout for the graph
        try:
            # Try spring layout first
            pos = nx.spring_layout(G, seed=42)
            logger.info("Using spring layout for graph")
        except Exception as e:
            # Fall back to circular layout if spring layout fails
            logger.warning(
                f"Spring layout failed: {str(e)}, using circular layout instead"
            )
            pos = nx.circular_layout(G)

        # Get edge weights for line thickness
        edge_weights = [G[u][v].get("weight", 1) for u, v in G.edges()]

        # Normalize edge weights for better visualization
        if edge_weights:
            max_weight = max(edge_weights)
            normalized_weights = [w / max_weight * 5 for w in edge_weights]
            logger.info(
                f"Edge weights range from {min(edge_weights)} to {max(edge_weights)}"
            )
        else:
            normalized_weights = []
            logger.warning("No edge weights found")

        # Create a colormap for edges based on weight
        if edge_weights:
            edge_colors = plt.cm.viridis(np.array(edge_weights) / max(edge_weights))
        else:
            edge_colors = []

        # Draw the network
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, width=normalized_weights, edge_color=edge_colors, alpha=0.7, ax=ax
        )

        # Draw nodes
        if cluster_colors:
            # Use provided cluster colors
            node_colors = [
                cluster_colors.get(cluster, (0.5, 0.5, 0.5, 1.0))
                for cluster in G.nodes()
            ]
            logger.info("Using provided cluster colors")
        else:
            # Generate colors based on cluster names
            colormap = plt.cm.tab20
            node_colors = [colormap(hash(str(cluster)) % 20) for cluster in G.nodes()]
            logger.info("Generated cluster colors using tab20 colormap")

        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=800,
            alpha=0.8,
            edgecolors="black",
            ax=ax,
        )

        # Draw labels
        nx.draw_networkx_labels(
            G, pos, font_size=small_fontsize, font_weight="bold", ax=ax
        )

        # Add edge labels for significant co-occurrences
        edge_labels = {
            (u, v): f"{d['weight']}"
            for u, v, d in G.edges(data=True)
            if d["weight"] > 1
        }
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=small_fontsize, ax=ax
        )

        # Remove axis
        ax.axis("off")

        # Add a title with more information
        ax.set_title(
            f"Cluster Co-occurrence Network\n({len(all_clusters)} clusters, {len(visible_layers)} layers)",
            fontsize=medium_fontsize,
        )

        logger.info(
            f"Successfully created cluster co-occurrence network with {len(all_clusters)} clusters"
        )

    except Exception as e:
        logger.error(f"Error creating cluster co-occurrence network: {str(e)}")
        ax.clear()
        ax.text(
            0.5,
            0.5,
            f"Error creating network: {str(e)}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off")
