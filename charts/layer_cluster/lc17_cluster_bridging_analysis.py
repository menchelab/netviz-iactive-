import logging
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_cluster_bridging_analysis(
    ax,
    visible_links,
    node_ids,
    node_clusters,
    nodes_per_layer,
    layers,
    visible_layer_indices,
    cluster_colors,
    layer_colors,
    analysis_type="bridge_score",
):
    """
    Create a visualization of how clusters bridge between different layers in the network.

    This visualization builds a custom network by duplicating each node for each layer it's in,
    using the naming convention <layer>_<node>. This creates a network where the duplicated nodes
    connect interlayer and intralayer edges.

    For interlayer edges, all possible connections are created between duplicated nodes.
    For intralayer edges, only existing edges from the original network are added.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to draw the visualization on
    visible_links : list
        List of (source_id, target_id) tuples representing visible edges
    node_ids : list
        List of all node IDs in the network
    node_clusters : dict
        Dictionary mapping node IDs to cluster assignments
    nodes_per_layer : dict or int
        Dictionary mapping layer indices to lists of node IDs, or integer for number of nodes per layer
    layers : list
        List of layer names
    visible_layer_indices : list
        List of indices of visible layers
    cluster_colors : dict
        Dictionary mapping cluster IDs to colors
    layer_colors : dict
        Dictionary mapping layer indices to colors
    analysis_type : str
        Type of bridging analysis to perform: 'bridge_score', 'flow_efficiency', 'layer_span',
        'centrality_distribution', 'cluster_cohesion', 'information_flow', 'structural_holes',
        'cross_layer_influence', 'cluster_resilience', 'path_diversity', 'boundary_spanning',
        'module_conservation', 'functional_enrichment', 'disease_association', 'co_expression_correlation'
    """
    try:
        logging.info(
            f"Visible links: {len(visible_links)}, Node IDs: {len(node_ids)}, Clusters: {len(node_clusters)}"
        )
        logging.info(f"Layers: {layers}")
        logging.info(f"Visible layer indices: {visible_layer_indices}")

        # Clear the axis and set title
        ax.clear()
        ax.set_title(
            f"Cluster Bridging Analysis: {analysis_type.replace('_', ' ').title()}"
        )

        # Filter to show only visible layers
        logging.info(
            f"Filtering bridging analysis to show only {len(visible_layer_indices)} visible layers"
        )

        # Build a multilayer network with duplicated nodes
        G = nx.Graph()

        # Check if nodes_per_layer is an integer or a dictionary
        if isinstance(nodes_per_layer, int):
            # If it's an integer, create a dictionary mapping layer indices to node ranges
            nodes_per_layer_dict = {}
            for layer_idx in range(len(layers)):
                start_idx = layer_idx * nodes_per_layer
                end_idx = start_idx + nodes_per_layer
                nodes_per_layer_dict[layer_idx] = [
                    node_ids[i] for i in range(start_idx, end_idx) if i < len(node_ids)
                ]
        else:
            # If it's already a dictionary, use it directly
            nodes_per_layer_dict = nodes_per_layer

        # Create a mapping from node index to node ID
        node_idx_to_id = {i: node_id for i, node_id in enumerate(node_ids)}

        # Create a mapping from node ID to node index
        node_id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

        # Track which layers each node appears in
        node_layers = defaultdict(list)

        # First, identify which layers each node appears in
        for layer_idx, node_list in nodes_per_layer_dict.items():
            if layer_idx in visible_layer_indices:
                for node_id in node_list:
                    if node_id in node_ids:
                        node_layers[node_id].append(layer_idx)

        # Track nodes by layer and cluster for analysis
        nodes_by_layer = defaultdict(list)
        nodes_by_cluster = defaultdict(list)

        # Create duplicated nodes for each layer a node appears in
        duplicated_nodes = {}  # Maps original node ID to list of duplicated node IDs

        for node_id, layers_list in node_layers.items():
            duplicated_nodes[node_id] = []
            cluster = node_clusters.get(node_id)
            if cluster is None:
                continue  # Skip nodes without cluster assignment

            for layer_idx in layers_list:
                # Create a new node ID in the format <layer>_<node>
                new_node_id = f"{layer_idx}_{node_id}"
                duplicated_nodes[node_id].append(new_node_id)

                # Add the node to the graph with attributes
                G.add_node(
                    new_node_id, original_id=node_id, cluster=cluster, layer=layer_idx
                )

                # Track nodes by layer and cluster for analysis
                nodes_by_layer[layer_idx].append(new_node_id)
                nodes_by_cluster[cluster].append(new_node_id)

        logging.info(
            f"Created {len(G.nodes)} duplicated nodes from {len(node_layers)} original nodes"
        )

        # Store original network edges
        original_edges = set()
        for source_idx, target_idx in visible_links:
            source_id = node_idx_to_id[source_idx]
            target_id = node_idx_to_id[target_idx]
            original_edges.add((source_id, target_id))
            original_edges.add(
                (target_id, source_id)
            )  # Add both directions since it's an undirected graph

        # Add intralayer edges (only for existing edges in the original network)
        intralayer_edges = []
        for source_id, target_id in original_edges:
            # Find the layer(s) where both nodes exist
            common_layers = set(node_layers[source_id]) & set(node_layers[target_id])

            for layer_idx in common_layers:
                # Create the duplicated node IDs
                source_node = f"{layer_idx}_{source_id}"
                target_node = f"{layer_idx}_{target_id}"

                # Add the intralayer edge
                G.add_edge(source_node, target_node, edge_type="intralayer")
                intralayer_edges.append((source_node, target_node))

        # Add interlayer edges (between all duplicated nodes of the same original node)
        interlayer_edges = []
        for source_idx, target_idx in visible_links:
            source_id = node_idx_to_id[source_idx]
            target_id = node_idx_to_id[target_idx]

            # Get the layers for these nodes
            source_layers = node_layers[source_id]
            target_layers = node_layers[target_id]

            # Connect nodes across different layers
            for source_layer in source_layers:
                for target_layer in target_layers:
                    if (
                        source_layer != target_layer
                    ):  # Only connect across different layers
                        # Create the duplicated node IDs
                        source_node = f"{source_layer}_{source_id}"
                        target_node = f"{target_layer}_{target_id}"

                        # Add the interlayer edge
                        G.add_edge(source_node, target_node, edge_type="interlayer")
                        interlayer_edges.append((source_node, target_node))

        logging.info(
            f"Added {len(interlayer_edges)} interlayer edges and {len(intralayer_edges)} intralayer edges"
        )

        # Get unique clusters
        unique_clusters = sorted(set(node_clusters.values()))
        logging.info(
            f"Found {len(unique_clusters)} unique clusters across {len(visible_layer_indices)} layers"
        )

        # Perform the selected analysis
        if analysis_type == "bridge_score":
            _analyze_bridge_score(
                ax,
                G,
                unique_clusters,
                nodes_by_cluster,
                nodes_by_layer,
                cluster_colors,
                layers,
                visible_layer_indices,
            )
        elif analysis_type == "flow_efficiency":
            _analyze_flow_efficiency(
                ax,
                G,
                unique_clusters,
                nodes_by_cluster,
                nodes_by_layer,
                cluster_colors,
                layers,
                visible_layer_indices,
            )
        elif analysis_type == "layer_span":
            _analyze_layer_span(
                ax,
                G,
                unique_clusters,
                nodes_by_cluster,
                nodes_by_layer,
                cluster_colors,
                layers,
                visible_layer_indices,
                layer_colors,
            )
        elif analysis_type == "centrality_distribution":
            _analyze_centrality_distribution(
                ax,
                G,
                unique_clusters,
                nodes_by_cluster,
                nodes_by_layer,
                cluster_colors,
                layers,
                visible_layer_indices,
            )
        elif analysis_type == "cluster_cohesion":
            _analyze_cluster_cohesion(
                ax,
                G,
                unique_clusters,
                nodes_by_cluster,
                nodes_by_layer,
                cluster_colors,
                layers,
                visible_layer_indices,
            )
        elif analysis_type == "information_flow":
            _analyze_information_flow(
                ax,
                G,
                unique_clusters,
                nodes_by_cluster,
                nodes_by_layer,
                cluster_colors,
                layers,
                visible_layer_indices,
            )
        elif analysis_type == "structural_holes":
            _analyze_structural_holes(
                ax,
                G,
                unique_clusters,
                nodes_by_cluster,
                nodes_by_layer,
                cluster_colors,
                layers,
                visible_layer_indices,
            )
        elif analysis_type == "cross_layer_influence":
            _analyze_cross_layer_influence(
                ax,
                G,
                unique_clusters,
                nodes_by_cluster,
                nodes_by_layer,
                cluster_colors,
                layers,
                visible_layer_indices,
            )
        elif analysis_type == "cluster_resilience":
            _analyze_cluster_resilience(
                ax,
                G,
                unique_clusters,
                nodes_by_cluster,
                nodes_by_layer,
                cluster_colors,
                layers,
                visible_layer_indices,
            )
        elif analysis_type == "path_diversity":
            _analyze_path_diversity(
                ax,
                G,
                unique_clusters,
                nodes_by_cluster,
                nodes_by_layer,
                cluster_colors,
                layers,
                visible_layer_indices,
            )
        elif analysis_type == "boundary_spanning":
            _analyze_boundary_spanning(
                ax,
                G,
                unique_clusters,
                nodes_by_cluster,
                nodes_by_layer,
                cluster_colors,
                layers,
                visible_layer_indices,
            )
        elif analysis_type == "module_conservation":
            _analyze_module_conservation(
                ax,
                G,
                unique_clusters,
                nodes_by_cluster,
                nodes_by_layer,
                cluster_colors,
                layers,
                visible_layer_indices,
            )
        elif analysis_type == "functional_enrichment":
            _analyze_functional_enrichment(
                ax,
                G,
                unique_clusters,
                nodes_by_cluster,
                nodes_by_layer,
                cluster_colors,
                layers,
                visible_layer_indices,
            )
        elif analysis_type == "disease_association":
            _analyze_disease_association(
                ax,
                G,
                unique_clusters,
                nodes_by_cluster,
                nodes_by_layer,
                cluster_colors,
                layers,
                visible_layer_indices,
            )
        elif analysis_type == "co_expression_correlation":
            _analyze_co_expression_correlation(
                ax,
                G,
                unique_clusters,
                nodes_by_cluster,
                nodes_by_layer,
                cluster_colors,
                layers,
                visible_layer_indices,
            )
        elif analysis_type == "pathway_alignment":
            _analyze_pathway_alignment(
                ax,
                G,
                unique_clusters,
                nodes_by_cluster,
                nodes_by_layer,
                cluster_colors,
                layers,
                visible_layer_indices,
            )
        else:
            ax.text(
                0.5,
                0.5,
                f"Unknown analysis type: {analysis_type}",
                ha="center",
                va="center",
            )

        logging.info(
            f"Successfully created cluster bridging analysis for {analysis_type}"
        )

    except Exception as e:
        logging.error(f"Error creating cluster bridging analysis: {str(e)}")
        ax.clear()
        ax.text(
            0.5,
            0.5,
            f"Error creating visualization: {str(e)}",
            ha="center",
            va="center",
        )


def _analyze_bridge_score(
    ax,
    G,
    unique_clusters,
    nodes_by_cluster,
    nodes_by_layer,
    cluster_colors,
    layers,
    visible_layer_indices,
):
    """
    Analyze and visualize bridge scores for clusters.
    Bridge score measures how well a cluster connects different layers.
    """
    # Calculate bridge scores for each cluster
    bridge_scores = {}
    for cluster in unique_clusters:
        cluster_nodes = nodes_by_cluster[cluster]
        if not cluster_nodes:
            bridge_scores[cluster] = 0
            continue

        # Count interlayer vs. intralayer connections
        interlayer_connections = 0
        total_connections = 0

        for node in cluster_nodes:
            if node not in G.nodes:
                continue

            # Extract layer from the duplicated node name (format: <layer>_<node>)
            node_layer = G.nodes[node]["layer"]

            for neighbor in G.neighbors(node):
                if (
                    G.nodes[neighbor]["cluster"] == cluster
                ):  # Only count connections within the same cluster
                    neighbor_layer = G.nodes[neighbor]["layer"]
                    total_connections += 1
                    if node_layer != neighbor_layer:
                        interlayer_connections += 1

        # Bridge score is the ratio of interlayer to total connections
        bridge_scores[cluster] = interlayer_connections / max(total_connections, 1)

    # Sort clusters by bridge score
    sorted_clusters = sorted(
        unique_clusters, key=lambda c: bridge_scores[c], reverse=True
    )

    # Create bar chart
    y_pos = np.arange(len(sorted_clusters))
    scores = [bridge_scores[c] for c in sorted_clusters]
    colors = [cluster_colors.get(c, "#CCCCCC") for c in sorted_clusters]

    bars = ax.barh(y_pos, scores, color=colors)

    # Add labels and formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Cluster {c}" for c in sorted_clusters])
    ax.set_xlabel("Bridge Score (higher = better bridging)")
    ax.set_xlim(0, max(scores) * 1.1 if scores else 0.1)

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            ha="left",
            va="center",
        )

def _analyze_flow_efficiency(
    ax,
    G,
    unique_clusters,
    nodes_by_cluster,
    nodes_by_layer,
    cluster_colors,
    layers,
    visible_layer_indices,
):
    """
    Analyze and visualize flow efficiency for clusters between layers.
    Flow efficiency measures how efficiently information can flow through clusters between layers.
    """
    # Create a matrix to store flow efficiency between layers through each cluster
    num_layers = len(visible_layer_indices)
    flow_matrix = np.zeros((len(unique_clusters), num_layers, num_layers))

    # For each cluster, calculate average shortest path length between layers
    for c_idx, cluster in enumerate(unique_clusters):
        cluster_nodes = nodes_by_cluster[cluster]

        # Skip clusters with no nodes
        if not cluster_nodes:
            continue

        # Create a subgraph for this cluster
        subgraph = G.subgraph(cluster_nodes)

        # For each pair of layers, calculate average shortest path length
        for i, layer1 in enumerate(visible_layer_indices):
            # Get nodes in this cluster and layer (format: <layer>_<node>)
            nodes1 = [n for n in cluster_nodes if G.nodes[n]["layer"] == layer1]

            for j, layer2 in enumerate(visible_layer_indices):
                if i == j:  # Same layer, set to 1.0 (perfect efficiency)
                    flow_matrix[c_idx, i, j] = 1.0
                    continue

                # Get nodes in this cluster and layer
                nodes2 = [n for n in cluster_nodes if G.nodes[n]["layer"] == layer2]

                # Skip if either layer has no nodes in this cluster
                if not nodes1 or not nodes2:
                    flow_matrix[c_idx, i, j] = 0
                    continue

                # Calculate average shortest path length
                path_lengths = []
                for n1 in nodes1:
                    for n2 in nodes2:
                        try:
                            path_length = nx.shortest_path_length(subgraph, n1, n2)
                            path_lengths.append(path_length)
                        except nx.NetworkXNoPath:
                            # No path exists
                            pass

                # Calculate flow efficiency as inverse of average path length
                if path_lengths:
                    avg_path_length = sum(path_lengths) / len(path_lengths)
                    flow_matrix[c_idx, i, j] = 1.0 / max(avg_path_length, 1.0)
                else:
                    flow_matrix[c_idx, i, j] = 0

    # Find the best cluster for each layer pair
    best_clusters = np.zeros((num_layers, num_layers), dtype=int)
    for i in range(num_layers):
        for j in range(num_layers):
            if i != j:
                best_clusters[i, j] = np.argmax(flow_matrix[:, i, j])

    # Create a heatmap of the best flow efficiency between layers
    best_flow = np.zeros((num_layers, num_layers))
    for i in range(num_layers):
        for j in range(num_layers):
            best_flow[i, j] = flow_matrix[best_clusters[i, j], i, j]

    # Create the heatmap
    im = ax.imshow(best_flow, cmap="viridis")

    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_label("Flow Efficiency (higher = better)")

    # Add labels
    layer_names = [layers[idx] for idx in visible_layer_indices]
    ax.set_xticks(np.arange(num_layers))
    ax.set_yticks(np.arange(num_layers))
    ax.set_xticklabels(layer_names)
    ax.set_yticklabels(layer_names)

    # Rotate x labels
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_rotation_mode("anchor")

    # Add annotations showing the best cluster and efficiency value
    for i in range(num_layers):
        for j in range(num_layers):
            if i != j:  # Skip diagonal
                best_cluster = unique_clusters[best_clusters[i, j]]
                text = ax.text(
                    j,
                    i,
                    f"C{best_cluster}\n{best_flow[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if best_flow[i, j] > 0.5 else "black",
                )
            else:
                ax.text(j, i, "—", ha="center", va="center")



def _analyze_layer_span(
    ax,
    G,
    unique_clusters,
    nodes_by_cluster,
    nodes_by_layer,
    cluster_colors,
    layers,
    visible_layer_indices,
    layer_colors,
):
    """
    Analyze and visualize how clusters span across different layers.
    This shows the distribution of cluster nodes across layers.
    """
    logging.info("Analyzing layer span for clusters")

    # Calculate layer distribution for each cluster
    layer_distribution = {}
    for cluster in unique_clusters:
        cluster_nodes = nodes_by_cluster[cluster]
        if not cluster_nodes:
            continue

        # Count nodes by layer
        layer_counts = Counter()
        for node in cluster_nodes:
            if node in G.nodes:
                layer_idx = G.nodes[node]["layer"]
                if layer_idx in visible_layer_indices:
                    layer_counts[layer_idx] += 1

        # Skip clusters with no nodes in visible layers
        if sum(layer_counts.values()) == 0:
            continue

        # Calculate percentage distribution
        total_nodes = sum(layer_counts.values())
        distribution = {
            layer: count / total_nodes for layer, count in layer_counts.items()
        }
        layer_distribution[cluster] = distribution

    # Skip if no clusters have layer distribution
    if not layer_distribution:
        ax.text(
            0.5,
            0.5,
            "No clusters with nodes in visible layers",
            ha="center",
            va="center",
        )
        return

    # Sort clusters by number of layers they span
    cluster_span = {c: len(dist) for c, dist in layer_distribution.items()}
    sorted_clusters = sorted(
        layer_distribution.keys(), key=lambda c: cluster_span[c], reverse=True
    )

    # Create stacked bar chart
    x_pos = np.arange(len(sorted_clusters))
    bottoms = np.zeros(len(sorted_clusters))

    # Get layer names for legend
    layer_names = [layers[idx] for idx in visible_layer_indices]

    # Create bars for each layer
    bars = []
    for layer_idx in visible_layer_indices:
        layer_values = []
        for cluster in sorted_clusters:
            dist = layer_distribution[cluster]
            layer_values.append(dist.get(layer_idx, 0))

        layer_color = layer_colors.get(layer_idx, "#CCCCCC")
        bar = ax.bar(
            x_pos,
            layer_values,
            bottom=bottoms,
            color=layer_color,
            label=layers[layer_idx],
        )
        bars.append(bar)
        bottoms += layer_values

    # Add labels and formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"C{c}" for c in sorted_clusters])
    ax.set_ylabel("Proportion of Cluster Nodes")
    ax.set_ylim(0, 1.0)

    # Add legend with smaller font size
    ax.legend(title="Layers", loc="upper right", fontsize="small")

    # Add span values above bars
    for i, cluster in enumerate(sorted_clusters):
        span = cluster_span[cluster]
        ax.text(i, 1.02, f"Span: {span}", ha="center", va="bottom", fontsize=9)




def _analyze_centrality_distribution(
    ax,
    G,
    unique_clusters,
    nodes_by_cluster,
    nodes_by_layer,
    cluster_colors,
    layers,
    visible_layer_indices,
):
    """
    Analyze and visualize the distribution of different centrality measures across clusters.
    This helps identify which clusters serve as central hubs in the network.
    """
    logging.info("Analyzing centrality distribution across clusters")

    # Calculate betweenness centrality
    betweenness = nx.betweenness_centrality(G)

    # Create a matrix to store betweenness centrality by cluster and layer
    cluster_layer_centrality = np.zeros(
        (len(unique_clusters), len(visible_layer_indices))
    )

    # Track node counts for each cluster-layer combination
    node_counts = np.zeros((len(unique_clusters), len(visible_layer_indices)))

    # Group centrality measures by cluster and layer
    for node, bc in betweenness.items():
        if node in G.nodes:
            cluster = G.nodes[node]["cluster"]
            layer = G.nodes[node]["layer"]

            # Get indices for the matrix
            try:
                cluster_idx = unique_clusters.index(cluster)
                layer_idx = visible_layer_indices.index(layer)

                # Add to the centrality matrix
                cluster_layer_centrality[cluster_idx, layer_idx] += bc
                node_counts[cluster_idx, layer_idx] += 1
            except (ValueError, IndexError):
                # Skip if cluster or layer not found in our lists
                continue

    # Calculate average betweenness centrality for each cluster-layer combination
    # Avoid division by zero by using np.divide with a default value
    avg_centrality = np.divide(
        cluster_layer_centrality,
        node_counts,
        out=np.zeros_like(cluster_layer_centrality),
        where=node_counts > 0,
    )

    # Calculate marginal sums (average across layers/clusters)
    cluster_avg = np.mean(
        avg_centrality, axis=1
    )  # Average across layers for each cluster
    layer_avg = np.mean(
        avg_centrality, axis=0
    )  # Average across clusters for each layer

    # Sort clusters by their average centrality
    sorted_cluster_indices = np.argsort(cluster_avg)[::-1]
    sorted_clusters = [unique_clusters[i] for i in sorted_cluster_indices]
    sorted_centrality = avg_centrality[sorted_cluster_indices, :]
    sorted_cluster_avg = cluster_avg[sorted_cluster_indices]

    # Create a figure with GridSpec for the heatmap and marginal bar charts
    ax.clear()
    gs = GridSpec(
        4,
        4,
        figure=ax.figure,
        left=0.12,
        right=0.95,
        top=0.95,
        bottom=0.12,
        wspace=0.05,
        hspace=0.05,
        width_ratios=[0.25, 0.05, 0.65, 0.05],
        height_ratios=[0.05, 0.65, 0.05, 0.25],
    )

    # Create axes for the heatmap and marginal bar charts
    ax_heatmap = ax.figure.add_subplot(gs[1, 2])
    ax_cluster_bar = ax.figure.add_subplot(gs[1, 0], sharey=ax_heatmap)
    ax_layer_bar = ax.figure.add_subplot(gs[3, 2], sharex=ax_heatmap)

    # Create the heatmap
    im = ax_heatmap.imshow(sorted_centrality, cmap="viridis", aspect="auto")

    # Add colorbar
    cbar_ax = ax.figure.add_subplot(gs[1, 3])
    cbar = ax.figure.colorbar(im, cax=cbar_ax)
    cbar.set_label('Betweenness Centrality', fontsize=10)

    # Add labels to the heatmap
    ax_heatmap.set_xticks(np.arange(len(visible_layer_indices)))
    ax_heatmap.set_yticks(np.arange(len(sorted_clusters)))

    # Get layer names for the x-axis
    layer_names = [layers[idx] for idx in visible_layer_indices]
    ax_heatmap.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=10)
    ax_heatmap.set_yticklabels([f"Cluster {c}" for c in sorted_clusters], fontsize=10)
    
    # Add a title to the heatmap axis
    ax_heatmap.set_title("Centrality by Cluster and Layer", fontsize=11, pad=10)

    # Add cluster bar chart (left margin)
    cluster_colors_sorted = [cluster_colors.get(c, "#CCCCCC") for c in sorted_clusters]
    bars_cluster = ax_cluster_bar.barh(
        np.arange(len(sorted_clusters)),
        sorted_cluster_avg,
        color=cluster_colors_sorted,
        height=0.8,
    )
    ax_cluster_bar.set_xlabel('Avg. Centrality', fontsize=10)
    ax_cluster_bar.invert_xaxis()  # Invert x-axis to point bars toward the heatmap
    
    # Add cluster labels to the left bar chart
    ax_cluster_bar.set_yticks(np.arange(len(sorted_clusters)))
    ax_cluster_bar.set_yticklabels([f"C{c}" for c in sorted_clusters], fontsize=10)

    # Add layer bar chart (bottom margin)
    bars_layer = ax_layer_bar.bar(
        np.arange(len(visible_layer_indices)), layer_avg, color="skyblue", width=0.8
    )
    ax_layer_bar.set_ylabel('Avg. Centrality', fontsize=10)
    
    # Add layer labels to the bottom bar chart
    ax_layer_bar.set_xticks(np.arange(len(visible_layer_indices)))
    ax_layer_bar.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=10)

    # Turn off the original axis
    ax.set_axis_off()

    # Add annotations to the heatmap for high centrality values
    threshold = np.percentile(
        sorted_centrality[sorted_centrality > 0], 75
    )  # Show top 25% values
    for i in range(len(sorted_clusters)):
        for j in range(len(visible_layer_indices)):
            val = sorted_centrality[i, j]
            if val > threshold:
                text_color = (
                    "white"
                    if val > np.percentile(sorted_centrality[sorted_centrality > 0], 90)
                    else "black"
                )
                ax_heatmap.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )


def _analyze_cluster_cohesion(
    ax,
    G,
    unique_clusters,
    nodes_by_cluster,
    nodes_by_layer,
    cluster_colors,
    layers,
    visible_layer_indices,
):
    """
    Analyze and visualize the cohesion of clusters within and between layers.
    Cohesion measures how tightly connected nodes are within a cluster.
    """
    logging.info("Analyzing cluster cohesion within and between layers")

    # Calculate cohesion metrics for each cluster
    intra_cohesion = {}  # Within-layer cohesion
    inter_cohesion = {}  # Between-layer cohesion

    for cluster in unique_clusters:
        cluster_nodes = nodes_by_cluster[cluster]
        if not cluster_nodes:
            intra_cohesion[cluster] = 0
            inter_cohesion[cluster] = 0
            continue

        # Create a subgraph for this cluster
        subgraph = G.subgraph(cluster_nodes)

        # Calculate within-layer cohesion (average clustering coefficient within layers)
        layer_cohesion = {}
        for layer_idx in visible_layer_indices:
            # Get nodes in this layer and cluster
            layer_nodes = [n for n in cluster_nodes if G.nodes[n]["layer"] == layer_idx]

            if len(layer_nodes) < 3:  # Need at least 3 nodes for clustering coefficient
                continue

            # Create a subgraph for this layer
            layer_subgraph = subgraph.subgraph(layer_nodes)

            # Calculate average clustering coefficient
            try:
                clustering = nx.average_clustering(layer_subgraph)
                layer_cohesion[layer_idx] = clustering
            except:
                layer_cohesion[layer_idx] = 0

        # Average within-layer cohesion across all layers
        intra_cohesion[cluster] = (
            np.mean(list(layer_cohesion.values())) if layer_cohesion else 0
        )

        # Calculate between-layer cohesion (ratio of interlayer to total edges)
        interlayer_edges = 0
        total_edges = 0

        for u, v in subgraph.edges():
            total_edges += 1
            u_layer = G.nodes[u]["layer"]
            v_layer = G.nodes[v]["layer"]
            if u_layer != v_layer:
                interlayer_edges += 1

        inter_cohesion[cluster] = interlayer_edges / max(total_edges, 1)

    # Create a bar chart of combined cohesion score
    # Calculate combined score as weighted average of intra and inter cohesion
    combined_scores = {}
    for cluster in unique_clusters:
        intra = intra_cohesion.get(cluster, 0)
        inter = inter_cohesion.get(cluster, 0)
        # Weight inter-layer cohesion more heavily as it's more important for bridging
        combined_scores[cluster] = 0.4 * intra + 0.6 * inter

    # Sort clusters by combined score
    sorted_clusters = sorted(
        unique_clusters, key=lambda c: combined_scores[c], reverse=True
    )

    # Create bar chart
    y_pos = np.arange(len(sorted_clusters))
    values = [combined_scores[c] for c in sorted_clusters]
    colors = [cluster_colors.get(c, "#CCCCCC") for c in sorted_clusters]

    bars = ax.barh(y_pos, values, color=colors)

    # Add labels and formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Cluster {c}" for c in sorted_clusters])
    ax.set_xlabel("Cohesion Score")
    ax.set_title("Cluster Cohesion Analysis")

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        intra = intra_cohesion.get(sorted_clusters[i], 0)
        inter = inter_cohesion.get(sorted_clusters[i], 0)
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f} (I:{intra:.2f}, E:{inter:.2f})",
            ha="left",
            va="center",
            fontsize=8,
        )



def _analyze_information_flow(
    ax,
    G,
    unique_clusters,
    nodes_by_cluster,
    nodes_by_layer,
    cluster_colors,
    layers,
    visible_layer_indices,
):
    """
    Simulate information flow through the network to identify key bridging clusters.
    This uses a diffusion model to see how information spreads across layers through clusters.
    """
    logging.info("Analyzing information flow through clusters")

    # Create a matrix to store information flow between layers through each cluster
    num_layers = len(visible_layer_indices)
    flow_matrix = np.zeros((len(unique_clusters), num_layers))

    # For each layer, simulate information flow starting from that layer
    for start_idx, start_layer in enumerate(visible_layer_indices):
        # For each cluster, measure how well it transmits information from the start layer
        for c_idx, cluster in enumerate(unique_clusters):
            cluster_nodes = nodes_by_cluster[cluster]

            # Skip clusters with no nodes
            if not cluster_nodes:
                continue

            # Get nodes in the start layer for this cluster
            start_nodes = [
                n for n in cluster_nodes if G.nodes[n]["layer"] == start_layer
            ]

            if not start_nodes:
                continue

            # Create a subgraph for this cluster
            subgraph = G.subgraph(cluster_nodes)

            # Initialize information values (1.0 for start nodes, 0.0 for others)
            info_values = {
                node: 1.0 if node in start_nodes else 0.0 for node in cluster_nodes
            }

            # Simulate diffusion for a few steps
            num_steps = 3
            damping = 0.85

            for _ in range(num_steps):
                new_values = {}

                # For each node, update its value based on neighbors
                for node in cluster_nodes:
                    if node not in subgraph:
                        continue

                    # Get neighbors
                    neighbors = list(subgraph.neighbors(node))

                    if not neighbors:
                        new_values[node] = info_values[node]
                        continue

                    # Calculate new value as weighted average of neighbors
                    neighbor_sum = sum(info_values[neigh] for neigh in neighbors)
                    new_value = (1 - damping) * info_values[
                        node
                    ] + damping * neighbor_sum / len(neighbors)
                    new_values[node] = new_value

                # Update all values at once
                info_values = new_values

            # Measure information flow to each layer
            for end_idx, end_layer in enumerate(visible_layer_indices):
                if end_layer == start_layer:
                    continue

                # Get nodes in the end layer for this cluster
                end_nodes = [
                    n for n in cluster_nodes if G.nodes[n]["layer"] == end_layer
                ]

                if not end_nodes:
                    continue

                # Calculate average information value at end nodes
                avg_info = sum(info_values[node] for node in end_nodes) / len(end_nodes)

                # Store in the flow matrix
                flow_matrix[c_idx, end_idx] += avg_info

    # Normalize the flow matrix by the number of source layers
    flow_matrix = flow_matrix / num_layers

    # Calculate overall flow score for each cluster (average across destination layers)
    flow_scores = np.mean(flow_matrix, axis=1)

    # Clear the original axis and create a new figure with a polar projection
    ax.clear()

    # Create a bar chart of flow scores instead of a radar chart
    # Sort clusters by flow score
    sorted_indices = np.argsort(flow_scores)[::-1]
    sorted_clusters = [unique_clusters[i] for i in sorted_indices]
    sorted_scores = [flow_scores[i] for i in sorted_indices]

    # Create bar chart
    y_pos = np.arange(len(sorted_clusters))
    colors = [cluster_colors.get(c, "#CCCCCC") for c in sorted_clusters]

    bars = ax.barh(y_pos, sorted_scores, color=colors)

    # Add labels and formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Cluster {c}" for c in sorted_clusters])
    ax.set_xlabel("Information Flow Score")
    ax.set_title("Information Flow Through Clusters")

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            ha="left",
            va="center",
        )




def _analyze_structural_holes(
    ax,
    G,
    unique_clusters,
    nodes_by_cluster,
    nodes_by_layer,
    cluster_colors,
    layers,
    visible_layer_indices,
):
    """
    Analyze and visualize structural holes in the network.
    Structural holes are gaps in the network that can be exploited for information or control advantages.
    This analysis identifies clusters that bridge structural holes between layers.
    """
    logging.info("Analyzing structural holes between layers")

    # Calculate constraint for each node (Burt's measure of structural holes)
    # Lower constraint means more structural holes (more brokerage opportunities)
    try:
        node_constraint = nx.constraint(G)
    except:
        # If constraint calculation fails, use a simpler approach
        node_constraint = {}
        for node in G.nodes():
            neighbors = set(G.neighbors(node))
            if len(neighbors) <= 1:
                node_constraint[node] = 1.0  # High constraint for isolated nodes
                continue
                
            # Calculate constraint manually
            constraint = 0
            for neighbor in neighbors:
                # Direct investment
                p_ij = 1.0 / len(neighbors)
                
                # Indirect investment
                p_iq_qj = 0
                for q in neighbors:
                    if q == neighbor:
                        continue
                    if G.has_edge(q, neighbor):
                        p_iq = 1.0 / len(neighbors)
                        q_neighbors = set(G.neighbors(q))
                        if len(q_neighbors) > 0:
                            p_qj = 1.0 / len(q_neighbors)
                            p_iq_qj += p_iq * p_qj
                
                constraint += (p_ij + p_iq_qj) ** 2
            
            node_constraint[node] = constraint

    # Calculate average constraint by cluster and layer
    layer_cluster_constraint = np.zeros((len(visible_layer_indices), len(unique_clusters)))
    
    for i, layer_idx in enumerate(visible_layer_indices):
        for j, cluster in enumerate(unique_clusters):
            # Get nodes in this layer and cluster
            layer_nodes = nodes_by_layer[layer_idx]
            cluster_nodes = nodes_by_cluster[cluster]
            
            # Find nodes that are in both this layer and cluster
            nodes = [n for n in layer_nodes if n in cluster_nodes]
            
            if nodes:
                # Calculate average constraint (lower is better - more structural holes)
                avg_constraint = sum(node_constraint.get(n, 1.0) for n in nodes) / len(nodes)
                # Invert so higher values mean more structural holes
                layer_cluster_constraint[i, j] = 1.0 - min(avg_constraint, 1.0)
            else:
                layer_cluster_constraint[i, j] = 0

    # Create a heatmap
    im = ax.imshow(layer_cluster_constraint, cmap="viridis")
    
    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_label("Structural Hole Advantage (higher = more advantage)")
    
    # Add labels
    ax.set_xticks(np.arange(len(unique_clusters)))
    ax.set_yticks(np.arange(len(visible_layer_indices)))
    ax.set_xticklabels([f"C{c}" for c in unique_clusters])
    ax.set_yticklabels([layers[idx] for idx in visible_layer_indices])
    
    # Rotate x labels
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    
    # Add annotations
    for i in range(len(visible_layer_indices)):
        for j in range(len(unique_clusters)):
            text = ax.text(j, i, f"{layer_cluster_constraint[i, j]:.2f}",
                          ha="center", va="center", 
                          color="white" if layer_cluster_constraint[i, j] < 0.5 else "black")
    



def _analyze_cross_layer_influence(
    ax,
    G,
    unique_clusters,
    nodes_by_cluster,
    nodes_by_layer,
    cluster_colors,
    layers,
    visible_layer_indices,
):
    """
    Analyze and visualize cross-layer influence of clusters.
    This measures how much influence each cluster has across different layers.
    """
    logging.info("Analyzing cross-layer influence of clusters")

    # Calculate eigenvector centrality for all nodes
    try:
        centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        # If eigenvector centrality fails, use degree centrality as fallback
        centrality = nx.degree_centrality(G)
    
    # Calculate influence scores for each cluster on each layer
    influence_matrix = np.zeros((len(unique_clusters), len(visible_layer_indices)))
    
    for i, cluster in enumerate(unique_clusters):
        cluster_nodes = nodes_by_cluster[cluster]
        
        if not cluster_nodes:
            continue
            
        # Get centrality of cluster nodes
        cluster_centrality = {node: centrality.get(node, 0) for node in cluster_nodes}
        
        # Calculate influence on each layer
        for j, layer_idx in enumerate(visible_layer_indices):
            # Get nodes in this layer
            layer_nodes = nodes_by_layer[layer_idx]
            
            # Skip if no nodes in this layer
            if not layer_nodes:
                continue
                
            # Calculate influence as the sum of centrality of cluster nodes connected to this layer
            influence = 0
            for node in cluster_nodes:
                # Skip nodes not in the graph
                if node not in G:
                    continue
                    
                # Check if this node has connections to the current layer
                for neighbor in G.neighbors(node):
                    if neighbor in layer_nodes and G.nodes[neighbor]["layer"] == layer_idx:
                        influence += cluster_centrality[node]
                        break
            
            # Normalize by number of nodes in cluster
            if cluster_nodes:
                influence_matrix[i, j] = influence / len(cluster_nodes)
    
    # Normalize the influence matrix
    max_influence = np.max(influence_matrix) if np.max(influence_matrix) > 0 else 1
    influence_matrix = influence_matrix / max_influence
    
    # Create a heatmap
    im = ax.imshow(influence_matrix, cmap="YlOrRd")
    
    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_label("Normalized Influence Score")
    
    # Add labels
    ax.set_yticks(np.arange(len(unique_clusters)))
    ax.set_xticks(np.arange(len(visible_layer_indices)))
    ax.set_yticklabels([f"C{c}" for c in unique_clusters])
    ax.set_xticklabels([layers[idx] for idx in visible_layer_indices])
    
    # Add title and labels
    ax.set_xlabel("Layers")
    ax.set_ylabel("Clusters")
    
    # Rotate x labels
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    
    # Add annotations
    for i in range(len(unique_clusters)):
        for j in range(len(visible_layer_indices)):
            text = ax.text(j, i, f"{influence_matrix[i, j]:.2f}",
                          ha="center", va="center", 
                          color="black" if influence_matrix[i, j] < 0.5 else "white")
    



def _analyze_cluster_resilience(
    ax,
    G,
    unique_clusters,
    nodes_by_cluster,
    nodes_by_layer,
    cluster_colors,
    layers,
    visible_layer_indices,
):
    """
    Analyze and visualize the resilience of clusters to node removal.
    This measures how well a cluster maintains connectivity when nodes are removed.
    """
    logging.info("Analyzing cluster resilience")

    # Calculate resilience scores for each cluster
    resilience_scores = {}
    cluster_layer_resilience = {}
    
    for cluster in unique_clusters:
        cluster_nodes = nodes_by_cluster[cluster]
        
        if not cluster_nodes or len(cluster_nodes) < 3:
            resilience_scores[cluster] = 0
            cluster_layer_resilience[cluster] = {layer_idx: 0 for layer_idx in visible_layer_indices}
            continue
        
        # Create a subgraph for this cluster
        subgraph = G.subgraph(cluster_nodes)
        
        # Calculate initial connectivity (using average shortest path length)
        try:
            initial_connectivity = nx.average_shortest_path_length(subgraph)
        except nx.NetworkXError:
            # Graph is not connected
            initial_connectivity = float('inf')
            
        if initial_connectivity == float('inf'):
            # If graph is not connected, use a different measure
            initial_connectivity = len(list(nx.connected_components(subgraph)))
        
        # Calculate resilience by layer
        layer_resilience = {}
        
        for layer_idx in visible_layer_indices:
            # Get nodes in this layer and cluster
            layer_nodes = [n for n in cluster_nodes if G.nodes[n]["layer"] == layer_idx]
            
            if not layer_nodes:
                layer_resilience[layer_idx] = 0
                continue
            
            # Calculate resilience by removing nodes from this layer
            resilience_sum = 0
            
            # For each node in this layer, remove it and measure connectivity change
            for node in layer_nodes:
                temp_graph = subgraph.copy()
                if node in temp_graph:
                    temp_graph.remove_node(node)
                
                # Skip if graph becomes too small
                if len(temp_graph) < 2:
                    continue
                
                try:
                    new_connectivity = nx.average_shortest_path_length(temp_graph)
                except nx.NetworkXError:
                    # Graph is not connected after node removal
                    new_connectivity = float('inf')
                    
                if new_connectivity == float('inf'):
                    # If graph is not connected, use component count
                    new_connectivity = len(list(nx.connected_components(temp_graph)))
                
                # Calculate resilience as the inverse of connectivity change
                # (smaller change = higher resilience)
                if initial_connectivity == float('inf'):
                    # If initial graph was not connected
                    change = abs(new_connectivity - initial_connectivity) / max(initial_connectivity, 1)
                else:
                    change = abs(new_connectivity - initial_connectivity) / initial_connectivity
                
                resilience = 1.0 / (1.0 + change)
                resilience_sum += resilience
            
            # Average resilience for this layer
            if layer_nodes:
                layer_resilience[layer_idx] = resilience_sum / len(layer_nodes)
            else:
                layer_resilience[layer_idx] = 0
        
        # Overall resilience for this cluster
        if visible_layer_indices:
            resilience_scores[cluster] = sum(layer_resilience.values()) / len(visible_layer_indices)
        else:
            resilience_scores[cluster] = 0
            
        cluster_layer_resilience[cluster] = layer_resilience
    
    # Create a heatmap of resilience by cluster and layer
    resilience_matrix = np.zeros((len(unique_clusters), len(visible_layer_indices)))
    
    for i, cluster in enumerate(unique_clusters):
        for j, layer_idx in enumerate(visible_layer_indices):
            resilience_matrix[i, j] = cluster_layer_resilience[cluster].get(layer_idx, 0)
    
    # Create the heatmap
    im = ax.imshow(resilience_matrix, cmap="RdYlGn")
    
    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_label("Resilience Score (higher = more resilient)")
    
    # Add labels
    ax.set_yticks(np.arange(len(unique_clusters)))
    ax.set_xticks(np.arange(len(visible_layer_indices)))
    ax.set_yticklabels([f"C{c}" for c in unique_clusters])
    ax.set_xticklabels([layers[idx] for idx in visible_layer_indices])
    
    # Add title and labels
    ax.set_xlabel("Layers")
    ax.set_ylabel("Clusters")
    
    # Rotate x labels
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    
    # Add annotations
    for i in range(len(unique_clusters)):
        for j in range(len(visible_layer_indices)):
            text = ax.text(j, i, f"{resilience_matrix[i, j]:.2f}",
                          ha="center", va="center", 
                          color="black" if resilience_matrix[i, j] < 0.5 else "white")
    


def _analyze_path_diversity(
    ax,
    G,
    unique_clusters,
    nodes_by_cluster,
    nodes_by_layer,
    cluster_colors,
    layers,
    visible_layer_indices,
):
    """
    Analyze and visualize path diversity through clusters between layers.
    This measures how many different paths exist between layers through each cluster.
    """
    logging.info("Analyzing path diversity between layers")

    # Calculate path diversity for each cluster between layers
    path_diversity = {}
    
    # For each pair of layers, calculate path diversity through each cluster
    layer_pairs = [(i, j) for i in range(len(visible_layer_indices)) 
                  for j in range(i+1, len(visible_layer_indices))]
    
    # Initialize the diversity matrix
    diversity_matrix = np.zeros((len(unique_clusters), len(layer_pairs)))
    
    for cluster_idx, cluster in enumerate(unique_clusters):
        cluster_nodes = nodes_by_cluster[cluster]
        
        if not cluster_nodes:
            continue
            
        # Create a subgraph for this cluster
        subgraph = G.subgraph(cluster_nodes)
        
        # For each pair of layers, calculate path diversity
        for pair_idx, (i, j) in enumerate(layer_pairs):
            layer_i = visible_layer_indices[i]
            layer_j = visible_layer_indices[j]
            
            # Get nodes in each layer for this cluster
            nodes_i = [n for n in cluster_nodes if G.nodes[n]["layer"] == layer_i]
            nodes_j = [n for n in cluster_nodes if G.nodes[n]["layer"] == layer_j]
            
            if not nodes_i or not nodes_j:
                diversity_matrix[cluster_idx, pair_idx] = 0
                continue
            
            # Calculate average number of paths between layers
            path_count = 0
            pair_count = 0
            
            # Sample node pairs if there are too many
            max_pairs = 10
            if len(nodes_i) * len(nodes_j) > max_pairs:
                import random
                random.seed(42)  # For reproducibility
                sample_i = random.sample(nodes_i, min(5, len(nodes_i)))
                sample_j = random.sample(nodes_j, min(5, len(nodes_j)))
            else:
                sample_i = nodes_i
                sample_j = nodes_j
            
            for node_i in sample_i:
                for node_j in sample_j:
                    try:
                        # Count edge-disjoint paths
                        paths = list(nx.edge_disjoint_paths(subgraph, node_i, node_j))
                        path_count += len(paths)
                        pair_count += 1
                    except nx.NetworkXNoPath:
                        # No path exists
                        pass
            
            # Calculate average path diversity
            if pair_count > 0:
                avg_diversity = path_count / pair_count
                # Normalize to [0, 1] using a sigmoid-like function
                normalized_diversity = 2 / (1 + np.exp(-avg_diversity)) - 1
                diversity_matrix[cluster_idx, pair_idx] = normalized_diversity
            else:
                diversity_matrix[cluster_idx, pair_idx] = 0
    
    # Create a heatmap
    im = ax.imshow(diversity_matrix, cmap="plasma")
    
    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_label("Path Diversity Score (higher = more diverse paths)")
    
    # Add labels
    ax.set_yticks(np.arange(len(unique_clusters)))
    ax.set_yticklabels([f"C{c}" for c in unique_clusters])
    
    # Create labels for layer pairs
    pair_labels = [f"{layers[visible_layer_indices[i]]}↔{layers[visible_layer_indices[j]]}" 
                  for i, j in layer_pairs]
    
    ax.set_xticks(np.arange(len(layer_pairs)))
    ax.set_xticklabels(pair_labels)
    
    # Add title and labels
    ax.set_xlabel("Layer Pairs")
    ax.set_ylabel("Clusters")
    
    # Rotate x labels
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    
    # Add annotations
    for i in range(len(unique_clusters)):
        for j in range(len(layer_pairs)):
            text = ax.text(j, i, f"{diversity_matrix[i, j]:.2f}",
                          ha="center", va="center", 
                          color="white" if diversity_matrix[i, j] < 0.5 else "black")
    


def _analyze_boundary_spanning(
    ax,
    G,
    unique_clusters,
    nodes_by_cluster,
    nodes_by_layer,
    cluster_colors,
    layers,
    visible_layer_indices,
):
    """
    Analyze and visualize boundary spanning capabilities of clusters.
    This measures how effectively clusters connect to other clusters across layer boundaries.
    """
    logging.info("Analyzing boundary spanning capabilities of clusters")

    # Calculate boundary spanning scores
    boundary_scores = np.zeros((len(unique_clusters), len(visible_layer_indices)))
    
    # For each cluster and layer, calculate boundary spanning score
    for i, cluster in enumerate(unique_clusters):
        cluster_nodes = nodes_by_cluster[cluster]
        
        if not cluster_nodes:
            continue
            
        # For each layer, calculate boundary spanning
        for j, layer_idx in enumerate(visible_layer_indices):
            # Get nodes in this layer for this cluster
            layer_nodes = [n for n in cluster_nodes if G.nodes[n]["layer"] == layer_idx]
            
            if not layer_nodes:
                boundary_scores[i, j] = 0
                continue
                
            # Count connections to other clusters from this layer
            external_connections = 0
            total_connections = 0
            
            for node in layer_nodes:
                for neighbor in G.neighbors(node):
                    # Skip if neighbor is not in the graph
                    if neighbor not in G.nodes:
                        continue
                        
                    # Get neighbor's cluster
                    neighbor_cluster = G.nodes[neighbor]["cluster"]
                    
                    # Count connection
                    total_connections += 1
                    
                    # If neighbor is in a different cluster, count as external
                    if neighbor_cluster != cluster:
                        external_connections += 1
            
            # Calculate boundary spanning score
            if total_connections > 0:
                boundary_scores[i, j] = external_connections / total_connections
            else:
                boundary_scores[i, j] = 0
    
    # Create a heatmap
    im = ax.imshow(boundary_scores, cmap="coolwarm")
    
    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_label("Boundary Spanning Score (higher = more external connections)")
    
    # Add labels
    ax.set_yticks(np.arange(len(unique_clusters)))
    ax.set_xticks(np.arange(len(visible_layer_indices)))
    ax.set_yticklabels([f"C{c}" for c in unique_clusters])
    ax.set_xticklabels([layers[idx] for idx in visible_layer_indices])
    
    # Add title and labels
    ax.set_xlabel("Layers")
    ax.set_ylabel("Clusters")
    
    # Rotate x labels
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    
    # Add annotations
    for i in range(len(unique_clusters)):
        for j in range(len(visible_layer_indices)):
            text = ax.text(j, i, f"{boundary_scores[i, j]:.2f}",
                          ha="center", va="center", 
                          color="white" if boundary_scores[i, j] > 0.5 else "black")
    
    # Calculate overall boundary spanning capability for each cluster
    overall_scores = np.mean(boundary_scores, axis=1)
    
    # Add a bar chart below the heatmap showing overall scores
    # Create a new axis for the bar chart
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size="30%", pad=0.6)
    
    # Create the bar chart
    bars = ax2.bar(np.arange(len(unique_clusters)), overall_scores, 
                  color=[cluster_colors.get(c, "#CCCCCC") for c in unique_clusters])
    
    # Add labels
    ax2.set_xticks(np.arange(len(unique_clusters)))
    ax2.set_xticklabels([f"C{c}" for c in unique_clusters])
    ax2.set_ylabel("Overall Score")
    ax2.set_ylim(0, max(overall_scores) * 1.2 if max(overall_scores) > 0 else 0.1)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{overall_scores[i]:.2f}",
                ha='center', va='bottom')
    



def _analyze_co_expression_correlation(
    ax,
    G,
    unique_clusters,
    nodes_by_cluster,
    nodes_by_layer,
    cluster_colors,
    layers,
    visible_layer_indices,
):
    """
    Analyze and visualize co-expression correlation between clusters across layers.
    This simulates co-expression analysis by measuring the correlation of connectivity
    patterns between layers, which can indicate shared gene regulation in biological networks.
    """
    logging.info("Analyzing co-expression correlation across layers")
    logging.info(f"Visible layer indices: {visible_layer_indices}")
    logging.info(f"Layers: {layers}")

    # For co-expression networks, we're particularly interested in comparing
    # the coexMSG, coexKDN, and coexSPL layers with the PPI and HP layers
    
    # Identify co-expression and non-co-expression layers
    coex_layer_indices = []
    other_layer_indices = []
    
    # First check if we have any layers with "coex" in the name
    has_coex_layers = any("coex" in layers[idx].lower() for idx in visible_layer_indices)
    
    if not has_coex_layers:
        # If no explicit coex layers, treat the first half of layers as "expression" layers
        # and the second half as "other" layers for comparison
        midpoint = len(visible_layer_indices) // 2
        if midpoint > 0:
            coex_layer_indices = visible_layer_indices[:midpoint]
            other_layer_indices = visible_layer_indices[midpoint:]
            logging.info(f"No explicit co-expression layers found. Using first {midpoint} layers as expression layers.")
        else:
            # Not enough layers to make a meaningful comparison
            logging.warning("Not enough layers for meaningful comparison")
            ax.text(
                0.5,
                0.5,
                "Co-expression correlation analysis requires at least 2 layers to compare",
                ha="center",
                va="center",
                wrap=True,
            )
            return ax
    else:
        # Use the standard approach of identifying coex layers by name
        for idx in visible_layer_indices:
            layer_name = layers[idx].lower()
            if "coex" in layer_name:
                coex_layer_indices.append(idx)
            else:
                other_layer_indices.append(idx)
    
    logging.info(f"Co-expression layer indices: {coex_layer_indices}")
    logging.info(f"Other layer indices: {other_layer_indices}")
    
    # If we don't have both types of layers, we can't do a meaningful comparison
    if not coex_layer_indices or not other_layer_indices:
        logging.warning("Missing either co-expression or non-co-expression layers")
        ax.text(
            0.5,
            0.5,
            "Co-expression correlation analysis requires both expression and non-expression layers",
            ha="center",
            va="center",
            wrap=True,
        )
        return ax
    
    # Calculate correlation matrix between co-expression and other layers for each cluster
    correlation_matrix = np.zeros((len(unique_clusters), len(coex_layer_indices) * len(other_layer_indices)))
    column_labels = []
    
    # Create column labels for the heatmap
    for coex_idx in coex_layer_indices:
        for other_idx in other_layer_indices:
            column_labels.append(f"{layers[coex_idx]}↔{layers[other_idx]}")
    
    logging.info(f"Number of clusters: {len(unique_clusters)}")
    logging.info(f"Column labels: {column_labels}")
    
    # Track if we've calculated any non-zero correlations
    has_correlations = False
    
    for i, cluster in enumerate(unique_clusters):
        cluster_nodes = nodes_by_cluster[cluster]
        logging.info(f"Cluster {cluster} has {len(cluster_nodes)} nodes")
        
        if not cluster_nodes:
            continue
            
        col_idx = 0
        for coex_idx in coex_layer_indices:
            for other_idx in other_layer_indices:
                # Get nodes in each layer for this cluster
                coex_nodes = [n for n in cluster_nodes if G.nodes.get(n, {}).get("layer") == coex_idx]
                other_nodes = [n for n in cluster_nodes if G.nodes.get(n, {}).get("layer") == other_idx]
                
                logging.info(f"Cluster {cluster}, {layers[coex_idx]}↔{layers[other_idx]}: {len(coex_nodes)} coex nodes, {len(other_nodes)} other nodes")
                
                if not coex_nodes or not other_nodes:
                    logging.info(f"Skipping due to missing nodes in one of the layers")
                    correlation_matrix[i, col_idx] = 0
                    col_idx += 1
                    continue
                
                # Extract original node IDs
                try:
                    coex_original_nodes = [n.split('_', 1)[1] for n in coex_nodes]
                    other_original_nodes = [n.split('_', 1)[1] for n in other_nodes]
                except IndexError:
                    # If node IDs don't follow the expected format, try using the original_id attribute
                    coex_original_nodes = [G.nodes[n].get("original_id", n) for n in coex_nodes]
                    other_original_nodes = [G.nodes[n].get("original_id", n) for n in other_nodes]
                
                # Find common nodes between layers
                common_nodes = set(coex_original_nodes).intersection(set(other_original_nodes))
                logging.info(f"Found {len(common_nodes)} common nodes between layers")
                
                if len(common_nodes) < 2:
                    logging.info(f"Skipping due to insufficient common nodes (need at least 2)")
                    correlation_matrix[i, col_idx] = 0
                    col_idx += 1
                    continue
                
                # For each common node, compare its connectivity pattern in both layers
                node_correlations = []
                
                for node_id in common_nodes:
                    # Find the corresponding nodes in each layer
                    coex_node = None
                    other_node = None
                    
                    for n in coex_nodes:
                        if '_' in n and n.split('_', 1)[1] == node_id:
                            coex_node = n
                            break
                        elif G.nodes[n].get("original_id") == node_id:
                            coex_node = n
                            break
                    
                    for n in other_nodes:
                        if '_' in n and n.split('_', 1)[1] == node_id:
                            other_node = n
                            break
                        elif G.nodes[n].get("original_id") == node_id:
                            other_node = n
                            break
                    
                    if not coex_node or not other_node:
                        continue
                    
                    # Get neighbors in each layer
                    coex_neighbors = set()
                    other_neighbors = set()
                    
                    for neighbor in G.neighbors(coex_node):
                        if G.nodes.get(neighbor, {}).get("layer") == coex_idx:
                            try:
                                coex_neighbors.add(neighbor.split('_', 1)[1])
                            except IndexError:
                                coex_neighbors.add(G.nodes[neighbor].get("original_id", neighbor))
                            
                    for neighbor in G.neighbors(other_node):
                        if G.nodes.get(neighbor, {}).get("layer") == other_idx:
                            try:
                                other_neighbors.add(neighbor.split('_', 1)[1])
                            except IndexError:
                                other_neighbors.add(G.nodes[neighbor].get("original_id", neighbor))
                    
                    # Calculate Jaccard similarity of neighbor sets
                    if coex_neighbors or other_neighbors:
                        intersection = len(coex_neighbors.intersection(other_neighbors))
                        union = len(coex_neighbors.union(other_neighbors))
                        jaccard = intersection / union if union > 0 else 0
                        node_correlations.append(jaccard)
                
                # Calculate average correlation for this layer pair
                if node_correlations:
                    correlation_matrix[i, col_idx] = sum(node_correlations) / len(node_correlations)
                    logging.info(f"Calculated correlation: {correlation_matrix[i, col_idx]:.4f} from {len(node_correlations)} node correlations")
                    if correlation_matrix[i, col_idx] > 0:
                        has_correlations = True
                else:
                    logging.info(f"No node correlations calculated")
                
                col_idx += 1
    
    logging.info(f"Correlation matrix shape: {correlation_matrix.shape}")
    logging.info(f"Correlation matrix max value: {np.max(correlation_matrix)}")
    logging.info(f"Correlation matrix min value: {np.min(correlation_matrix)}")
    
    # If we didn't calculate any non-zero correlations, show a message
    if not has_correlations:
        ax.clear()
        ax.text(
            0.5,
            0.5,
            "No correlations found. This may be because:\n"
            "1. No common nodes between layers\n"
            "2. No connectivity patterns to compare\n"
            "3. No similar connectivity patterns",
            ha="center",
            va="center",
            wrap=True,
        )
        return ax
    
    # Create a heatmap
    im = ax.imshow(correlation_matrix, cmap="coolwarm", vmin=0, vmax=1)
    
    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_label("Co-expression Correlation")
    
    # Add labels
    ax.set_yticks(np.arange(len(unique_clusters)))
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_yticklabels([f"C{c}" for c in unique_clusters])
    ax.set_xticklabels(column_labels)
    
    # Add title and labels
    ax.set_xlabel("Layer Pairs")
    ax.set_ylabel("Clusters")
    ax.set_title("Co-expression Correlation Analysis")
    
    # Rotate x labels
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    
    # Add annotations
    for i in range(len(unique_clusters)):
        for j in range(len(column_labels)):
            text = ax.text(j, i, f"{correlation_matrix[i, j]:.2f}",
                          ha="center", va="center", 
                          color="white" if correlation_matrix[i, j] < 0.5 else "black")
    
    return ax


def _analyze_pathway_alignment(
    ax,
    G,
    unique_clusters,
    nodes_by_cluster,
    nodes_by_layer,
    cluster_colors,
    layers,
    visible_layer_indices,
):
    """
    Analyze and visualize pathway alignment of clusters across layers.
    This simulates pathway analysis by measuring the consistency of cluster structures
    across layers, which can indicate shared biological pathways in disease networks.
    """
    logging.info("Analyzing pathway alignment across layers")

    # For pathway alignment, we want to identify clusters that maintain consistent
    # internal structure across different layers, suggesting they represent coherent
    # biological pathways
    
    # Calculate pathway alignment scores for each cluster
    alignment_scores = np.zeros(len(unique_clusters))
    layer_alignment = np.zeros((len(unique_clusters), len(visible_layer_indices)))
    
    for i, cluster in enumerate(unique_clusters):
        cluster_nodes = nodes_by_cluster[cluster]
        
        if not cluster_nodes:
            continue
            
        # Calculate pathway alignment for this cluster across all layers
        layer_scores = {}
        
        for layer_idx in visible_layer_indices:
            # Get nodes in this layer for this cluster
            layer_nodes = [n for n in cluster_nodes if G.nodes[n]["layer"] == layer_idx]
            
            if len(layer_nodes) < 3:  # Need at least 3 nodes for meaningful analysis
                layer_scores[layer_idx] = 0
                continue
                
            # Create a subgraph for this cluster in this layer
            subgraph = G.subgraph(layer_nodes)
            
            # Calculate metrics that indicate pathway coherence
            
            # 1. Clustering coefficient (higher = more pathway-like)
            try:
                clustering = nx.average_clustering(subgraph)
            except:
                clustering = 0
                
            # 2. Diameter (lower = more pathway-like, but needs normalization)
            try:
                diameter = nx.diameter(subgraph)
                # Normalize diameter (invert and scale to [0,1])
                normalized_diameter = 1.0 / (1.0 + diameter)
            except:
                normalized_diameter = 0
                
            # 3. Edge density (higher = more pathway-like)
            num_edges = subgraph.number_of_edges()
            max_edges = len(layer_nodes) * (len(layer_nodes) - 1) / 2
            density = num_edges / max(max_edges, 1)
            
            # Combine metrics with weights favoring clustering and density
            pathway_score = 0.4 * clustering + 0.2 * normalized_diameter + 0.4 * density
            layer_scores[layer_idx] = pathway_score
            layer_alignment[i, visible_layer_indices.index(layer_idx)] = pathway_score
        
        # Calculate overall alignment as consistency across layers
        if layer_scores:
            # Calculate mean and standard deviation of scores
            values = list(layer_scores.values())
            mean_score = sum(values) / len(values)
            
            # Higher mean = better pathway representation
            # Lower std dev = more consistent across layers
            if len(values) > 1:
                std_dev = np.std(values)
                # Combine mean and consistency (inverse of std dev)
                alignment_scores[i] = mean_score * (1.0 / (1.0 + std_dev))
            else:
                alignment_scores[i] = mean_score
    
    # Clear the axis
    ax.clear()
    
    # Create a figure with two subplots using the existing figure
    fig = ax.figure
    
    # Remove any existing axes from the figure
    for ax_old in fig.axes:
        if ax_old != ax:
            fig.delaxes(ax_old)
    
    # Create a new GridSpec
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3, figure=fig)
    
    # Create new axes
    ax_heatmap = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])
    
    # Create heatmap of layer-specific alignment scores
    im = ax_heatmap.imshow(layer_alignment, cmap="YlGnBu")
    
    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax_heatmap)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = ax_heatmap.figure.colorbar(im, cax=cax)
    cbar.set_label("Pathway Coherence Score")
    
    # Add labels to heatmap
    ax_heatmap.set_yticks(np.arange(len(unique_clusters)))
    ax_heatmap.set_xticks(np.arange(len(visible_layer_indices)))
    ax_heatmap.set_yticklabels([f"C{c}" for c in unique_clusters])
    ax_heatmap.set_xticklabels([layers[idx] for idx in visible_layer_indices])
    
    # Add title and labels to heatmap
    ax_heatmap.set_title("Pathway Coherence by Layer")
    ax_heatmap.set_xlabel("Layers")
    ax_heatmap.set_ylabel("Clusters")
    
    # Rotate x labels
    for label in ax_heatmap.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
        label.set_rotation_mode('anchor')
    
    # Add annotations to heatmap
    for i in range(len(unique_clusters)):
        for j in range(len(visible_layer_indices)):
            text = ax_heatmap.text(j, i, f"{layer_alignment[i, j]:.2f}",
                          ha="center", va="center", 
                          color="black" if layer_alignment[i, j] < 0.6 else "white")
    
    # Create bar chart of overall alignment scores
    sorted_indices = np.argsort(alignment_scores)[::-1]
    sorted_clusters = [unique_clusters[i] for i in sorted_indices]
    sorted_scores = [alignment_scores[i] for i in sorted_indices]
    colors = [cluster_colors.get(c, "#CCCCCC") for c in sorted_clusters]
    
    bars = ax_bar.bar(np.arange(len(sorted_clusters)), sorted_scores, color=colors)
    
    # Add labels to bar chart
    ax_bar.set_xticks(np.arange(len(sorted_clusters)))
    ax_bar.set_xticklabels([f"C{c}" for c in sorted_clusters])
    ax_bar.set_ylabel("Pathway Alignment Score")
    ax_bar.set_title("Overall Pathway Alignment")
    
    # Add value labels to bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.2f}", ha='center', va='bottom')
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Set the main axis invisible but keep it for reference
    ax.set_visible(False)
    
    return ax


def _analyze_functional_enrichment(
    ax,
    G,
    unique_clusters,
    nodes_by_cluster,
    nodes_by_layer,
    cluster_colors,
    layers,
    visible_layer_indices,
):
    """
    Analyze functional enrichment by measuring the density of connections within clusters
    compared to between clusters.
    
    This function calculates internal density (connections within the cluster) and external density
    (connections to other clusters), then computes an enrichment score as the ratio of internal
    to external density.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    G : networkx.Graph
        The graph to analyze
    unique_clusters : list
        List of unique cluster IDs
    nodes_by_cluster : dict
        Dictionary mapping cluster IDs to lists of node IDs
    nodes_by_layer : dict
        Dictionary mapping layer IDs to lists of node IDs
    cluster_colors : dict
        Dictionary mapping cluster IDs to colors
    layers : list
        List of layer names
    visible_layer_indices : list
        List of indices of visible layers
    """
    logging.info("Analyzing functional enrichment across clusters and layers")
    
    # Get visible layers
    visible_layers = [layers[i] for i in visible_layer_indices]
    
    # Initialize arrays for scores
    num_clusters = len(unique_clusters)
    num_layers = len(visible_layers)
    enrichment_scores = np.zeros((num_clusters, num_layers))
    
    # For each cluster, calculate enrichment scores across layers
    for c_idx, cluster in enumerate(unique_clusters):
        cluster_nodes = nodes_by_cluster[cluster]
        
        # For each layer, calculate enrichment score
        for l_idx, layer_idx in enumerate(visible_layer_indices):
            layer = layers[layer_idx]
            
            # Get nodes in this layer
            layer_nodes = [n for n in G.nodes() if G.nodes[n]["layer"] == layer_idx]
            
            # Get nodes in this cluster and layer
            cluster_layer_nodes = [n for n in cluster_nodes if n in layer_nodes]
            
            if len(cluster_layer_nodes) < 2:
                enrichment_scores[c_idx, l_idx] = 0
                continue
            
            # Get other nodes in this layer (not in this cluster)
            other_layer_nodes = [n for n in layer_nodes if n not in cluster_nodes]
            
            # Calculate internal density (connections within the cluster)
            internal_edges = 0
            possible_internal_edges = len(cluster_layer_nodes) * (len(cluster_layer_nodes) - 1) / 2
            
            for i, u in enumerate(cluster_layer_nodes):
                for v in cluster_layer_nodes[i+1:]:
                    if G.has_edge(u, v):
                        internal_edges += 1
            
            internal_density = internal_edges / possible_internal_edges if possible_internal_edges > 0 else 0
            
            # Calculate external density (connections to other clusters)
            external_edges = 0
            possible_external_edges = len(cluster_layer_nodes) * len(other_layer_nodes)
            
            for u in cluster_layer_nodes:
                for v in other_layer_nodes:
                    if G.has_edge(u, v):
                        external_edges += 1
            
            external_density = external_edges / possible_external_edges if possible_external_edges > 0 else 0
            
            # Calculate enrichment score (ratio of internal to external density)
            if external_density > 0:
                enrichment_scores[c_idx, l_idx] = internal_density / external_density
            else:
                # If no external connections, use internal density as score
                enrichment_scores[c_idx, l_idx] = internal_density * 2  # Multiply by 2 to emphasize
    
    # Create heatmap
    ax.clear()
    
    # Cap scores for better visualization
    max_score = 5.0  # Cap at 5x enrichment
    capped_scores = np.minimum(enrichment_scores, max_score)
    
    # Create heatmap
    im = ax.imshow(capped_scores, cmap='YlOrRd', aspect='auto')
    
    # Set labels
    ax.set_xticks(np.arange(num_layers))
    ax.set_yticks(np.arange(num_clusters))
    ax.set_xticklabels([layers[idx] for idx in visible_layer_indices])
    ax.set_yticklabels([f"C{c}" for c in unique_clusters])
    
    # Rotate x labels
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
        label.set_rotation_mode('anchor')
    
    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_label("Functional Enrichment Score")
    
    # Add text annotations
    for i in range(num_clusters):
        for j in range(num_layers):
            if capped_scores[i, j] > 2.5:
                text_color = "white"
            else:
                text_color = "black"
            ax.text(j, i, f"{enrichment_scores[i, j]:.2f}", ha="center", va="center", color=text_color)
    
    # Set title and labels
    ax.set_title("Functional Enrichment Analysis")
    ax.set_xlabel("Layers")
    ax.set_ylabel("Clusters")
    
    return ax


def _analyze_module_conservation(
    ax,
    G,
    unique_clusters,
    nodes_by_cluster,
    nodes_by_layer,
    cluster_colors,
    layers,
    visible_layer_indices,
):
    """
    Analyze how well the topological structure of clusters is preserved across different layers.
    
    This function calculates both node conservation (using Jaccard similarity) and edge conservation
    between layers, combining them into a weighted score. Higher values indicate more consistent
    module structure across layers.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    G : networkx.Graph
        The graph to analyze
    unique_clusters : list
        List of unique cluster IDs
    nodes_by_cluster : dict
        Dictionary mapping cluster IDs to lists of node IDs
    nodes_by_layer : dict
        Dictionary mapping layer IDs to lists of node IDs
    cluster_colors : dict
        Dictionary mapping cluster IDs to colors
    layers : list
        List of layer names
    visible_layer_indices : list
        List of indices of visible layers
    """
    logging.info("Analyzing module conservation across layers")
    
    # Get visible layers
    visible_layers = [layers[i] for i in visible_layer_indices]
    
    # Initialize arrays for scores
    num_clusters = len(unique_clusters)
    num_layers = len(visible_layers)
    conservation_scores = np.zeros((num_clusters, num_layers))
    
    # For each cluster, calculate conservation scores across layers
    for c_idx, cluster in enumerate(unique_clusters):
        cluster_nodes = nodes_by_cluster[cluster]
        
        # For each layer, calculate conservation score
        for l_idx, layer_idx in enumerate(visible_layer_indices):
            # Get nodes in this layer for this cluster
            cluster_layer_nodes = [n for n in cluster_nodes if G.nodes.get(n, {}).get("layer") == layer_idx]
            
            if not cluster_layer_nodes:
                conservation_scores[c_idx, l_idx] = 0
                continue
            
            # Calculate node conservation (Jaccard similarity with other layers)
            node_conservation = 0
            edge_conservation = 0
            comparisons = 0
            
            # Compare with all other layers
            for other_l_idx, other_layer_idx in enumerate(visible_layer_indices):
                if other_l_idx == l_idx:
                    continue
                
                # Get nodes in this cluster and other layer
                cluster_other_layer_nodes = [n for n in cluster_nodes if G.nodes.get(n, {}).get("layer") == other_layer_idx]
                
                if not cluster_other_layer_nodes:
                    continue
                
                # Node conservation: Jaccard similarity
                intersection = set(cluster_layer_nodes).intersection(set(cluster_other_layer_nodes))
                union = set(cluster_layer_nodes).union(set(cluster_other_layer_nodes))
                if union:
                    node_conservation += len(intersection) / len(union)
                
                # Edge conservation: Compare edge patterns
                # Get subgraphs for this layer and other layer
                layer_subgraph = nx.subgraph(G, cluster_layer_nodes)
                other_layer_subgraph = nx.subgraph(G, cluster_other_layer_nodes)
                
                # Get common nodes
                common_nodes = list(intersection)
                if len(common_nodes) > 1:
                    # Get edges between common nodes in both layers
                    layer_edges = set([(u, v) if u < v else (v, u) for u, v in layer_subgraph.edges() 
                                      if u in common_nodes and v in common_nodes])
                    other_layer_edges = set([(u, v) if u < v else (v, u) for u, v in other_layer_subgraph.edges() 
                                            if u in common_nodes and v in common_nodes])
                    
                    # Calculate edge conservation (Jaccard similarity of edges)
                    edge_intersection = layer_edges.intersection(other_layer_edges)
                    edge_union = layer_edges.union(other_layer_edges)
                    if edge_union:
                        edge_conservation += len(edge_intersection) / len(edge_union)
                
                comparisons += 1
            
            # Calculate average conservation scores
            if comparisons > 0:
                node_conservation /= comparisons
                edge_conservation /= comparisons
                
                # Combine node and edge conservation (weighted average)
                conservation_scores[c_idx, l_idx] = 0.6 * node_conservation + 0.4 * edge_conservation
            else:
                conservation_scores[c_idx, l_idx] = 0
    
    # Create heatmap
    ax.clear()
    
    # Normalize scores for better visualization
    max_score = np.max(conservation_scores) if np.max(conservation_scores) > 0 else 1
    normalized_scores = conservation_scores / max_score
    
    # Create heatmap
    im = ax.imshow(normalized_scores, cmap='viridis', aspect='auto')
    
    # Set labels
    ax.set_xticks(np.arange(num_layers))
    ax.set_yticks(np.arange(num_clusters))
    ax.set_xticklabels(visible_layers)
    ax.set_yticklabels([f"C{c}" for c in unique_clusters])
    
    # Rotate x labels
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
        label.set_rotation_mode('anchor')
    
    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_label("Module Conservation Score")
    
    # Add text annotations
    for i in range(num_clusters):
        for j in range(num_layers):
            if normalized_scores[i, j] > 0.5:
                text_color = "white"
            else:
                text_color = "black"
            ax.text(j, i, f"{normalized_scores[i, j]:.2f}", ha="center", va="center", color=text_color)
    
    # Set title and labels
    ax.set_title("Module Conservation Analysis")
    ax.set_xlabel("Layers")
    ax.set_ylabel("Clusters")
    
    return ax


def _analyze_disease_association(
    ax,
    G,
    unique_clusters,
    nodes_by_cluster,
    nodes_by_layer,
    cluster_colors,
    layers,
    visible_layer_indices,
):
    """
    Analyze potential disease relevance based on centrality, clustering, and cross-layer presence.
    
    This function combines three key metrics: centrality (important nodes tend to be disease-associated),
    clustering coefficient (disease genes tend to form modules), and cross-layer presence
    (disease genes tend to be present in multiple data types).
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    G : networkx.Graph
        The graph to analyze
    unique_clusters : list
        List of unique cluster IDs
    nodes_by_cluster : dict
        Dictionary mapping cluster IDs to lists of node IDs
    nodes_by_layer : dict
        Dictionary mapping layer IDs to lists of node IDs
    cluster_colors : dict
        Dictionary mapping cluster IDs to colors
    layers : list
        List of layer names
    visible_layer_indices : list
        List of indices of visible layers
    """
    logging.info("Analyzing disease association potential across clusters and layers")
    
    # Get visible layers
    visible_layers = [layers[i] for i in visible_layer_indices]
    
    # Initialize arrays for scores
    num_clusters = len(unique_clusters)
    num_layers = len(visible_layers)
    disease_scores = np.zeros((num_clusters, num_layers))
    
    # Calculate node centrality for all nodes
    try:
        centrality = nx.betweenness_centrality(G)
    except:
        # Fall back to degree centrality if betweenness fails (e.g., for disconnected graphs)
        centrality = nx.degree_centrality(G)
    
    # Calculate clustering coefficient for all nodes
    try:
        clustering = nx.clustering(G)
    except:
        # Fall back to a simple dictionary if clustering fails
        clustering = {node: 0 for node in G.nodes()}
    
    # For each cluster, calculate disease association scores across layers
    for c_idx, cluster in enumerate(unique_clusters):
        cluster_nodes = nodes_by_cluster[cluster]
        
        # Calculate cross-layer presence for nodes in this cluster
        cross_layer_presence = {}
        for node in cluster_nodes:
            # Count in how many layers this node appears
            layer_count = sum(1 for layer_idx in visible_layer_indices if G.nodes.get(node, {}).get("layer") == layer_idx)
            cross_layer_presence[node] = layer_count / len(visible_layers)
        
        # For each layer, calculate disease association score
        for l_idx, layer_idx in enumerate(visible_layer_indices):
            # Get nodes in this layer for this cluster
            cluster_layer_nodes = [n for n in cluster_nodes if G.nodes.get(n, {}).get("layer") == layer_idx]
            
            if not cluster_layer_nodes:
                disease_scores[c_idx, l_idx] = 0
                continue
            
            # Calculate metrics for nodes in this cluster and layer
            avg_centrality = np.mean([centrality.get(node, 0) for node in cluster_layer_nodes])
            avg_clustering = np.mean([clustering.get(node, 0) for node in cluster_layer_nodes])
            avg_cross_layer = np.mean([cross_layer_presence.get(node, 0) for node in cluster_layer_nodes])
            
            # Normalize metrics to [0, 1] range
            norm_centrality = min(avg_centrality * 10, 1.0)  # Scale up centrality (typically small values)
            norm_clustering = avg_clustering
            norm_cross_layer = avg_cross_layer
            
            # Combine metrics into a disease association score
            # Weight: 40% centrality, 30% clustering, 30% cross-layer presence
            disease_scores[c_idx, l_idx] = (
                0.4 * norm_centrality + 
                0.3 * norm_clustering + 
                0.3 * norm_cross_layer
            )
    
    # Create heatmap
    ax.clear()
    
    # Create heatmap
    im = ax.imshow(disease_scores, cmap='plasma', aspect='auto')
    
    # Set labels
    ax.set_xticks(np.arange(num_layers))
    ax.set_yticks(np.arange(num_clusters))
    ax.set_xticklabels([layers[idx] for idx in visible_layer_indices])
    ax.set_yticklabels([f"C{c}" for c in unique_clusters])
    
    # Rotate x labels
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
        label.set_rotation_mode('anchor')
    
    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_label("Disease Association Score")
    
    # Add text annotations
    for i in range(num_clusters):
        for j in range(num_layers):
            if disease_scores[i, j] > 0.5:
                text_color = "white"
            else:
                text_color = "black"
            ax.text(j, i, f"{disease_scores[i, j]:.2f}", ha="center", va="center", color=text_color)
    
    # Set title and labels
    ax.set_title("Disease Association Analysis")
    ax.set_xlabel("Layers")
    ax.set_ylabel("Clusters")
    
    return ax
