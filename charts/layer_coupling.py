import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform


def calculate_coupling_metric(layer_connections, metric="Edge Density"):
    """Calculate coupling between layers using specified metric"""
    n_layers = layer_connections.shape[0]
    
    # Validate input matrix
    if not isinstance(layer_connections, np.ndarray):
        raise ValueError("layer_connections must be a numpy array")
    if layer_connections.shape[0] != layer_connections.shape[1]:
        raise ValueError("layer_connections must be a square matrix")
    if np.any(layer_connections < 0):
        raise ValueError("layer_connections cannot contain negative values")
    
    # Print input matrix statistics for debugging
    print(f"Input matrix statistics:")
    print(f"- Shape: {layer_connections.shape}")
    print(f"- Range: [{layer_connections.min():.3f}, {layer_connections.max():.3f}]")
    print(f"- Mean: {layer_connections.mean():.3f}")
    print(f"- Unique values: {len(np.unique(layer_connections))}")
        
    coupling_matrix = np.zeros((n_layers, n_layers))

    if metric == "Edge Density":
        # Calculate edge density between layers
        for i in range(n_layers):
            for j in range(n_layers):
                if i == j:
                    # For intralayer coupling, calculate actual density
                    nodes_in_layer = layer_connections[i, i]
                    if nodes_in_layer > 1:
                        max_possible = nodes_in_layer * (nodes_in_layer - 1) / 2
                        coupling_matrix[i, j] = layer_connections[i, i] / max_possible if max_possible > 0 else 0
                    else:
                        coupling_matrix[i, j] = 0.0  # Single node has no internal coupling
                else:
                    # For interlayer coupling
                    connections = layer_connections[i, j]
                    # Maximum possible connections based on number of nodes
                    nodes_in_layer_i = layer_connections[i, i]
                    nodes_in_layer_j = layer_connections[j, j]
                    max_connections = nodes_in_layer_i * nodes_in_layer_j
                    coupling_matrix[i, j] = connections / max_connections if max_connections > 0 else 0

    elif metric == "Topological Overlap":
        # Calculate number of shared neighbors between layers
        for i in range(n_layers):
            for j in range(n_layers):
                if i == j:
                    coupling_matrix[i, j] = 1.0
                else:
                    # Get neighbors of layer i and j
                    neighbors_i = set()
                    neighbors_j = set()
                    for k in range(n_layers):
                        if layer_connections[i, k] > 0 and k != i:
                            neighbors_i.add(k)
                        if layer_connections[j, k] > 0 and k != j:
                            neighbors_j.add(k)
                    
                    # Calculate overlap
                    union = len(neighbors_i.union(neighbors_j))
                    intersection = len(neighbors_i.intersection(neighbors_j))
                    if union > 0:
                        coupling_matrix[i, j] = intersection / union
                    else:
                        coupling_matrix[i, j] = 0.0

    elif metric == "Information Flow":
        # Create a graph from layer connections
        G = nx.Graph()
        for i in range(n_layers):
            G.add_node(i)

        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                if layer_connections[i, j] > 0:
                    G.add_edge(i, j, weight=layer_connections[i, j])

        # Calculate information flow using personalized PageRank
        for i in range(n_layers):
            personalization = {j: 0.0 for j in range(n_layers)}
            personalization[i] = 1.0

            if G.number_of_edges() > 0:
                pr = nx.pagerank(G, alpha=0.85, personalization=personalization)
                for j in range(n_layers):
                    coupling_matrix[i, j] = pr.get(j, 0)
            else:
                coupling_matrix[i, i] = 1.0

    return coupling_matrix


def create_layer_coupling_charts(
    coupling_heatmap_ax,
    coupling_bar_ax,
    dendrogram_ax,
    circular_ax,
    layer_connections,
    layers,
    medium_font=None,
    large_font=None,
    edge_threshold=0.1,
    metric="Edge Density"
):
    """Create a visualization of layer coupling with hierarchical organization"""
    if medium_font is None:
        medium_font = {"fontsize": 8}
    if large_font is None:
        large_font = {"fontsize": 10}

    # Calculate coupling matrix
    coupling_matrix = calculate_coupling_metric(
        layer_connections, metric=metric
    )
    
    # Print coupling matrix statistics
    print(f"\nCoupling matrix statistics for {metric}:")
    print(f"- Range: [{coupling_matrix.min():.3f}, {coupling_matrix.max():.3f}]")
    print(f"- Mean: {coupling_matrix.mean():.3f}")
    
    # Create heatmap
    im = coupling_heatmap_ax.imshow(coupling_matrix, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, ax=coupling_heatmap_ax, fraction=0.046, pad=0.04)
    coupling_heatmap_ax.set_title(f"Layer Coupling Matrix ({metric})", **large_font)

    # Calculate overall coupling scores
    np.fill_diagonal(coupling_matrix, 0)  # Zero out self-coupling for score calculation
    coupling_scores = coupling_matrix.sum(axis=1)  # Sum of all inter-layer coupling

    # Create bar chart of coupling scores
    sorted_indices = np.argsort(coupling_scores)[::-1]
    sorted_layers = [layers[i] for i in sorted_indices]
    sorted_scores = coupling_scores[sorted_indices]

    bars = coupling_bar_ax.barh(
        range(len(sorted_layers)), sorted_scores, color="skyblue"
    )
    coupling_bar_ax.set_yticks(range(len(sorted_layers)))
    coupling_bar_ax.set_yticklabels(sorted_layers, **medium_font)
    coupling_bar_ax.set_title("Layer Coupling Scores", **large_font)
    coupling_bar_ax.set_xlabel("Coupling Score", **medium_font)

    # Add value labels to bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        coupling_bar_ax.text(
            width * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            **medium_font,
        )

    # Create hierarchical clustering
    # Convert coupling matrix to distance matrix
    distance_matrix = 1 - coupling_matrix
    np.fill_diagonal(distance_matrix, 0)

    # Perform hierarchical clustering
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = hierarchy.linkage(condensed_dist, method="average")

    # Create dendrogram
    dendrogram = hierarchy.dendrogram(
        linkage_matrix, labels=layers, orientation="right", ax=dendrogram_ax
    )
    dendrogram_ax.set_title("Layer Hierarchy", **large_font)

    # Create circular layout based on hierarchical clustering
    G = nx.Graph()

    # Add nodes
    for i, layer in enumerate(layers):
        G.add_node(i, name=layer)

    # Add edges based on coupling strength with threshold
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            if coupling_matrix[i, j] > edge_threshold:  # Only add significant edges
                G.add_edge(i, j, weight=coupling_matrix[i, j])
    
    # Print network statistics
    print(f"Number of edges in visualization: {G.number_of_edges()}")
    print(f"Average coupling strength: {np.mean([d['weight'] for (u,v,d) in G.edges(data=True)]):.3f}")

    # Get the order of leaves from the dendrogram
    leaf_order = dendrogram["leaves"]

    # Create circular layout
    pos = {}
    num_nodes = len(leaf_order)
    for i, leaf_idx in enumerate(leaf_order):
        angle = 2 * np.pi * i / num_nodes
        pos[leaf_idx] = (np.cos(angle), np.sin(angle))

    # Draw the network
    nx.draw_networkx_nodes(
        G, pos, node_size=1000, node_color="lightblue", alpha=0.6, ax=circular_ax
    )

    # Draw edges with width and color based on coupling strength
    edges = G.edges(data=True)
    if edges:
        edge_weights = [d["weight"] for _, _, d in edges]
        max_weight = max(edge_weights)
        edge_widths = [1 + 4 * (w / max_weight) for w in edge_weights]
        nx.draw_networkx_edges(
            G,
            pos,
            width=edge_widths,
            edge_color=edge_weights,
            edge_cmap=plt.cm.viridis,
            ax=circular_ax,
        )

    # Add labels
    labels = {i: layer for i, layer in enumerate(layers)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=circular_ax)

    circular_ax.set_title("Layer Organization", **large_font)
    circular_ax.axis("off")
