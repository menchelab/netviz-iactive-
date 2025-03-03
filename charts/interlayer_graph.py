import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from utils.calc_layout import get_layout_position


def create_interlayer_graph(
    ax,
    layer_connections,
    layers,
    small_font,
    medium_font,
    visible_layers=None,
    layer_colors=None,
    layout_algorithm="spring",
):
    """Create graph visualization of layer connections"""
    # If no connections or empty matrix, show message and return
    if layer_connections.size == 0 or np.sum(layer_connections) == 0:
        ax.text(
            0.5,
            0.5,
            "No connections between visible layers",
            horizontalalignment="center",
            verticalalignment="center",
            **small_font,
        )
        ax.axis("off")
        return

    # Since layer_connections is already filtered, we should use sequential indices
    active_layers = layers
    if visible_layers is not None:
        # Validate indices are within bounds
        matrix_size = layer_connections.shape[0]
        if any(i >= len(layers) for i in visible_layers):
            ax.text(
                0.5,
                0.5,
                "Invalid layer indices detected",
                horizontalalignment="center",
                verticalalignment="center",
                **small_font,
            )
            ax.axis("off")
            return
        active_layers = layers

    # Create graph without node attributes
    G = nx.Graph()
    G.add_nodes_from(range(len(active_layers)))

    # Add edges from the connection matrix
    for i in range(layer_connections.shape[0]):
        for j in range(i + 1, layer_connections.shape[1]):
            if layer_connections[i, j] > 0:
                G.add_edge(i, j, weight=layer_connections[i, j])

    # Get layout positions using the new module
    pos = get_layout_position(G, layout_algorithm)

    # Get colors directly from layer names
    if layer_colors is None:
        layer_colors = {}
    node_colors = [layer_colors.get(active_layers[node], "skyblue") for node in G.nodes()]

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=300,
        node_color=node_colors,
        ax=ax
    )

    # Draw edges with adaptive weight scaling
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    min_weight = min(edge_weights) if edge_weights else 0
    
    # Linear scaling between 0.3 and 10.0
    if max_weight > min_weight:
        scaled_weights = [
            0.3 + 9.7 * ((w - min_weight) / (max_weight - min_weight))
            for w in edge_weights
        ]
    else:
        scaled_weights = [1.0 for _ in edge_weights]

    nx.draw_networkx_edges(
        G, pos,
        width=scaled_weights,
        alpha=0.4,
        ax=ax
    )

    # Draw labels using layer names directly
    labels = {node: active_layers[node] for node in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=6,
        ax=ax
    )

    # Set title and turn off axis
    ax.set_title(f"Layer Connection Graph ({layout_algorithm} layout)", **medium_font)
    ax.axis("off")

    # Add explanation and axis lines for connection_centric layout
    if layout_algorithm == "connection_centric":
        explanation = "Y-axis: Nodes above/below based on connection strength\nX-axis: Horizontal offset based on connection to center node\nNumbers: Total connection weight"
        ax.text(
            0.5,
            -0.05,
            explanation,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=6,
            alpha=0.7,
        )

        # Add x and y axis lines
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
