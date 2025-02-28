import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats


def detect_anomalies(layer_connections, layers, threshold=1.0):
    """
    Detect anomalous connections between layers using statistical methods
    """
    n_layers = len(layers)
    anomalies = []
    anomaly_scores = np.zeros((n_layers, n_layers))

    # Calculate mean and std of connection counts
    connection_counts = layer_connections[~np.eye(n_layers, dtype=bool)]
    mean_connections = np.mean(connection_counts)
    std_connections = np.std(connection_counts)

    # Detect anomalies using z-score
    if std_connections > 0:  # Only if there's variation in connections
        for i in range(n_layers):
            for j in range(n_layers):
                if i != j:  # Skip self-connections
                    z_score = (
                        layer_connections[i, j] - mean_connections
                    ) / std_connections
                    anomaly_scores[i, j] = abs(z_score)

                    if abs(z_score) > threshold:
                        anomalies.append((i, j, layer_connections[i, j], z_score))

    return anomalies, anomaly_scores


def create_connection_anomaly_chart(
    ax, layer_connections, layers, threshold=1.0, medium_font=None, large_font=None
):
    """
    Create a visualization of connection anomalies between layers
    """
    if medium_font is None:
        medium_font = {"fontsize": 8}
    if large_font is None:
        large_font = {"fontsize": 10}

    # Detect anomalies
    anomalies, anomaly_scores = detect_anomalies(layer_connections, layers, threshold)

    # Create a graph
    G = nx.Graph()

    # Add nodes
    for i, layer in enumerate(layers):
        G.add_node(i, name=layer)

    # Add edges with weights based on anomaly scores
    edge_weights = []
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            if anomaly_scores[i, j] > threshold:
                G.add_edge(i, j, weight=anomaly_scores[i, j])
                edge_weights.append(anomaly_scores[i, j])

    if not G.edges():
        ax.text(
            0.5,
            0.5,
            "No significant anomalies detected",
            ha="center",
            va="center",
            **medium_font,
        )
        ax.axis("off")
        return

    # Create layout
    pos = nx.spring_layout(G)

    # Draw the network
    nodes = nx.draw_networkx_nodes(
        G, pos, node_size=1000, node_color="lightblue", alpha=0.6, ax=ax
    )

    # Draw edges with width and color based on anomaly score
    if edge_weights:
        max_weight = max(edge_weights)
        edge_widths = [2 + 3 * (w / max_weight) for w in edge_weights]
        edges = nx.draw_networkx_edges(
            G,
            pos,
            width=edge_widths,
            edge_color=edge_weights,
            edge_cmap=plt.cm.Reds,
            ax=ax,
        )

        # Add colorbar
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.Reds,
            norm=plt.Normalize(vmin=threshold, vmax=max(max_weight, threshold)),
        )
        plt.colorbar(sm, ax=ax, label="Anomaly Score")

    # Add labels
    labels = {i: layer for i, layer in enumerate(layers)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

    # Add title
    ax.set_title("Layer Connection Anomalies", **large_font)
    ax.axis("off")
