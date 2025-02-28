import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix


def create_information_flow_chart(
    flow_ax,
    network_ax,
    layer_connections,
    layers,
    medium_font,
    large_font,
    visible_layers=None,
    layer_colors=None,
    metric="Betweenness Centrality",
):
    """
    Create visualizations of information flow between layers.

    Parameters:
    -----------
    flow_ax : matplotlib.axes.Axes
        Axes for the flow heatmap visualization
    network_ax : matplotlib.axes.Axes
        Axes for the flow network visualization
    layer_connections : numpy.ndarray
        Matrix of connection counts between layers (already filtered)
    layers : list
        List of visible layer names (already filtered)
    medium_font, large_font : dict
        Font configuration dictionaries
    visible_layers : list, optional
        Sequential indices for visible layers (0 to len(layers)-1)
    layer_colors : dict, optional
        Dictionary mapping layer names to colors
    metric : str
        The flow metric to use ("Betweenness Centrality", "Flow Betweenness", or "Information Centrality")
    """
    # No need to filter the layer_connections matrix as it's already filtered
    # Just check if we have any data to visualize
    if len(layers) > 0 and np.sum(layer_connections) > 0:
        # Create a graph where nodes are layers and edges represent connections
        G = nx.Graph()

        # Add nodes (layers) using sequential indices
        for i, layer in enumerate(layers):
            G.add_node(i, name=layer)

        # Add edges with weights based on connection counts
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                if layer_connections[i, j] > 0:
                    G.add_edge(i, j, weight=layer_connections[i, j])

        # Check if the graph has any edges
        if len(G.edges()) == 0:
            # No edges in the graph, display a message
            message = "No connections between visible layers"
            flow_ax.text(0.5, 0.5, message, ha="center", va="center", **medium_font)
            flow_ax.axis("off")

            network_ax.text(0.5, 0.5, message, ha="center", va="center", **medium_font)
            network_ax.axis("off")
            return

        # Calculate flow metrics based on selected metric
        try:
            if metric == "Betweenness Centrality":
                # Calculate betweenness centrality
                centrality = nx.betweenness_centrality(
                    G, weight="weight", normalized=True
                )

                # Calculate flow matrix based on betweenness
                flow_matrix = np.zeros_like(layer_connections, dtype=float)
                for i in range(len(layers)):
                    for j in range(len(layers)):
                        if i != j and layer_connections[i, j] > 0:
                            # Flow from i to j depends on the betweenness of both layers
                            flow_matrix[i, j] = (
                                layer_connections[i, j]
                                * (centrality[i] + centrality[j])
                                / 2
                            )

                metric_name = "Betweenness Centrality"

            elif metric == "Flow Betweenness":
                # Calculate flow betweenness centrality (if available)
                try:
                    centrality = nx.current_flow_betweenness_centrality(
                        G, weight="weight", normalized=True
                    )
                except:
                    # Fallback to regular betweenness if current_flow_betweenness is not available
                    print(
                        "Warning: Flow betweenness calculation failed, falling back to regular betweenness"
                    )
                    centrality = nx.betweenness_centrality(
                        G, weight="weight", normalized=True
                    )

                # Calculate flow matrix based on flow betweenness
                flow_matrix = np.zeros_like(layer_connections, dtype=float)
                for i in range(len(layers)):
                    for j in range(len(layers)):
                        if i != j and layer_connections[i, j] > 0:
                            # Flow from i to j depends on the flow betweenness of both layers
                            flow_matrix[i, j] = (
                                layer_connections[i, j]
                                * (centrality[i] + centrality[j])
                                / 2
                            )

                metric_name = "Flow Betweenness"

            else:  # Information Centrality
                # Calculate information centrality (approximation)
                # Information centrality is related to the inverse of the sum of resistances
                # We'll use closeness centrality as an approximation
                centrality = nx.closeness_centrality(G, distance="weight")

                # Calculate flow matrix based on information centrality
                flow_matrix = np.zeros_like(layer_connections, dtype=float)
                for i in range(len(layers)):
                    for j in range(len(layers)):
                        if i != j and layer_connections[i, j] > 0:
                            # Flow from i to j depends on the information centrality of both layers
                            flow_matrix[i, j] = (
                                layer_connections[i, j]
                                * (centrality[i] + centrality[j])
                                / 2
                            )

                metric_name = "Information Centrality"
        except Exception as e:
            # Handle any errors in centrality calculations
            print(f"Warning: Error calculating {metric}: {e}")
            # Create a default flow matrix and centrality
            flow_matrix = np.zeros_like(layer_connections, dtype=float)
            centrality = {
                i: 1.0 / len(layers) for i in range(len(layers))
            }
            metric_name = f"{metric} (calculation failed)"

        # Create heatmap of flow matrix
        im = flow_ax.imshow(flow_matrix, cmap="viridis")

        # Add colorbar
        cbar = flow_ax.figure.colorbar(im, ax=flow_ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Information Flow", fontsize=8)

        # Add labels
        flow_ax.set_xticks(range(len(layers)))
        flow_ax.set_yticks(range(len(layers)))
        flow_ax.set_xticklabels(layers, rotation=90, fontsize=8)
        flow_ax.set_yticklabels(layers, fontsize=8)

        flow_ax.set_title(f"Information Flow Heatmap ({metric_name})", **large_font)
        flow_ax.set_xlabel("Destination Layer", **medium_font)
        flow_ax.set_ylabel("Source Layer", **medium_font)

        # Create flow graph visualization
        # Use a directed graph to show flow direction
        D = nx.DiGraph()

        # Add nodes
        for i, layer in enumerate(layers):
            D.add_node(i, name=layer)

        # Add directed edges with weights based on flow
        for i in range(len(layers)):
            for j in range(len(layers)):
                if i != j and flow_matrix[i, j] > 0:
                    D.add_edge(i, j, weight=flow_matrix[i, j])

        # Calculate node sizes based on total flow (in + out)
        total_flow = flow_matrix.sum(axis=1) + flow_matrix.sum(axis=0)
        node_sizes = (
            total_flow * 500 / max(total_flow)
            if max(total_flow) > 0
            else np.ones(len(layers)) * 300
        )

        # Use spring layout for better visualization
        pos = nx.spring_layout(D, seed=42)

        # Prepare node colors
        node_colors = []
        if layer_colors:
            for layer in layers:
                if layer in layer_colors:
                    node_colors.append(layer_colors[layer])
                else:
                    node_colors.append("skyblue")
        else:
            node_colors = ["skyblue" for _ in range(len(layers))]

        # Draw the nodes
        nx.draw_networkx_nodes(
            D,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8,
            ax=network_ax,
        )

        # Draw the edges with width and color based on flow strength
        edges = D.edges(data=True)
        if edges:
            edge_widths = [
                d["weight"] * 5 / max(flow_matrix.max(), 0.001) for _, _, d in edges
            ]

            # Use a colormap for edge colors based on flow strength
            edge_colors = [
                plt.cm.plasma(d["weight"] / max(flow_matrix.max(), 0.001))
                for _, _, d in edges
            ]

            # Draw edges with arrows to show direction
            nx.draw_networkx_edges(
                D,
                pos,
                width=edge_widths,
                edge_color=edge_colors,
                alpha=0.7,
                ax=network_ax,
                arrowstyle="->",
                arrowsize=10,
            )

        # Draw labels
        nx.draw_networkx_labels(
            D,
            pos,
            labels={i: layers[i] for i in range(len(layers))},
            font_size=8,
            ax=network_ax,
        )

        network_ax.set_title(f"Information Flow Network ({metric_name})", **large_font)
        network_ax.axis("off")

        # Add a legend explaining the visualization
        legend_text = (
            "Node size represents total information flow\n"
            "Edge width and color represent flow strength\n"
            "Arrows indicate flow direction"
        )
        network_ax.text(
            0.5,
            -0.1,
            legend_text,
            transform=network_ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7),
        )

        # Calculate and display key metrics
        try:
            if metric == "Betweenness Centrality":
                # Calculate information spread efficiency
                efficiency = (
                    flow_matrix.sum()
                    / (len(layers) * (len(layers) - 1))
                    if len(layers) > 1
                    else 0
                )

                # Identify bottleneck layers (high betweenness)
                if centrality:
                    bottleneck_idx = max(centrality.items(), key=lambda x: x[1])[0]
                    bottleneck_layer = layers[bottleneck_idx]

                    # Identify peripheral layers (low betweenness)
                    peripheral_idx = min(centrality.items(), key=lambda x: x[1])[0]
                    peripheral_layer = layers[peripheral_idx]

                    metrics_text = (
                        f"Information Flow Efficiency: {efficiency:.3f}\n"
                        f"Bottleneck Layer: {bottleneck_layer}\n"
                        f"Peripheral Layer: {peripheral_layer}"
                    )
                else:
                    metrics_text = "Insufficient data for metrics"

            elif metric == "Flow Betweenness":
                # Calculate average flow
                avg_flow = (
                    flow_matrix.sum()
                    / (len(layers) * (len(layers) - 1))
                    if len(layers) > 1
                    else 0
                )

                # Identify key flow layers
                if centrality:
                    sorted_centrality = sorted(
                        centrality.items(), key=lambda x: x[1], reverse=True
                    )
                    top_layers = [
                        layers[idx]
                        for idx, _ in sorted_centrality[
                            : min(3, len(sorted_centrality))
                        ]
                    ]

                    metrics_text = (
                        f"Average Flow: {avg_flow:.3f}\n"
                        f"Key Flow Layers: {', '.join(top_layers)}"
                    )
                else:
                    metrics_text = "Insufficient data for metrics"

            else:  # Information Centrality
                # Calculate information centrality metrics
                if centrality:
                    sorted_centrality = sorted(
                        centrality.items(), key=lambda x: x[1], reverse=True
                    )
                    top_layers = [
                        layers[idx]
                        for idx, _ in sorted_centrality[
                            : min(3, len(sorted_centrality))
                        ]
                    ]

                    metrics_text = (
                        f"Top Information Central Layers: {', '.join(top_layers)}"
                    )
                else:
                    metrics_text = "Insufficient data for metrics"
        except Exception as e:
            # Handle any errors in metrics calculations
            print(f"Warning: Error calculating metrics: {e}")
            metrics_text = "Error calculating metrics"

        # Add metrics text to the flow_ax
        flow_ax.text(
            0.5,
            -0.15,
            metrics_text,
            transform=flow_ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7),
        )

    else:
        if len(layers) == 0:
            message = "No visible layers to display"
        else:
            message = "No interlayer connections to display"

        flow_ax.text(0.5, 0.5, message, ha="center", va="center", **medium_font)
        flow_ax.axis("off")

        network_ax.text(0.5, 0.5, message, ha="center", va="center", **medium_font)
        network_ax.axis("off")
