import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable


def create_layer_influence_chart(
    bar_ax,
    network_ax,
    layer_connections,
    layers,
    medium_font,
    large_font,
    visible_layers=None,
    layer_colors=None,
    metric="PageRank",
):
    """
    Create visualizations of layer influence metrics.

    Parameters:
    -----------
    bar_ax : matplotlib.axes.Axes
        Axes for the bar chart of influence metrics
    network_ax : matplotlib.axes.Axes
        Axes for the network visualization with node sizes based on influence
    layer_connections : numpy.ndarray
        Matrix of connection counts between layers
    layers : list
        List of layer names
    medium_font, large_font : dict
        Font configuration dictionaries
    visible_layers : list, optional
        Indices of visible layers
    layer_colors : dict, optional
        Dictionary mapping layer names to colors
    metric : str
        The influence metric to calculate ("PageRank", "Eigenvector Centrality", or "Combined Influence Index")
    """
    # If visible_layers is None, show all layers
    if visible_layers is None:
        visible_layers = list(range(len(layers)))

    # Filter the layer_connections matrix to only include visible layers
    visible_indices = np.array(visible_layers)
    if len(visible_indices) > 0:
        filtered_connections = layer_connections[
            np.ix_(visible_indices, visible_indices)
        ]
        filtered_layers = [layers[i] for i in visible_indices]
    else:
        filtered_connections = np.zeros((0, 0))
        filtered_layers = []

    if len(filtered_layers) > 0 and np.sum(filtered_connections) > 0:
        # Create a graph where nodes are layers and edges represent connections
        G = nx.Graph()

        # Add nodes (layers)
        for i, layer in enumerate(filtered_layers):
            G.add_node(i, name=layer)

        # Add edges with weights based on connection counts
        for i in range(len(filtered_layers)):
            for j in range(i + 1, len(filtered_layers)):
                if filtered_connections[i, j] > 0:
                    G.add_edge(i, j, weight=filtered_connections[i, j])

        # Check if the graph has any edges
        if len(G.edges()) == 0:
            # No edges in the graph, display a message
            message = "No connections between visible layers"
            bar_ax.text(0.5, 0.5, message, ha="center", va="center", **medium_font)
            bar_ax.axis("off")

            network_ax.text(0.5, 0.5, message, ha="center", va="center", **medium_font)
            network_ax.axis("off")
            return

        # Calculate influence metrics based on selected metric
        try:
            if metric == "PageRank":
                influence_scores = nx.pagerank(G, weight="weight")
                metric_name = "PageRank"
            elif metric == "Eigenvector Centrality":
                influence_scores = nx.eigenvector_centrality(
                    G, weight="weight", max_iter=1000
                )
                metric_name = "Eigenvector Centrality"
            else:  # Combined Influence Index
                # Calculate multiple centrality measures and combine them
                try:
                    pagerank = nx.pagerank(G, weight="weight")
                except:
                    # Fallback if PageRank fails
                    print("Warning: PageRank calculation failed, using default values")
                    pagerank = {i: 1.0 / len(G) for i in G.nodes()}

                try:
                    eigenvector = nx.eigenvector_centrality(
                        G, weight="weight", max_iter=1000
                    )
                except:
                    # Fallback if eigenvector centrality fails
                    print(
                        "Warning: Eigenvector centrality calculation failed, using default values"
                    )
                    eigenvector = {i: 1.0 / len(G) for i in G.nodes()}

                try:
                    betweenness = nx.betweenness_centrality(
                        G, weight="weight", normalized=True
                    )
                except:
                    # Fallback if betweenness centrality fails
                    print(
                        "Warning: Betweenness centrality calculation failed, using default values"
                    )
                    betweenness = {i: 1.0 / len(G) for i in G.nodes()}

                try:
                    closeness = nx.closeness_centrality(G, distance="weight")
                except:
                    # Fallback if closeness centrality fails
                    print(
                        "Warning: Closeness centrality calculation failed, using default values"
                    )
                    closeness = {i: 1.0 / len(G) for i in G.nodes()}

                # Normalize each metric to [0,1] range
                def normalize_dict(d):
                    min_val = min(d.values())
                    max_val = max(d.values())
                    range_val = max_val - min_val
                    if range_val == 0:
                        return {k: 0.5 for k in d}
                    return {k: (v - min_val) / range_val for k, v in d.items()}

                pagerank_norm = normalize_dict(pagerank)
                eigenvector_norm = normalize_dict(eigenvector)
                betweenness_norm = normalize_dict(betweenness)
                closeness_norm = normalize_dict(closeness)

                # Combine metrics with weights
                influence_scores = {}
                for node in G.nodes():
                    influence_scores[node] = (
                        0.35 * pagerank_norm[node]
                        + 0.35 * eigenvector_norm[node]
                        + 0.15 * betweenness_norm[node]
                        + 0.15 * closeness_norm[node]
                    )

                metric_name = "Combined Influence Index"
        except Exception as e:
            # Handle any errors in centrality calculations
            print(f"Warning: Error calculating {metric}: {e}")
            # Create default influence scores
            influence_scores = {
                i: 1.0 / len(filtered_layers) for i in range(len(filtered_layers))
            }
            metric_name = f"{metric} (calculation failed)"

        # Sort layers by influence score for the bar chart
        sorted_items = sorted(
            influence_scores.items(), key=lambda x: x[1], reverse=True
        )
        sorted_indices = [item[0] for item in sorted_items]
        sorted_scores = [item[1] for item in sorted_items]
        sorted_layers = [filtered_layers[i] for i in sorted_indices]

        # Prepare colors for the bar chart
        bar_colors = []
        if layer_colors:
            for layer in sorted_layers:
                if layer in layer_colors:
                    bar_colors.append(layer_colors[layer])
                else:
                    bar_colors.append("skyblue")
        else:
            bar_colors = ["skyblue" for _ in range(len(sorted_layers))]

        # Create bar chart of influence scores
        bars = bar_ax.barh(sorted_layers, sorted_scores, color=bar_colors)
        bar_ax.set_title(f"Layer {metric_name}", **large_font)
        bar_ax.set_xlabel(metric_name, **medium_font)
        bar_ax.set_ylabel("Layer", **medium_font)

        # Add value labels to the bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x_pos = width * 1.01
            bar_ax.text(
                label_x_pos,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                va="center",
                **medium_font,
            )

        # Create network visualization with node sizes based on influence
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, seed=42, weight="weight")

        # Scale node sizes based on influence scores
        node_sizes = [influence_scores[i] * 3000 for i in range(len(filtered_layers))]

        # Prepare node colors
        node_colors = []
        if layer_colors:
            for layer in filtered_layers:
                if layer in layer_colors:
                    node_colors.append(layer_colors[layer])
                else:
                    node_colors.append("skyblue")
        else:
            node_colors = ["skyblue" for _ in range(len(filtered_layers))]

        # Draw the network
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8,
            ax=network_ax,
        )

        # Draw edges with width proportional to weight
        edge_widths = [G[u][v]["weight"] / 5 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, ax=network_ax)

        # Draw labels
        nx.draw_networkx_labels(
            G,
            pos,
            labels={
                i: f"{layer}\n({influence_scores[i]:.3f})"
                for i, layer in enumerate(filtered_layers)
            },
            font_size=8,
            ax=network_ax,
        )

        network_ax.set_title(f"Network Visualization by {metric_name}", **large_font)
        network_ax.axis("off")

        # Add a legend explaining the visualization
        legend_text = (
            f"Node size represents {metric_name}\n"
            f"Edge width represents connection strength\n"
            f"Values in parentheses are {metric_name} scores"
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

    else:
        if len(filtered_layers) == 0:
            message = "No visible layers to display"
        else:
            message = "No interlayer connections to display"

        bar_ax.text(0.5, 0.5, message, ha="center", va="center", **medium_font)
        bar_ax.axis("off")

        network_ax.text(0.5, 0.5, message, ha="center", va="center", **medium_font)
        network_ax.axis("off")
