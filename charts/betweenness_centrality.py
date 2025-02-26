import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import logging

def create_betweenness_centrality_chart(ax, layer_connections, layers, visible_layer_indices, small_font, medium_font):
    """Create a chart showing betweenness centrality of layers in the network"""
    logger = logging.getLogger(__name__)

    # If visible_layers is None, show all layers
    if visible_layer_indices is None or len(visible_layer_indices) == 0:
        logger.info("No visible layers for betweenness centrality analysis")
        ax.text(0.5, 0.5, 'No visible layers to analyze', 
               horizontalalignment='center', verticalalignment='center', **small_font)
        ax.axis('off')
        return

    # Filter the layer_connections matrix to only include visible layers
    visible_indices = np.array(visible_layer_indices)
    filtered_connections = layer_connections[np.ix_(visible_indices, visible_indices)]
    filtered_layers = [layers[i] for i in visible_indices]

    if len(filtered_layers) <= 1 or np.sum(filtered_connections) == 0:
        logger.info("Not enough connections for betweenness centrality analysis")
        ax.text(0.5, 0.5, 'Not enough connections for analysis', 
               horizontalalignment='center', verticalalignment='center', **small_font)
        ax.axis('off')
        return

    # Create a graph where nodes are layers and edges represent connections
    G = nx.Graph()

    # Add nodes (layers)
    for i, layer in enumerate(filtered_layers):
        G.add_node(i, name=layer)

    # Add edges with weights based on connection counts
    for i in range(len(filtered_layers)):
        for j in range(i+1, len(filtered_layers)):
            if filtered_connections[i, j] > 0:
                G.add_edge(i, j, weight=filtered_connections[i, j])

    # Calculate betweenness centrality
    try:
        # Use weight as the strength of connection
        betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)

        # Sort layers by betweenness centrality
        sorted_layers = sorted([(filtered_layers[node], score) for node, score in betweenness.items()], 
                              key=lambda x: x[1], reverse=True)

        # Create bar chart
        y_pos = np.arange(len(sorted_layers))
        layer_names = [layer for layer, _ in sorted_layers]
        centrality_scores = [score for _, score in sorted_layers]

        # Plot horizontal bars
        bars = ax.barh(y_pos, centrality_scores, align='center', alpha=0.7)

        # Add value labels to the right of each bar
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{centrality_scores[i]:.3f}', 
                   va='center', **small_font)

        # Set labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels(layer_names, **small_font)
        ax.set_xlabel('Betweenness Centrality', **small_font)
        ax.set_title('Layer Betweenness Centrality', **medium_font)

        # Set x-axis limits
        ax.set_xlim(0, max(centrality_scores) * 1.15)  # Add some padding

        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.7)

        logger.info(f"Created betweenness centrality chart with {len(filtered_layers)} layers")
    except Exception as e:
        logger.error(f"Error creating betweenness centrality chart: {str(e)}")
        ax.text(0.5, 0.5, f'Error: {str(e)}', 
               horizontalalignment='center', verticalalignment='center', **small_font)
        ax.axis('off') 