import networkx as nx
import numpy as np

def create_interlayer_graph(ax, layer_connections, layers, small_font, medium_font, visible_layers=None, layer_colors=None):
    """Create graph visualization of layer connections"""
    # If visible_layers is None, show all layers
    if visible_layers is None:
        visible_layers = list(range(len(layers)))
    
    # Filter the layer_connections matrix to only include visible layers
    visible_indices = np.array(visible_layers)
    if len(visible_indices) > 0:
        filtered_connections = layer_connections[np.ix_(visible_indices, visible_indices)]
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
            for j in range(i+1, len(filtered_layers)):
                if filtered_connections[i, j] > 0:
                    G.add_edge(i, j, weight=filtered_connections[i, j])
        
        pos = nx.spring_layout(G, seed=42, weight='weight', k=0.3)

        
        # Prepare node colors if layer_colors is provided
        node_colors = []
        if layer_colors:
            for layer in filtered_layers:
                if layer in layer_colors:
                    node_colors.append(layer_colors[layer])
                else:
                    node_colors.append('skyblue')  # Default color
        else:
            node_colors = ['skyblue' for _ in range(len(filtered_layers))]
        
        # Draw the graph
        node_sizes = [300 for _ in range(len(filtered_layers))]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                              ax=ax)
        
        # Draw edges with width proportional to weight and increased transparency
        edge_widths = [G[u][v]['weight']/5 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4,
                              ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, labels={i: layer for i, layer in enumerate(filtered_layers)}, 
                               font_size=6, ax=ax)
        
        ax.set_title('Layer Connection Graph', **medium_font)
        ax.axis('off')  # Turn off axis
    else:
        if len(filtered_layers) == 0:
            message = 'No visible layers to display'
        else:
            message = 'No interlayer connections to display'
            
        ax.text(0.5, 0.5, message, 
               horizontalalignment='center', verticalalignment='center', **small_font)
        ax.axis('off') 