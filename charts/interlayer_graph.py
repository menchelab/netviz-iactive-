import networkx as nx
import numpy as np

def create_interlayer_graph(ax, layer_connections, layers, small_font, medium_font):
    """Create graph visualization of layer connections"""
    if np.sum(layer_connections) > 0:
        # Create a graph where nodes are layers and edges represent connections
        G = nx.Graph()
        
        # Add nodes (layers)
        for i, layer in enumerate(layers):
            G.add_node(i, name=layer)
        
        # Add edges with weights based on connection counts
        for i in range(len(layers)):
            for j in range(i+1, len(layers)):
                if layer_connections[i, j] > 0:
                    G.add_edge(i, j, weight=layer_connections[i, j])
        
        # Position nodes using spring layout instead of circular
        pos = nx.spring_layout(G, seed=42)  # Using seed for consistency
        
        # Draw the graph
        node_sizes = [300 for _ in range(len(layers))]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', 
                              ax=ax)
        
        # Draw edges with width proportional to weight and increased transparency
        edge_widths = [G[u][v]['weight']/5 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4,
                              ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, labels={i: layer for i, layer in enumerate(layers)}, 
                               font_size=6, ax=ax)
        
        ax.set_title('Layer Connection Graph', **medium_font)
        ax.axis('off')  # Turn off axis
    else:
        ax.text(0.5, 0.5, 'No interlayer connections to display', 
               horizontalalignment='center', verticalalignment='center', **small_font)
        ax.axis('off') 