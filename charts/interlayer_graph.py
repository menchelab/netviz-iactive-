import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from fa2_modified import ForceAtlas2

def create_interlayer_graph(ax, layer_connections, layers, small_font, medium_font, visible_layers=None, layer_colors=None, layout_algorithm='spring'):
    """Create graph visualization of layer connections
    """
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
        
        # Choose layout algorithm based on parameter
        if layout_algorithm == 'spring':
            pos = nx.spring_layout(G, seed=42, weight='weight', k=0.3)
        elif layout_algorithm == 'circular':
            pos = nx.circular_layout(G)
        elif layout_algorithm == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G, weight='weight')
        elif layout_algorithm == 'spectral':
            pos = nx.spectral_layout(G, weight='weight')
        elif layout_algorithm == 'shell':
            pos = nx.shell_layout(G)
        elif layout_algorithm == 'spiral':
            pos = nx.spiral_layout(G)
        elif layout_algorithm == 'force_atlas2':
            try:
                forceatlas2 = ForceAtlas2(
                    # Behavior alternatives
                    outboundAttractionDistribution=True,  # Dissuade hubs
                    linLogMode=False,  # NOT IMPLEMENTED
                    adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                    edgeWeightInfluence=1.0,

                    # Performance
                    jitterTolerance=1.0,  # Tolerance
                    barnesHutOptimize=True,
                    barnesHutTheta=1.2,
                    multiThreaded=False,  # NOT IMPLEMENTED

                    # Tuning
                    scalingRatio=2.0,
                    strongGravityMode=False,
                    gravity=1.0,

                    # Log
                    verbose=False
                )

                # Convert networkx graph to positions dict
                initial_pos = {node: np.array([0.5, 0.5]) + np.random.random(2) * 0.1 for node in G.nodes()}
                positions = forceatlas2.forceatlas2_networkx_layout(G, pos=initial_pos, iterations=100)

                # Normalize positions to fit in [0, 1] range
                pos = {k: np.array([v[0], v[1]]) for k, v in positions.items()}

                # Scale positions to be centered
                x_values = [p[0] for p in pos.values()]
                y_values = [p[1] for p in pos.values()]
                x_min, x_max = min(x_values), max(x_values)
                y_min, y_max = min(y_values), max(y_values)

                for k in pos:
                    pos[k][0] = (pos[k][0] - x_min) / (x_max - x_min) if x_max > x_min else 0.5
                    pos[k][1] = (pos[k][1] - y_min) / (y_max - y_min) if y_max > y_min else 0.5
                    pos[k] = pos[k] * 2 - 1  # Scale to [-1, 1]
            except Exception as e:
                print(f"ForceAtlas2 failed: {e}")
                pos = nx.spring_layout(G, seed=42)
        elif layout_algorithm == 'radial':
            # Find the node with highest degree (most connections)
            degrees = dict(G.degree(weight='weight'))
            center_node = max(degrees, key=degrees.get)
            pos = nx.kamada_kawai_layout(G, weight='weight')

            # Adjust positions to make the center node at (0,0)
            center_pos = pos[center_node]
            for node in pos:
                pos[node] = pos[node] - center_pos

            # Scale positions based on distance from center
            for node in pos:
                if node != center_node:
                    # Calculate distance from center
                    dist = np.sqrt(pos[node][0]**2 + pos[node][1]**2)
                    # Scale position based on weight of edge to center
                    if G.has_edge(node, center_node):
                        weight = G[node][center_node]['weight']
                        scale_factor = 1.0 / (weight + 1)  # Closer for higher weights
                        pos[node] = pos[node] * scale_factor
        elif layout_algorithm == 'weighted_spring':
            # Spring layout with stronger weight influence
            # Scale weights to have more impact
            for u, v, d in G.edges(data=True):
                d['weight'] = d['weight'] * 2  # Amplify weight effect
            pos = nx.spring_layout(G, seed=42, weight='weight', k=0.2, iterations=100)
        elif layout_algorithm == 'weighted_spectral':
            # Create a weighted adjacency matrix
            A = nx.to_numpy_array(G, weight='weight')
            # Compute the Laplacian
            D = np.diag(A.sum(axis=1))
            L = D - A
            # Compute eigenvectors of the Laplacian
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            # Use the eigenvectors corresponding to the second and third smallest eigenvalues
            pos = {i: (eigenvectors[i, 1], eigenvectors[i, 2]) for i in range(len(filtered_layers))}
        elif layout_algorithm == 'hierarchical_betweeness_centrality':
            # Hierarchical layout based on node importance (betweenness centrality)
            # I am trying this to find a potential order for layers
            # betweeness centrality then show from top to bottom in half circle
            # half circle is just to see edges better
            # Calculate betweenness centrality
            centrality = nx.betweenness_centrality(G, weight='weight')

            # Sort nodes by centrality
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

            # Create positions with most central nodes at top
            pos = {}
            levels = min(20, len(G.nodes()))
            nodes_per_level = max(1, len(G.nodes()) // levels)
            
            for i, (node, _) in enumerate(sorted_nodes):
                level = min(i // nodes_per_level, levels - 1)
                position_in_level = i % nodes_per_level
                total_in_level = min(nodes_per_level, len(G.nodes()) - level * nodes_per_level)
                
                # Calculate x position (spread across level)
                if total_in_level > 1:
                    x = position_in_level / (total_in_level - 1) * 2 - 1
                else:
                    x = 0
                
                # Calculate y position (level)
                y = 1 - 2 * level / (levels - 1) if levels > 1 else 0
                
                pos[node] = np.array([x, y])
            
            # Arrange nodes in a half-circle from 12 o'clock to 6 o'clock
            # Calculate the angle for each node based on its position in the sorted list
            total_nodes = len(sorted_nodes)
            for i, (node, _) in enumerate(sorted_nodes):
                # Calculate angle from 0 (12 o'clock) to 180 degrees (6 o'clock)
                angle = (i / (total_nodes - 1) if total_nodes > 1 else 0) * np.pi
                
                # Keep the original y value
                y = pos[node][1]
                
                # Calculate new x position based on the angle (radius = 1)
                x = np.sin(angle)
                
                pos[node] = np.array([x, y])


        elif layout_algorithm == 'connection_centric':
            # Find the most connected node (highest weighted degree)
            weighted_degrees = {node: sum(data['weight'] for _, _, data in G.edges(node, data=True)) 
                                for node in G.nodes()}
            center_node = max(weighted_degrees, key=weighted_degrees.get)
            
            # Initialize positions dictionary
            pos = {}
            
            # Place the center node at (0,0)
            pos[center_node] = np.array([0.0, 0.0])
            
            # Create a list of unplaced nodes
            unplaced_nodes = list(G.nodes())
            unplaced_nodes.remove(center_node)
            
            # Create lists for nodes to place above and below
            nodes_above = []
            nodes_below = []
            
            # First, find direct connections to the center node
            direct_connections = []
            for node in unplaced_nodes:
                if G.has_edge(center_node, node):
                    direct_connections.append((node, G[center_node][node]['weight']))
            
            # Sort direct connections by weight (strongest first)
            direct_connections.sort(key=lambda x: x[1], reverse=True)
            
            # Alternate placing nodes above and below
            for i, (node, _) in enumerate(direct_connections):
                if i % 2 == 0:
                    nodes_above.append(node)
                else:
                    nodes_below.append(node)
                unplaced_nodes.remove(node)
            
            # For remaining nodes, assign based on their connections to already placed nodes
            while unplaced_nodes:
                best_node = None
                best_score = -1
                best_position = "above"
                
                for node in unplaced_nodes:
                    # Calculate connection strength to nodes above and below
                    above_score = sum(G[node][n]['weight'] if G.has_edge(node, n) else 0 
                                        for n in nodes_above + [center_node])
                    below_score = sum(G[node][n]['weight'] if G.has_edge(node, n) else 0 
                                        for n in nodes_below + [center_node])
                    
                    # Determine best position and score
                    if above_score > below_score and above_score > best_score:
                        best_node = node
                        best_score = above_score
                        best_position = "above"
                    elif below_score > best_score:
                        best_node = node
                        best_score = below_score
                        best_position = "below"
                
                # If no connections found, just pick the first node
                if best_node is None and unplaced_nodes:
                    best_node = unplaced_nodes[0]
                    best_position = "above" if len(nodes_above) <= len(nodes_below) else "below"
                
                # Place the node
                if best_position == "above":
                    nodes_above.append(best_node)
                else:
                    nodes_below.append(best_node)
                
                unplaced_nodes.remove(best_node)
            
            # Position nodes vertically
            # Center node is at y=0
            # Nodes above have positive y, nodes below have negative y
            
            # Position nodes above
            for i, node in enumerate(nodes_above):
                y = (i + 1) * (2.0 / (len(nodes_above) + 1)) if nodes_above else 0
                # Add some horizontal variation based on connection strength to center
                x_offset = 0
                if G.has_edge(node, center_node):
                    # Stronger connections are closer to x=0
                    x_offset = 0.3 / (G[node][center_node]['weight'] + 1)
                else:
                    x_offset = 0.5
                
                # Alternate left and right for better visibility
                x = x_offset if i % 2 == 0 else -x_offset
                pos[node] = np.array([x, y])
            
            # Position nodes below
            for i, node in enumerate(nodes_below):
                y = -(i + 1) * (2.0 / (len(nodes_below) + 1)) if nodes_below else 0
                # Add some horizontal variation based on connection strength to center
                x_offset = 0
                if G.has_edge(node, center_node):
                    # Stronger connections are closer to x=0
                    x_offset = 0.3 / (G[node][center_node]['weight'] + 1)
                else:
                    x_offset = 0.5
                
                # Alternate left and right for better visibility
                x = x_offset if i % 2 == 0 else -x_offset
                pos[node] = np.array([x, y])
            
        else:
            # Default to spring layout if invalid algorithm specified
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
        if layout_algorithm == 'hierarchical_betweeness_centrality':
            # Add betweenness centrality value to the label
            centrality = nx.betweenness_centrality(G, weight='weight')
            labels = {i: f"{layer}\n({centrality[i]:.3f})" for i, layer in enumerate(filtered_layers)}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, ax=ax)
        elif layout_algorithm == 'connection_centric':
            # Add weighted degree value to the label
            weighted_degrees = {node: sum(data['weight'] for _, _, data in G.edges(node, data=True)) 
                                for node in G.nodes()}
            labels = {i: f"{layer}\n({weighted_degrees[i]:.1f})" for i, layer in enumerate(filtered_layers)}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, ax=ax)
        else:
            nx.draw_networkx_labels(G, pos, labels={i: layer for i, layer in enumerate(filtered_layers)}, 
                                   font_size=6, ax=ax)
        
        ax.set_title(f'Layer Connection Graph ({layout_algorithm} layout)', **medium_font)
        
        # Add explanation and axis lines for connection_centric layout
        if layout_algorithm == 'connection_centric':
            explanation = "Y-axis: Nodes above/below based on connection strength\nX-axis: Horizontal offset based on connection to center node\nNumbers: Total connection weight"
            ax.text(0.5, -0.05, explanation, transform=ax.transAxes, 
                   horizontalalignment='center', verticalalignment='top', 
                   fontsize=6, alpha=0.7)
            
            # Add x and y axis lines
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
        ax.axis('off')  # Turn off axis
    else:
        if len(filtered_layers) == 0:
            message = 'No visible layers to display'
        else:
            message = 'No interlayer connections to display'

        ax.text(0.5, 0.5, message, 
               horizontalalignment='center', verticalalignment='center', **small_font)
        ax.axis('off') 