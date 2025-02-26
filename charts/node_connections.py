import collections
import numpy as np

def create_node_connections_charts(intralayer_ax, interlayer_ax, visible_links, node_ids, 
                                  nodes_per_layer, small_font, medium_font):
    """Create charts showing top nodes by intra and inter-layer connections"""
    # Count intralayer and interlayer connections for each node
    intralayer_connections = collections.Counter()
    interlayer_connections = collections.Counter()
    
    for start_idx, end_idx in visible_links:
        # Get base node names
        start_node = node_ids[start_idx].split('_')[0]
        end_node = node_ids[end_idx].split('_')[0]
        
        # Check if this is an intralayer or interlayer connection
        start_layer = start_idx // nodes_per_layer
        end_layer = end_idx // nodes_per_layer
        
        if start_layer == end_layer:  # Intralayer connection
            intralayer_connections[start_node] += 1
            intralayer_connections[end_node] += 1
        else:  # Interlayer connection
            interlayer_connections[start_node] += 1
            interlayer_connections[end_node] += 1
    
    # Divide by 2 to avoid double counting
    for node in intralayer_connections:
        intralayer_connections[node] = intralayer_connections[node] // 2

    for node in interlayer_connections:
        interlayer_connections[node] = interlayer_connections[node] // 2

    # Plot top 20 nodes by intralayer connections
    top_intralayer = intralayer_connections.most_common(20)
    
    if top_intralayer:
        nodes, counts = zip(*top_intralayer)
        y_pos = np.arange(len(nodes))
        
        intralayer_ax.barh(y_pos, counts, align='center')
        intralayer_ax.set_yticks(y_pos)
        intralayer_ax.set_yticklabels(nodes, **small_font)
        intralayer_ax.invert_yaxis()  # Labels read top-to-bottom
        intralayer_ax.set_xlabel('Connection Count', **small_font)
        intralayer_ax.set_title('Top Intra conn', **medium_font)
        intralayer_ax.tick_params(axis='x', labelsize=6)
    else:
        intralayer_ax.text(0.5, 0.5, 'No intralayer connections to display', 
                          horizontalalignment='center', verticalalignment='center', **small_font)
    
    # Plot top 20 nodes by interlayer connections
    top_interlayer = interlayer_connections.most_common(20)
    
    if top_interlayer:
        nodes, counts = zip(*top_interlayer)
        y_pos = np.arange(len(nodes))
        
        interlayer_ax.barh(y_pos, counts, align='center')
        interlayer_ax.set_yticks(y_pos)
        interlayer_ax.set_yticklabels(nodes, **small_font)
        interlayer_ax.invert_yaxis()  # Labels read top-to-bottom
        interlayer_ax.set_xlabel('Connection Count', **small_font)
        interlayer_ax.set_title('Top Inter Conn', **medium_font)
        interlayer_ax.tick_params(axis='x', labelsize=6)
    else:
        interlayer_ax.text(0.5, 0.5, 'No interlayer connections to display', 
                          horizontalalignment='center', verticalalignment='center', **small_font)
    
    return intralayer_connections, interlayer_connections 