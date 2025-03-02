import logging
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import matplotlib.colors as mcolors

def create_interlayer_path_analysis(
    ax,
    visible_links,
    node_ids,
    node_clusters,
    nodes_per_layer,
    layers,
    visible_layer_indices=None,
    cluster_colors=None,
    analysis_type="path_length",  # Options: "path_length", "betweenness", "bottleneck"
    medium_fontsize=12,
    small_fontsize=9
):
    """
    Analyze and visualize interlayer paths between layers, regardless of cluster.
    
    This visualization builds a custom network that connects all visible interlayer edges 
    with all visible intralayer edges over common nodes, showing how layers are connected.
    
    The analysis focuses on:
    1. Path lengths: The shortest paths between layers
    2. Betweenness: Which nodes serve as bridges between layers
    3. Bottlenecks: Critical connections that, if removed, would disconnect layers
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to draw the visualization on
    visible_links : list
        List of (source_id, target_id) tuples representing visible edges
    node_ids : list
        List of all node IDs in the network
    node_clusters : dict
        Dictionary mapping node IDs to cluster assignments
    nodes_per_layer : dict
        Dictionary mapping layer indices to lists of node IDs
    layers : list
        List of layer names
    visible_layer_indices : list
        List of indices of visible layers
    cluster_colors : dict
        Dictionary mapping cluster IDs to colors
    analysis_type : str
        Type of analysis to perform: "path_length", "betweenness", or "bottleneck"
    medium_fontsize : int, optional
        Font size for medium text
    small_fontsize : int, optional
        Font size for small text
    """
    try:
        # Ensure visible_layer_indices is defined
        visible_layers = visible_layer_indices if visible_layer_indices is not None else []
        
        logging.info(f"Creating custom network for interlayer path analysis with {len(visible_links)} visible links")
        
        # Check if nodes_per_layer is an integer and convert it to a dictionary if needed
        if isinstance(nodes_per_layer, int):
            # Create a dictionary with the same number of nodes for each layer
            nodes_per_layer_dict = {}
            for layer_idx in visible_layers:
                if layer_idx < len(layers):
                    nodes_per_layer_dict[layer_idx] = list(range(layer_idx * nodes_per_layer, (layer_idx + 1) * nodes_per_layer))
        else:
            nodes_per_layer_dict = nodes_per_layer
        
        # Create a graph from the visible links
        G = nx.Graph()
        
        # Add nodes with attributes
        for node_id in node_ids:
            if node_id in node_clusters:
                # Find which layer this node belongs to
                layer_idx = None
                for layer_idx, nodes in nodes_per_layer_dict.items():
                    if node_id in nodes:
                        break
                
                if layer_idx is not None:
                    G.add_node(node_id, 
                              cluster=node_clusters[node_id], 
                              layer=layer_idx)
        
        # Classify edges as interlayer or intralayer
        interlayer_edges = []
        intralayer_edges = []
        
        for source_id, target_id in visible_links:
            if source_id in G.nodes and target_id in G.nodes:
                source_layer = G.nodes[source_id]['layer']
                target_layer = G.nodes[target_id]['layer']
                
                if source_layer == target_layer:
                    intralayer_edges.append((source_id, target_id))
                else:
                    interlayer_edges.append((source_id, target_id))
        
        logging.info(f"Classified {len(interlayer_edges)} interlayer edges and {len(intralayer_edges)} intralayer edges")
        
        # Add all edges to the graph
        for source_id, target_id in interlayer_edges + intralayer_edges:
            G.add_edge(source_id, target_id)
        
        # Find common nodes that connect interlayer and intralayer edges
        common_nodes = set()
        
        # Nodes that are part of interlayer edges
        interlayer_nodes = set()
        for source_id, target_id in interlayer_edges:
            interlayer_nodes.add(source_id)
            interlayer_nodes.add(target_id)
        
        # Nodes that are part of intralayer edges
        intralayer_nodes = set()
        for source_id, target_id in intralayer_edges:
            intralayer_nodes.add(source_id)
            intralayer_nodes.add(target_id)
        
        # Find common nodes
        common_nodes = interlayer_nodes.intersection(intralayer_nodes)
        
        logging.info(f"Found {len(common_nodes)} common nodes connecting interlayer and intralayer edges")
        
        # If no visible layer indices provided, use all layers
        if visible_layer_indices is None:
            visible_layer_indices = list(range(len(layers)))
        
        # Filter to only include visible layers
        visible_layers = [layers[i] for i in visible_layer_indices if i < len(layers)]
        
        # Default cluster colors if not provided
        if cluster_colors is None:
            unique_clusters = set()
            for node_id in G.nodes():
                if 'cluster' in G.nodes[node_id]:
                    unique_clusters.add(G.nodes[node_id]['cluster'])
            
            import matplotlib.cm as cm
            cmap = cm.get_cmap('tab10')
            cluster_colors = {cluster: cmap(i % 10) for i, cluster in enumerate(sorted(unique_clusters))}
        
        # Add node attributes for visualization
        for node in G.nodes:
            G.nodes[node]['is_common'] = node in common_nodes
            if node in G.nodes and 'layer' in G.nodes[node]:
                layer_idx = G.nodes[node]['layer']
                if layer_idx < len(layers):
                    G.nodes[node]['layer_name'] = layers[layer_idx]
                else:
                    G.nodes[node]['layer_name'] = f"Layer {layer_idx}"
        
        # Perform the selected analysis
        if analysis_type == "path_length":
            _analyze_path_lengths(ax, G, visible_layer_indices, layers, node_clusters, cluster_colors, common_nodes)
        elif analysis_type == "betweenness":
            _analyze_betweenness(ax, G, visible_layer_indices, layers, node_clusters, cluster_colors, common_nodes)
        elif analysis_type == "bottleneck":
            _analyze_bottlenecks(ax, G, visible_layer_indices, layers, node_clusters, cluster_colors, common_nodes)
        else:
            ax.text(0.5, 0.5, f"Unknown analysis type: {analysis_type}", 
                   ha='center', va='center')
            
        # Set title based on analysis type
        title_map = {
            "path_length": "Interlayer-Intralayer Path Analysis",
            "betweenness": "Node Betweenness in Combined Network",
            "bottleneck": "Critical Connections in Combined Network"
        }
        ax.set_title(title_map.get(analysis_type, f"Combined Network Analysis: {analysis_type}"))
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        return ax
        
    except Exception as e:
        logging.error(f"Error in create_interlayer_path_analysis: {str(e)}")
        ax.text(0.5, 0.5, f"Error creating interlayer path analysis: {str(e)}", 
               ha='center', va='center')
        return ax

def _analyze_path_lengths(ax, G, visible_layer_indices, layers, node_clusters, cluster_colors, common_nodes=None):
    """
    Analyze and visualize the shortest paths between layers in the combined network.
    """
    logger = logging.getLogger(__name__)
    logger.info("Analyzing path lengths in the combined interlayer-intralayer network")
    
    # Group nodes by layer
    layer_nodes = defaultdict(list)
    for node in G.nodes:
        layer = G.nodes[node]['layer']
        layer_nodes[layer].append(node)
    
    # Calculate average shortest path lengths between layers
    path_lengths = {}
    
    # Get unique layers
    unique_layers = sorted(layer_nodes.keys())
    
    # Create a matrix to store path lengths
    # Format: (source_layer, target_layer) -> avg_path_length
    for source_layer in unique_layers:
        # Skip if no nodes in this layer
        if not layer_nodes[source_layer]:
            continue
            
        for target_layer in unique_layers:
            # Skip same layer
            if source_layer == target_layer:
                continue
                
            # Skip if no nodes in this layer
            if not layer_nodes[target_layer]:
                continue
            
            # Calculate shortest paths between all node pairs
            total_length = 0
            count = 0
            
            for source_node in layer_nodes[source_layer]:
                for target_node in layer_nodes[target_layer]:
                    try:
                        # Calculate shortest path length
                        path_length = nx.shortest_path_length(G, source=source_node, target=target_node)
                        total_length += path_length
                        count += 1
                    except nx.NetworkXNoPath:
                        # No path exists
                        pass
            
            # Calculate average path length if paths exist
            if count > 0:
                avg_path_length = total_length / count
                path_lengths[(source_layer, target_layer)] = avg_path_length
    
    # Check if we have any path data
    if not path_lengths:
        ax.text(0.5, 0.5, "No paths found between different layers", 
                ha="center", va="center")
        ax.axis("off")
        return
    
    # Create a visualization of the path lengths
    # We'll use a heatmap visualization with layers on x and y axes
    
    # Get the number of unique layers
    n_layers = len(unique_layers)
    
    # Create a grid for the heatmap
    grid = np.full((n_layers, n_layers), np.nan)
    
    # Fill the grid with path lengths
    for (source_layer, target_layer), path_length in path_lengths.items():
        # Convert layer indices to positions in the sorted unique_layers list
        source_layer_pos = unique_layers.index(source_layer)
        target_layer_pos = unique_layers.index(target_layer)
        
        # Set the value in the grid
        grid[source_layer_pos, target_layer_pos] = path_length
    
    # Create a custom colormap from blue (short paths) to red (long paths)
    cmap = LinearSegmentedColormap.from_list('path_length_cmap', ['#2c7bb6', '#ffffbf', '#d7191c'])
    
    # Create the heatmap
    im = ax.imshow(grid, cmap=cmap, interpolation='nearest')
    
    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Average Path Length')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_layers))
    ax.set_yticks(np.arange(n_layers))
    
    # Use layer names for labels
    layer_labels = []
    for layer_idx in unique_layers:
        if layer_idx < len(layers):
            layer_labels.append(layers[layer_idx])
        else:
            layer_labels.append(f"Layer {layer_idx}")
    
    ax.set_xticklabels(layer_labels, rotation=45, ha="right")
    ax.set_yticklabels(layer_labels)
    
    # Add text annotations
    for i in range(n_layers):
        for j in range(n_layers):
            if not np.isnan(grid[i, j]):
                text_color = "white" if grid[i, j] > np.nanmax(grid) / 2 else "black"
                ax.text(j, i, f"{grid[i, j]:.2f}", 
                       ha="center", va="center", 
                       color=text_color)
    
    # Add grid lines
    ax.set_xticks(np.arange(-.5, n_layers, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_layers, 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    
    # Set labels
    ax.set_xlabel("Target Layer")
    ax.set_ylabel("Source Layer")
    
    # Add title
    ax.set_title("Average Shortest Path Length Between Layers\n(Combined Interlayer-Intralayer Network)", fontsize=12)

def _analyze_betweenness(ax, G, visible_layer_indices, layers, node_clusters, cluster_colors, common_nodes=None):
    """Analyze and visualize betweenness centrality of nodes as bridges between layers in the combined network"""
    logger = logging.getLogger(__name__)
    logger.info("Analyzing betweenness centrality in the combined interlayer-intralayer network")
    
    # Calculate betweenness centrality for all nodes
    betweenness = nx.betweenness_centrality(G, weight='weight')
    
    # Group betweenness by layer
    layer_betweenness = defaultdict(float)
    common_node_betweenness = defaultdict(float)
    
    for node, bc in betweenness.items():
        layer = G.nodes[node]['layer']
        layer_betweenness[layer] += bc
        
        # Track betweenness of common nodes separately
        if common_nodes and node in common_nodes:
            common_node_betweenness[layer] += bc
    
    # Get unique layers
    unique_layers = sorted(layer_betweenness.keys())
    
    # Create a visualization of the betweenness values
    # We'll use a grouped bar plot with layers on the x-axis and betweenness values on the y-axis
    
    # Set up the bar positions
    x = np.arange(len(unique_layers))
    width = 0.35
    
    # Create the bars
    ax.bar(x - width/2, [layer_betweenness[layer] for layer in unique_layers], 
           width, label='All Nodes', color='skyblue')
    
    if common_nodes:
        ax.bar(x + width/2, [common_node_betweenness[layer] for layer in unique_layers], 
               width, label='Common Nodes', color='orange')
    
    # Set ticks and labels
    ax.set_xticks(x)
    
    # Use layer names for labels
    layer_labels = []
    for layer_idx in unique_layers:
        if layer_idx < len(layers):
            layer_labels.append(layers[layer_idx])
        else:
            layer_labels.append(f"Layer {layer_idx}")
            
    ax.set_xticklabels(layer_labels, rotation=45, ha="right")
    ax.set_ylabel('Betweenness Centrality')
    
    # Add a legend
    ax.legend()
    
    # Add a title
    ax.set_title('Node Betweenness Centrality by Layer\n(Combined Interlayer-Intralayer Network)', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    logger.info(f"Created betweenness visualization with {len(unique_layers)} layers")

def _analyze_bottlenecks(ax, G, visible_layer_indices, layers, node_clusters, cluster_colors, common_nodes=None):
    """Analyze and visualize bottleneck connections in the combined interlayer-intralayer network"""
    logger = logging.getLogger(__name__)
    logger.info("Analyzing bottleneck connections in the combined interlayer-intralayer network")
    
    # Calculate edge betweenness centrality to identify bottlenecks
    edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight')
    
    # Normalize edge betweenness by the maximum value
    max_betweenness = max(edge_betweenness.values()) if edge_betweenness else 1.0
    normalized_betweenness = {edge: bc / max_betweenness for edge, bc in edge_betweenness.items()}
    
    # Create a visualization of the network with bottlenecks highlighted
    # Position nodes using a spring layout, grouped by layer
    pos = nx.spring_layout(G, seed=42)
    
    # Draw the graph
    # 1. Draw edges with width and color based on betweenness
    for edge, betweenness in normalized_betweenness.items():
        u, v = edge
        
        # Determine if this is an interlayer edge
        is_interlayer = G.nodes[u]['layer'] != G.nodes[v]['layer']
        
        # Calculate edge width based on betweenness
        width = 1 + 5 * betweenness
        
        # Calculate edge color based on type and betweenness
        if is_interlayer:
            color = plt.cm.Blues(betweenness)  # Blue for interlayer
        else:
            color = plt.cm.Reds(betweenness)   # Red for intralayer
        
        # Draw the edge
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
               linewidth=width, color=color, alpha=0.7,
               zorder=1)  # Lower zorder to draw behind nodes
        
        # Add edge label for significant bottlenecks
        if betweenness > 0.5:
            # Position the label at the middle of the edge
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            
            # Add a small offset to avoid overlapping with the edge
            dx = pos[v][0] - pos[u][0]
            dy = pos[v][1] - pos[u][1]
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 0:
                offset_x = -dy / norm * 0.02
                offset_y = dx / norm * 0.02
            else:
                offset_x = offset_y = 0
            
            # Add the label
            edge_type = "Inter" if is_interlayer else "Intra"
            ax.text(x + offset_x, y + offset_y, 
                   f"{edge_type}\nBC={betweenness:.2f}", 
                   fontsize=8, 
                   ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                   zorder=3)  # Higher zorder to draw on top
    
    # 2. Draw nodes with size based on betweenness and color based on layer
    node_betweenness = nx.betweenness_centrality(G, weight='weight')
    max_betweenness = max(node_betweenness.values()) if node_betweenness else 1.0
    
    for node in G.nodes:
        # Get node attributes
        layer = G.nodes[node]['layer']
        cluster = G.nodes[node]['cluster']
        is_common = G.nodes[node].get('is_common', False)
        
        # Calculate node size based on betweenness
        betweenness = node_betweenness.get(node, 0)
        size = 100 + 500 * (betweenness / max_betweenness)
        
        # Get node color based on layer
        if layer < len(layers):
            layer_name = layers[layer]
        else:
            layer_name = f"Layer {layer}"
            
        # Use a different color for common nodes
        if is_common:
            color = 'yellow'  # Highlight common nodes
            edgecolor = 'red'
            linewidth = 2
        else:
            # Use cluster colors
            if cluster_colors and cluster in cluster_colors:
                color = cluster_colors[cluster]
            else:
                # Use a default colormap if cluster colors not provided
                color = plt.cm.tab10(cluster % 10)
            edgecolor = 'black'
            linewidth = 1
        
        # Draw the node
        ax.scatter(pos[node][0], pos[node][1], 
                  s=size, color=color, edgecolor=edgecolor, linewidth=linewidth,
                  zorder=2)  # Middle zorder to draw between edges and labels
        
        # Add node label for significant nodes
        if betweenness > 0.1 or is_common:
            label = f"L{layer}C{cluster}"
            if is_common:
                label += "\n(Common)"
            if betweenness > 0.1:
                label += f"\nBC={betweenness:.2f}"
                
            ax.text(pos[node][0], pos[node][1], 
                   label, 
                   fontsize=8, 
                   ha='center', va='center',
                   zorder=4)  # Highest zorder to draw on top of everything
    
    # Add a legend for edge types
    inter_line = plt.Line2D([0], [0], color=plt.cm.Blues(0.7), linewidth=3, label='Interlayer Edge')
    intra_line = plt.Line2D([0], [0], color=plt.cm.Reds(0.7), linewidth=3, label='Intralayer Edge')
    common_node = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                            markeredgecolor='red', markersize=10, label='Common Node')
    
    ax.legend(handles=[inter_line, intra_line, common_node], loc='upper right')
    
    # Add a colorbar for edge betweenness
    sm = ScalarMappable(cmap=plt.cm.Reds, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Edge Betweenness Centrality')
    
    # Add a title
    ax.set_title('Critical Connections in Combined Network\n(Interlayer + Intralayer over Common Nodes)', fontsize=12)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Adjust layout
    plt.tight_layout()
    
    logger.info(f"Created bottleneck visualization with {len(G.nodes)} nodes and {len(G.edges)} edges") 