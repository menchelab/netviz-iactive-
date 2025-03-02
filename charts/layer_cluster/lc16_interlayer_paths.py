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
    
    This visualization builds a custom network by duplicating each node for each layer it's in,
    using the naming convention <layer>_<node>. This creates a network where the duplicated nodes
    connect interlayer and intralayer edges.
    
    For interlayer edges, all possible connections are created between duplicated nodes.
    For intralayer edges, only existing edges from the original network are added.
    
    The analysis focuses on:
    1. Path lengths: The shortest paths between layers
    2. Betweenness: Which nodes serve as bridges between layers
    3. Bottlenecks: Critical connections that, if removed, would disconnect layers
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to draw the visualization on
    visible_links : list
        List of (source_idx, target_idx) tuples representing visible edges
    node_ids : list
        List of all node IDs in the network
    node_clusters : dict
        Dictionary mapping node IDs to cluster assignments
    nodes_per_layer : dict or int
        Dictionary mapping layer indices to lists of node IDs, or an integer representing
        the number of nodes per layer if all layers have the same number of nodes
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
        visible_layers = visible_layer_indices if visible_layer_indices is not None else list(range(len(layers)))
        
        logging.info(f"Creating custom network for interlayer path analysis with {len(visible_links)} visible links")
        
        # Create a new graph with duplicated nodes
        G = nx.Graph()
        
        # Check if nodes_per_layer is an integer or a dictionary
        if isinstance(nodes_per_layer, int):
            # If it's an integer, create a dictionary mapping layer indices to node ranges
            nodes_per_layer_dict = {}
            for layer_idx in range(len(layers)):
                start_idx = layer_idx * nodes_per_layer
                end_idx = start_idx + nodes_per_layer
                nodes_per_layer_dict[layer_idx] = [node_ids[i] for i in range(start_idx, end_idx) if i < len(node_ids)]
        else:
            # If it's already a dictionary, use it directly
            nodes_per_layer_dict = nodes_per_layer
        
        # Create a mapping from node index to node ID
        node_idx_to_id = {i: node_id for i, node_id in enumerate(node_ids)}
        
        # Create a mapping from node ID to node index
        node_id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        
        # Track which layers each node appears in
        node_layers = defaultdict(list)
        
        # First, identify which layers each node appears in
        for layer_idx, node_list in nodes_per_layer_dict.items():
            if layer_idx in visible_layers:
                for node_id in node_list:
                    if node_id in node_ids:
                        node_layers[node_id].append(layer_idx)
        
        # Create duplicated nodes for each layer a node appears in
        duplicated_nodes = {}  # Maps original node ID to list of duplicated node IDs
        
        for node_id, layers_list in node_layers.items():
            duplicated_nodes[node_id] = []
            for layer_idx in layers_list:
                # Create a new node ID in the format <layer>_<node>
                new_node_id = f"{layer_idx}_{node_id}"
                duplicated_nodes[node_id].append(new_node_id)
                
                # Add the node to the graph with attributes
                cluster = node_clusters.get(node_id, 0)  # Default to cluster 0 if not found
                G.add_node(new_node_id, 
                          original_id=node_id,
                          cluster=cluster, 
                          layer=layer_idx)
        
        logging.info(f"Created {len(G.nodes)} duplicated nodes from {len(node_layers)} original nodes")
        
        # Store original network edges
        original_edges = set()
        for source_idx, target_idx in visible_links:
            source_id = node_idx_to_id[source_idx]
            target_id = node_idx_to_id[target_idx]
            original_edges.add((source_id, target_id))
            original_edges.add((target_id, source_id))  # Add both directions since it's an undirected graph
        
        # Add intralayer edges (only for existing edges in the original network)
        intralayer_edges = []
        for (source_id, target_id) in original_edges:
            # Find the layer(s) where both nodes exist
            common_layers = set(node_layers[source_id]) & set(node_layers[target_id])
            
            for layer_idx in common_layers:
                # Create the duplicated node IDs
                source_node = f"{layer_idx}_{source_id}"
                target_node = f"{layer_idx}_{target_id}"
                
                # Add the intralayer edge
                G.add_edge(source_node, target_node, edge_type="intralayer")
                intralayer_edges.append((source_node, target_node))
        
        # Add interlayer edges (between all duplicated nodes of the same original node)
        interlayer_edges = []
        for source_idx, target_idx in visible_links:
            source_id = node_idx_to_id[source_idx]
            target_id = node_idx_to_id[target_idx]
            
            # Get the layers for these nodes
            source_layers = node_layers[source_id]
            target_layers = node_layers[target_id]
            
            # Connect nodes across different layers
            for source_layer in source_layers:
                for target_layer in target_layers:
                    if source_layer != target_layer:  # Only connect across different layers
                        # Create the duplicated node IDs
                        source_node = f"{source_layer}_{source_id}"
                        target_node = f"{target_layer}_{target_id}"
                        
                        # Add the interlayer edge
                        G.add_edge(source_node, target_node, edge_type="interlayer")
                        interlayer_edges.append((source_node, target_node))
        
        logging.info(f"Added {len(interlayer_edges)} interlayer edges and {len(intralayer_edges)} intralayer edges")
        
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
            if node in G.nodes and 'layer' in G.nodes[node]:
                layer_idx = G.nodes[node]['layer']
                if layer_idx < len(layers):
                    G.nodes[node]['layer_name'] = layers[layer_idx]
                else:
                    G.nodes[node]['layer_name'] = f"Layer {layer_idx}"
        
        # Check if we have any nodes or edges
        if len(G.nodes) == 0 or len(G.edges) == 0:
            ax.text(0.5, 0.5, "No nodes or edges found for analysis", 
                   ha='center', va='center')
            ax.axis('off')
            return ax
        
        # Perform the selected analysis
        if analysis_type == "path_length":
            _analyze_path_lengths(ax, G, visible_layer_indices, layers, node_clusters, cluster_colors)
        elif analysis_type == "betweenness":
            _analyze_betweenness(ax, G, visible_layer_indices, layers, node_clusters, cluster_colors)
        elif analysis_type == "bottleneck":
            _analyze_bottlenecks(ax, G, visible_layer_indices, layers, node_clusters, cluster_colors)
        else:
            ax.text(0.5, 0.5, f"Unknown analysis type: {analysis_type}", 
                   ha='center', va='center')
            
        # Set title based on analysis type
        title_map = {
            "path_length": "Interlayer Path Analysis (Duplicated Nodes)",
            "betweenness": "Node Betweenness in Duplicated Network",
            "bottleneck": "Critical Connections in Duplicated Network"
        }
        ax.set_title(title_map.get(analysis_type, f"Duplicated Network Analysis: {analysis_type}"))
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        return ax
        
    except Exception as e:
        logging.error(f"Error in create_interlayer_path_analysis: {str(e)}")
        ax.text(0.5, 0.5, f"Error creating interlayer path analysis: {str(e)}", 
               ha='center', va='center')
        return ax

def _analyze_path_lengths(ax, G, visible_layer_indices, layers, node_clusters, cluster_colors):
    """
    Analyze and visualize the shortest paths between layers in the duplicated node network.
    """
    logger = logging.getLogger(__name__)
    logger.info("Analyzing path lengths in the duplicated node network")
    
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
    ax.set_title("Average Shortest Path Length Between Layers\n(Using Duplicated Nodes Network)", fontsize=12)

def _analyze_betweenness(ax, G, visible_layer_indices, layers, node_clusters, cluster_colors):
    """Analyze and visualize betweenness centrality of nodes in the duplicated node network"""
    logger = logging.getLogger(__name__)
    logger.info("Analyzing betweenness centrality in the duplicated node network")
    
    # Calculate betweenness centrality for all nodes
    betweenness = nx.betweenness_centrality(G, weight='weight')
    
    # Group betweenness by layer
    layer_betweenness = defaultdict(float)
    
    for node, bc in betweenness.items():
        layer = G.nodes[node]['layer']
        layer_betweenness[layer] += bc
    
    # Get unique layers
    unique_layers = sorted(layer_betweenness.keys())
    
    # Create a visualization of the betweenness values
    # We'll use a bar plot with layers on the x-axis and betweenness values on the y-axis
    
    # Set up the bar positions
    x = np.arange(len(unique_layers))
    
    # Create the bars
    ax.bar(x, [layer_betweenness[layer] for layer in unique_layers], color='skyblue')
    
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
    
    # Add a title
    ax.set_title('Node Betweenness Centrality by Layer\n(Using Duplicated Nodes Network)', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    logger.info(f"Created betweenness visualization with {len(unique_layers)} layers")

def _analyze_bottlenecks(ax, G, visible_layer_indices, layers, node_clusters, cluster_colors):
    """Analyze and visualize bottleneck connections in the duplicated node network"""
    logger = logging.getLogger(__name__)
    logger.info("Analyzing bottleneck connections in the duplicated node network")
    
    # Calculate edge betweenness centrality to identify bottlenecks
    edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight')
    
    # Normalize edge betweenness by the maximum value
    max_betweenness = max(edge_betweenness.values()) if edge_betweenness else 1.0
    normalized_betweenness = {edge: bc / max_betweenness for edge, bc in edge_betweenness.items()}
    
    # Create a visualization of the network with bottlenecks highlighted
    # Position nodes using a spring layout, grouped by layer
    pos = nx.spring_layout(G, seed=42)
    
    # Draw the graph
    # 1. Draw edges with width and color based on betweenness and edge type
    for edge, betweenness in normalized_betweenness.items():
        u, v = edge
        
        # Get edge type (interlayer or intralayer)
        edge_type = G.edges[u, v].get('edge_type', 'unknown')
        
        # Calculate edge width based on betweenness
        width = 1 + 5 * betweenness
        
        # Calculate edge color based on type and betweenness
        if edge_type == "interlayer":
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
            edge_label = "Inter" if edge_type == "interlayer" else "Intra"
            ax.text(x + offset_x, y + offset_y, 
                   f"{edge_label}\nBC={betweenness:.2f}", 
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
        original_id = G.nodes[node].get('original_id', node)
        
        # Calculate node size based on betweenness
        betweenness = node_betweenness.get(node, 0)
        size = 100 + 500 * (betweenness / max_betweenness)
        
        # Get node color based on cluster
        if cluster_colors and cluster in cluster_colors:
            color = cluster_colors[cluster]
        else:
            # Use a default colormap if cluster colors not provided
            color = plt.cm.tab10(cluster % 10)
        
        # Draw the node
        ax.scatter(pos[node][0], pos[node][1], 
                  s=size, color=color, edgecolor='black', linewidth=1,
                  zorder=2)  # Middle zorder to draw between edges and labels
        
        # Add node label for significant nodes
        if betweenness > 0.1:
            # Extract layer and original node ID from the node name
            node_parts = node.split('_', 1)
            if len(node_parts) == 2:
                layer_str, orig_id = node_parts
            else:
                layer_str, orig_id = str(layer), str(original_id)
                
            ax.text(pos[node][0], pos[node][1], 
                   f"L{layer_str}_N{orig_id}\nBC={betweenness:.2f}", 
                   fontsize=8, 
                   ha='center', va='center',
                   zorder=4)  # Highest zorder to draw on top of everything
    
    # Add a legend for edge types
    inter_line = plt.Line2D([0], [0], color=plt.cm.Blues(0.7), linewidth=3, label='Interlayer Edge')
    intra_line = plt.Line2D([0], [0], color=plt.cm.Reds(0.7), linewidth=3, label='Intralayer Edge')
    
    ax.legend(handles=[inter_line, intra_line], loc='upper right')
    
    # Add a colorbar for edge betweenness
    sm = ScalarMappable(cmap=plt.cm.Reds, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Edge Betweenness Centrality')
    
    # Add a title
    ax.set_title('Critical Connections in Network\n(Using Duplicated Nodes <layer>_<node>)', fontsize=12)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Adjust layout
    plt.tight_layout()
    
    logger.info(f"Created bottleneck visualization with {len(G.nodes)} nodes and {len(G.edges)} edges") 