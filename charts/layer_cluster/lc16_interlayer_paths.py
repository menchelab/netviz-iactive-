import logging
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def create_interlayer_path_analysis(
    ax,
    visible_links,
    node_ids,
    node_clusters,
    nodes_per_layer,
    layers,
    visible_layer_indices=None,
    cluster_colors=None,
    analysis_type="path_length"  # Options: "path_length", "betweenness", "bottleneck"
):
    """
    Analyze and visualize interlayer paths between clusters.
    
    This visualization shows how clusters are connected across layers, focusing on:
    1. Path lengths: The shortest paths between clusters across different layers
    2. Betweenness: Which clusters serve as bridges between layers
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
    """
    try:
        # Create a graph from the visible links
        G = nx.Graph()
        
        # Add nodes with attributes
        for node_id in node_ids:
            if node_id in node_clusters:
                # Find which layer this node belongs to
                layer_idx = None
                for layer_idx, nodes in nodes_per_layer.items():
                    if node_id in nodes:
                        break
                
                if layer_idx is not None:
                    G.add_node(node_id, 
                              cluster=node_clusters[node_id], 
                              layer=layer_idx)
        
        # Add edges
        for source_id, target_id in visible_links:
            if source_id in G.nodes and target_id in G.nodes:
                G.add_edge(source_id, target_id)
        
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
        
        # Perform the selected analysis
        if analysis_type == "path_length":
            _analyze_path_lengths(ax, G, visible_layers, layers, node_clusters, cluster_colors)
        elif analysis_type == "betweenness":
            _analyze_betweenness(ax, G, visible_layers, layers, node_clusters, cluster_colors)
        elif analysis_type == "bottleneck":
            _analyze_bottlenecks(ax, G, visible_layers, layers, node_clusters, cluster_colors)
        else:
            ax.text(0.5, 0.5, f"Unknown analysis type: {analysis_type}", 
                   ha='center', va='center')
            
        # Set title based on analysis type
        title_map = {
            "path_length": "Interlayer Path Length Analysis",
            "betweenness": "Cluster Betweenness in Interlayer Paths",
            "bottleneck": "Interlayer Path Bottleneck Analysis"
        }
        ax.set_title(title_map.get(analysis_type, f"Interlayer Path Analysis: {analysis_type}"))
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        return True
        
    except Exception as e:
        logging.error(f"Error in create_interlayer_path_analysis: {str(e)}")
        ax.text(0.5, 0.5, f"Error creating interlayer path analysis: {str(e)}", 
               ha='center', va='center')
        return False

def _analyze_path_lengths(ax, G, visible_layers, layers, node_clusters, cluster_colors):
    """
    Analyze and visualize the shortest paths between clusters across different layers.
    """
    logger = logging.getLogger(__name__)
    logger.info("Analyzing interlayer path lengths between clusters")
    
    # Group nodes by layer and cluster
    layer_cluster_nodes = defaultdict(lambda: defaultdict(list))
    for node in G.nodes:
        layer = G.nodes[node]['layer']
        cluster = G.nodes[node]['cluster']
        layer_cluster_nodes[layer][cluster].append(node)
    
    # Calculate average shortest path lengths between clusters in different layers
    path_lengths = {}
    
    # Get unique layers and clusters
    unique_layers = sorted(visible_layers)
    unique_clusters = sorted(set(node_clusters.values()))
    
    # Create a matrix to store path lengths
    # Format: (source_layer, source_cluster, target_layer, target_cluster) -> avg_path_length
    for source_layer in unique_layers:
        for source_cluster in unique_clusters:
            # Skip if no nodes in this layer-cluster combination
            if not layer_cluster_nodes[source_layer][source_cluster]:
                continue
                
            for target_layer in unique_layers:
                # Skip same layer
                if source_layer == target_layer:
                    continue
                    
                for target_cluster in unique_clusters:
                    # Skip if no nodes in this layer-cluster combination
                    if not layer_cluster_nodes[target_layer][target_cluster]:
                        continue
                    
                    # Calculate shortest paths between all node pairs
                    total_length = 0
                    count = 0
                    
                    for source_node in layer_cluster_nodes[source_layer][source_cluster]:
                        for target_node in layer_cluster_nodes[target_layer][target_cluster]:
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
                        path_lengths[(source_layer, source_cluster, target_layer, target_cluster)] = avg_path_length
    
    # Check if we have any path data
    if not path_lengths:
        ax.text(0.5, 0.5, "No paths found between clusters in different layers", 
                ha="center", va="center")
        ax.axis("off")
        return
    
    # Create a visualization of the path lengths
    # We'll use a heatmap-like visualization with layers on x and y axes
    # and clusters represented as cells within each layer-layer block
    
    # Get the number of unique layers and clusters
    n_layers = len(unique_layers)
    n_clusters = len(unique_clusters)
    
    # Create a grid for the heatmap
    grid = np.full((n_layers * n_clusters, n_layers * n_clusters), np.nan)
    
    # Fill the grid with path lengths
    for (source_layer_idx, source_cluster, target_layer_idx, target_cluster), path_length in path_lengths.items():
        # Convert layer indices to positions in the sorted unique_layers list
        source_layer_pos = unique_layers.index(source_layer_idx)
        target_layer_pos = unique_layers.index(target_layer_idx)
        
        # Convert cluster indices to positions in the sorted unique_clusters list
        source_cluster_pos = unique_clusters.index(source_cluster)
        target_cluster_pos = unique_clusters.index(target_cluster)
        
        # Calculate grid positions
        row = source_layer_pos * n_clusters + source_cluster_pos
        col = target_layer_pos * n_clusters + target_cluster_pos
        
        # Set the value in the grid
        grid[row, col] = path_length
    
    # Create a custom colormap from blue (short paths) to red (long paths)
    cmap = LinearSegmentedColormap.from_list('path_length_cmap', ['#2c7bb6', '#ffffbf', '#d7191c'])
    
    # Create the heatmap
    im = ax.imshow(grid, cmap=cmap, interpolation='nearest')
    
    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Average Path Length')
    
    # Set ticks and labels
    # Major ticks for layers
    layer_ticks = [i * n_clusters + n_clusters // 2 for i in range(n_layers)]
    layer_labels = [layers[i] if i < len(layers) else f"Layer {i}" for i in unique_layers]
    
    # Set x and y ticks for layers
    ax.set_xticks(layer_ticks)
    ax.set_yticks(layer_ticks)
    ax.set_xticklabels(layer_labels, fontsize=10, rotation=45, ha="right")
    ax.set_yticklabels(layer_labels, fontsize=10)
    
    # Add minor ticks for clusters
    minor_ticks = []
    for i in range(n_layers):
        for j in range(n_clusters):
            minor_ticks.append(i * n_clusters + j)
    
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    
    # Add grid lines to separate layers
    for i in range(1, n_layers):
        ax.axhline(y=i * n_clusters - 0.5, color='black', linestyle='-', linewidth=1)
        ax.axvline(x=i * n_clusters - 0.5, color='black', linestyle='-', linewidth=1)
    
    # Add cluster labels in the first column and row
    for i, cluster in enumerate(unique_clusters):
        # Use cluster colors if provided
        text_color = 'black'
        if cluster_colors and cluster in cluster_colors:
            text_color = 'white' if sum(cluster_colors[cluster][:3]) < 1.5 else 'black'
        
        # Add cluster labels to the first column
        for j in range(n_layers):
            ax.text(-0.5, j * n_clusters + i, str(cluster), 
                   ha='right', va='center', fontsize=8, color=text_color)
        
        # Add cluster labels to the first row
        for j in range(n_layers):
            ax.text(j * n_clusters + i, -0.5, str(cluster), 
                   ha='center', va='bottom', fontsize=8, color=text_color, 
                   rotation=90)
    
    # Set axis labels
    ax.set_xlabel('Target Layer-Cluster')
    ax.set_ylabel('Source Layer-Cluster')
    
    # Add a title
    ax.set_title('Average Shortest Path Length Between Clusters Across Layers')
    
    # Adjust layout
    plt.tight_layout()
    
    logger.info(f"Created path length heatmap with {n_layers} layers and {n_clusters} clusters")

def _analyze_betweenness(ax, G, visible_layers, layers, node_clusters, cluster_colors):
    """Analyze and visualize betweenness centrality of clusters as bridges between layers"""
    logger = logging.getLogger(__name__)
    logger.info("Analyzing betweenness centrality of clusters as bridges between layers")
    
    # Calculate betweenness centrality for all nodes
    betweenness = nx.betweenness_centrality(G, weight='weight')
    
    # Group betweenness by cluster and layer
    cluster_layer_betweenness = defaultdict(lambda: defaultdict(float))
    for node, bc in betweenness.items():
        layer = G.nodes[node]['layer']
        cluster = G.nodes[node]['cluster']
        cluster_layer_betweenness[cluster][layer] += bc
    
    # Get unique layers and clusters
    unique_layers = sorted(visible_layers)
    unique_clusters = sorted(set(node_clusters.values()))
    
    # Create a matrix for the heatmap
    matrix = np.zeros((len(unique_clusters), len(unique_layers)))
    
    # Fill the matrix with betweenness values
    for i, cluster in enumerate(unique_clusters):
        for j, layer_idx in enumerate(unique_layers):
            matrix[i, j] = cluster_layer_betweenness[cluster][layer_idx]
    
    # Create a custom colormap from white (low betweenness) to dark blue (high betweenness)
    cmap = plt.cm.Blues
    
    # Create the heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    
    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Betweenness Centrality')
    
    # Set ticks and labels
    ax.set_xticks(range(len(unique_layers)))
    ax.set_yticks(range(len(unique_clusters)))
    
    # Set x and y labels
    layer_labels = [layers[i] if i < len(layers) else f"Layer {i}" for i in unique_layers]
    ax.set_xticklabels(layer_labels, fontsize=10, rotation=45, ha="right")
    ax.set_yticklabels([str(c) for c in unique_clusters], fontsize=10)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, len(unique_layers), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(unique_clusters), 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    
    # Add values to the cells
    for i in range(len(unique_clusters)):
        for j in range(len(unique_layers)):
            value = matrix[i, j]
            if value > 0:
                text_color = 'white' if value > matrix.max() * 0.7 else 'black'
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", 
                       fontsize=8, color=text_color)
    
    # Set axis labels
    ax.set_xlabel('Layers')
    ax.set_ylabel('Clusters')
    
    # Add a title
    ax.set_title('Cluster Betweenness Centrality by Layer')
    
    # Adjust layout
    plt.tight_layout()
    
    logger.info(f"Created betweenness heatmap with {len(unique_clusters)} clusters and {len(unique_layers)} layers")

def _analyze_bottlenecks(ax, G, visible_layers, layers, node_clusters, cluster_colors):
    """Analyze and visualize bottleneck connections between layers through clusters"""
    logger = logging.getLogger(__name__)
    logger.info("Analyzing bottleneck connections between layers through clusters")
    
    # Create a simplified graph where nodes are (layer, cluster) pairs
    # and edges represent connections between them
    layer_cluster_graph = nx.Graph()
    
    # Add nodes for each layer-cluster combination
    for node in G.nodes:
        layer = G.nodes[node]['layer']
        layer_name = G.nodes[node]['layer_name']
        cluster = G.nodes[node]['cluster']
        
        # Add the layer-cluster node if it doesn't exist
        node_id = (layer, cluster)
        if node_id not in layer_cluster_graph.nodes:
            layer_cluster_graph.add_node(node_id, 
                                        layer=layer, 
                                        layer_name=layer_name,
                                        cluster=cluster,
                                        count=0)
        
        # Increment the node count
        layer_cluster_graph.nodes[node_id]['count'] += 1
    
    # Add edges for connections between layer-cluster pairs
    for u, v in G.edges:
        u_layer = G.nodes[u]['layer']
        u_cluster = G.nodes[u]['cluster']
        v_layer = G.nodes[v]['layer']
        v_cluster = G.nodes[v]['cluster']
        
        # Skip if same layer-cluster
        if (u_layer, u_cluster) == (v_layer, v_cluster):
            continue
        
        # Add or update the edge
        u_node = (u_layer, u_cluster)
        v_node = (v_layer, v_cluster)
        
        if layer_cluster_graph.has_edge(u_node, v_node):
            layer_cluster_graph[u_node][v_node]['weight'] += 1
        else:
            layer_cluster_graph.add_edge(u_node, v_node, weight=1)
    
    # Calculate edge betweenness centrality to identify bottlenecks
    edge_betweenness = nx.edge_betweenness_centrality(layer_cluster_graph, weight='weight')
    
    # Normalize edge betweenness by the maximum value
    max_betweenness = max(edge_betweenness.values()) if edge_betweenness else 1.0
    normalized_betweenness = {edge: bc / max_betweenness for edge, bc in edge_betweenness.items()}
    
    # Create a visualization of the layer-cluster graph with bottlenecks highlighted
    pos = {}
    
    # Position nodes in a grid: layers on x-axis, clusters on y-axis
    unique_layers = sorted(visible_layers)
    unique_clusters = sorted(set(node_clusters.values()))
    
    # Calculate positions
    for node in layer_cluster_graph.nodes:
        layer, cluster = node
        layer_pos = unique_layers.index(layer) / max(1, len(unique_layers) - 1)
        cluster_pos = unique_clusters.index(cluster) / max(1, len(unique_clusters) - 1)
        pos[node] = (layer_pos, cluster_pos)
    
    # Draw the graph
    # 1. Draw edges with width and color based on betweenness
    for edge, betweenness in normalized_betweenness.items():
        u, v = edge
        # Get edge weight (number of connections)
        weight = layer_cluster_graph[u][v]['weight']
        
        # Calculate edge width based on weight and betweenness
        width = 1 + 5 * betweenness
        
        # Calculate edge color based on betweenness (red for high betweenness)
        color = plt.cm.Reds(betweenness)
        
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
            ax.text(x + offset_x, y + offset_y, 
                   f"{weight}\nBC={betweenness:.2f}", 
                   fontsize=8, 
                   ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                   zorder=3)  # Higher zorder to draw on top
    
    # 2. Draw nodes with size based on count and color based on cluster
    for node in layer_cluster_graph.nodes:
        layer, cluster = node
        count = layer_cluster_graph.nodes[node]['count']
        
        # Calculate node size based on count
        size = 100 + 500 * (count / max(1, max(layer_cluster_graph.nodes[n]['count'] 
                                             for n in layer_cluster_graph.nodes)))
        
        # Get node color based on cluster
        if cluster_colors and cluster in cluster_colors:
            color = cluster_colors[cluster]
        else:
            # Use a default colormap if cluster colors not provided
            color = plt.cm.tab10(unique_clusters.index(cluster) % 10)
        
        # Draw the node
        ax.scatter(pos[node][0], pos[node][1], 
                  s=size, color=color, edgecolor='black', linewidth=1,
                  zorder=2)  # Middle zorder to draw between edges and labels
        
        # Add node label
        layer_name = layer_cluster_graph.nodes[node]['layer_name']
        ax.text(pos[node][0], pos[node][1], 
               f"{layer_name}\nC{cluster}\n({count})", 
               fontsize=8, 
               ha='center', va='center',
               zorder=4)  # Highest zorder to draw on top of everything
    
    # Set axis limits with some padding
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add layer labels on the x-axis
    for i, layer_idx in enumerate(unique_layers):
        layer_name = layers[layer_idx] if layer_idx < len(layers) else f"Layer {layer_idx}"
        x = i / max(1, len(unique_layers) - 1)
        ax.text(x, -0.05, layer_name, 
               fontsize=8, 
               ha='center', va='top',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Add cluster labels on the y-axis
    for i, cluster in enumerate(unique_clusters):
        y = i / max(1, len(unique_clusters) - 1)
        ax.text(-0.05, y, f"Cluster {cluster}", 
               fontsize=8, 
               ha='right', va='center',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Add a colorbar for edge betweenness
    sm = ScalarMappable(cmap=plt.cm.Reds, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Edge Betweenness Centrality')
    
    # Add a title
    ax.set_title('Bottleneck Connections Between Layer-Cluster Pairs')
    
    # Adjust layout
    plt.tight_layout()
    
    logger.info(f"Created bottleneck visualization with {len(layer_cluster_graph.nodes)} layer-cluster pairs") 