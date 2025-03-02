import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def create_layer_cluster_network_diagram(
    ax,
    visible_links,
    node_ids,
    node_clusters,
    nodes_per_layer,
    layers,
    small_font,
    medium_font,
    visible_layer_indices=None,
    cluster_colors=None,
    layout_algorithm="community",  # New parameter for layout algorithm
    aspect_ratio=1.0,  # New parameter to control aspect ratio
):
    """
    Create a network diagram showing connections between layers and clusters.
    Layers and clusters are represented as nodes, with edges representing the number of nodes
    that belong to a specific cluster in a specific layer.
    
    Parameters:
    -----------
    layout_algorithm : str
        The layout algorithm to use. Options:
        - "community": Force-directed layout with community detection
        - "bipartite": Optimized bipartite layout (layers on left, clusters on right)
        - "circular": Circular layout with edge bundling
        - "spectral": Spectral layout based on graph Laplacian
        - "spring": Standard spring layout with weight-based attraction
    aspect_ratio : float
        Controls the aspect ratio of the layout (width/height)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating layer-cluster network diagram with layout={layout_algorithm}")
    
    # Handle font parameters correctly
    if isinstance(medium_font, dict):
        medium_fontsize = medium_font.get("fontsize", 12)
    else:
        medium_fontsize = medium_font
        
    if isinstance(small_font, dict):
        small_fontsize = small_font.get("fontsize", 10)
    else:
        small_fontsize = small_font
    
    # Clear the axis and set title
    ax.clear()
    ax.set_title(f"Layer-Cluster Network ({layout_algorithm.title()} Layout)", fontsize=medium_fontsize)
    
    # Filter visible layers if specified
    if visible_layer_indices is not None:
        visible_layers = set(visible_layer_indices)
    else:
        visible_layers = set(range(len(layers)))
    
    # Get visible node indices
    visible_node_indices = set()
    for start_idx, end_idx in visible_links:
        visible_node_indices.add(start_idx)
        visible_node_indices.add(end_idx)
    
    # Count nodes by cluster and layer
    cluster_layer_counts = defaultdict(lambda: defaultdict(int))
    
    for node_idx in visible_node_indices:
        # Calculate layer index
        layer_idx = node_idx // nodes_per_layer
        
        # Skip if layer is not visible
        if layer_idx not in visible_layers:
            continue
            
        node_id = node_ids[node_idx]
        cluster = node_clusters.get(node_id, "Unknown")
        
        cluster_layer_counts[cluster][layer_idx] += 1
    
    # Check if we have any data
    if not cluster_layer_counts:
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center")
        ax.axis("off")
        return
    
    # Get unique layers and clusters
    unique_clusters = sorted(cluster_layer_counts.keys())
    unique_layer_indices = sorted(visible_layers)
    unique_layers = [layers[i] for i in unique_layer_indices if i < len(layers)]
    
    # Create a colormap for clusters if not provided
    if cluster_colors is None:
        colormap = plt.cm.tab20
        cluster_colors = {cluster: colormap(i % 20) for i, cluster in enumerate(unique_clusters)}
    
    # Create a simple graph
    G = nx.Graph()
    
    # Add layer nodes (left side)
    for layer_name in unique_layers:
        G.add_node(f"L_{layer_name}", type="layer")
    
    # Add cluster nodes (right side)
    for cluster in unique_clusters:
        G.add_node(f"C_{cluster}", type="cluster")
    
    # Add edges between layers and clusters with weights
    total_connections = 0
    for cluster, layer_dict in cluster_layer_counts.items():
        for layer_idx, count in layer_dict.items():
            if layer_idx in unique_layer_indices and layer_idx < len(layers):
                layer_name = layers[layer_idx]
                G.add_edge(f"L_{layer_name}", f"C_{cluster}", weight=count)
                total_connections += count
    
    # Get layer and cluster nodes
    layer_nodes = [n for n in G.nodes() if n.startswith("L_")]
    cluster_nodes = [n for n in G.nodes() if n.startswith("C_")]
    
    # Calculate node sizes based on total connections
    layer_sizes = {}
    for layer in unique_layers:
        layer_count = sum(cluster_layer_counts[cluster][layers.index(layer)] 
                          for cluster in unique_clusters 
                          if layers.index(layer) in cluster_layer_counts[cluster])
        layer_sizes[f"L_{layer}"] = 300 + 700 * (layer_count / total_connections)
    
    cluster_sizes = {}
    for cluster in unique_clusters:
        cluster_count = sum(cluster_layer_counts[cluster].values())
        cluster_sizes[f"C_{cluster}"] = 300 + 700 * (cluster_count / total_connections)
    
    # Choose layout algorithm
    if layout_algorithm == "bipartite":
        # Optimized bipartite layout
        pos = _create_bipartite_layout(G, layer_nodes, cluster_nodes, aspect_ratio)
    elif layout_algorithm == "circular":
        # Circular layout with edge bundling
        pos = _create_circular_layout(G, layer_nodes, cluster_nodes)
    elif layout_algorithm == "spectral":
        # Spectral layout based on graph Laplacian
        pos = _create_spectral_layout(G, layer_nodes, cluster_nodes, aspect_ratio)
    elif layout_algorithm == "spring":
        # Standard spring layout with weight-based attraction
        pos = nx.spring_layout(G, k=0.5, iterations=100, weight='weight', seed=42)
        # Adjust positions to ensure layers are on the left and clusters on the right
        _adjust_spring_layout(pos, layer_nodes, cluster_nodes, aspect_ratio)
    else:  # Default to "community"
        # Force-directed layout with community detection
        pos = _create_community_layout(G, layer_nodes, cluster_nodes, aspect_ratio)
    
    # Scale the layout to fit within the plot area
    _scale_layout(pos, padding=0.1)
    
    # Draw the edges with width based on weight and bundling for cleaner visualization
    edge_weights = [G.edges[edge]['weight'] for edge in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    
    # Sort edges by weight for better visualization (draw thicker edges first)
    edges_with_weights = [(u, v, G.edges[u, v]['weight']) for u, v in G.edges()]
    edges_with_weights.sort(key=lambda x: x[2], reverse=True)
    
    # Create a colormap for edge weights
    edge_cmap = plt.cm.Blues
    edge_norm = Normalize(vmin=0, vmax=max_weight)
    
    # Draw edges with bundling effect
    for u, v, weight in edges_with_weights:
        # Calculate edge width based on weight
        width = 0.5 + 3.5 * (weight / max_weight)
        
        # Use a color that reflects the weight
        edge_color = edge_cmap(0.2 + 0.8 * (weight / max_weight))
        
        # Draw curved edges for better visualization
        # More important connections get straighter lines
        rad = 0.2 * (1 - (weight / max_weight))
        
        # Draw the edge
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=[(u, v)], 
            width=width,
            alpha=0.7,
            edge_color=[edge_color],
            connectionstyle=f'arc3,rad={rad}',
            ax=ax
        )
        
        # Add edge weight labels with more detail
        # Position the label along the curved edge
        edge_x = (pos[u][0] + pos[v][0]) / 2
        edge_y = (pos[u][1] + pos[v][1]) / 2
        
        # Adjust label position for curved edges
        if u.startswith('L_') and v.startswith('C_'):
            # Adjust based on curvature
            edge_x += rad * (pos[v][1] - pos[u][1]) * 0.5
            edge_y -= rad * (pos[v][0] - pos[u][0]) * 0.5
        
        # Calculate percentage of total connections
        percentage = (weight / sum(edge_weights)) * 100
        
        # Only show labels for edges with significant weight to reduce clutter
        if weight > max_weight * 0.05 or percentage > 2.0:
            ax.text(edge_x, edge_y, f"{weight}\n({percentage:.1f}%)", 
                    fontsize=small_fontsize-1, 
                    ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    # Draw layer nodes
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=layer_nodes,
        node_color="lightblue",
        node_size=[layer_sizes.get(n, 500) for n in layer_nodes],
        alpha=0.8,
        edgecolors='black',
        ax=ax
    )
    
    # Draw cluster nodes with their respective colors
    for cluster in unique_clusters:
        node = f"C_{cluster}"
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=[node],
            node_color=[cluster_colors.get(cluster, "gray")],
            node_size=[cluster_sizes.get(node, 500)],
            alpha=0.8,
            edgecolors='black',
            ax=ax
        )
    
    # Add node labels with counts
    layer_labels = {}
    for layer in unique_layers:
        layer_count = sum(cluster_layer_counts[cluster][layers.index(layer)] 
                          for cluster in unique_clusters 
                          if layers.index(layer) in cluster_layer_counts[cluster])
        layer_labels[f"L_{layer}"] = f"{layer}\n({layer_count})"
    
    nx.draw_networkx_labels(
        G, pos, 
        labels=layer_labels,
        font_size=small_fontsize,
        font_weight='bold',
        ax=ax
    )
    
    cluster_labels = {}
    for cluster in unique_clusters:
        cluster_count = sum(cluster_layer_counts[cluster].values())
        cluster_labels[f"C_{cluster}"] = f"{cluster}\n({cluster_count})"
    
    nx.draw_networkx_labels(
        G, pos, 
        labels=cluster_labels,
        font_size=small_fontsize,
        font_weight='bold',
        ax=ax
    )
    
    # Create a legend for clusters
    legend_elements = []
    for cluster in unique_clusters:
        legend_elements.append(
            plt.Line2D([0], [0], 
                      marker='o', 
                      color='w',
                      markerfacecolor=cluster_colors.get(cluster, "gray"),
                      markersize=10,
                      label=str(cluster))
        )
    
    # Add the legend
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=small_fontsize,
        title="Clusters",
        title_fontsize=small_fontsize
    )
    
    # Add a colorbar for edge weights
    sm = ScalarMappable(cmap=edge_cmap, norm=edge_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Connection Strength', fontsize=small_fontsize)
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add padding around the plot
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Apply tight layout with padding
    plt.tight_layout(pad=1.5)
    
    logger.info(f"Successfully created layer-cluster network with {len(unique_layers)} layers and {len(unique_clusters)} clusters") 
    return G  # Return the graph for potential further analysis

def _create_bipartite_layout(G, layer_nodes, cluster_nodes, aspect_ratio=1.0):
    """Create an optimized bipartite layout with layers on left, clusters on right"""
    pos = {}
    
    # Calculate node weights (total connections)
    node_weights = {}
    for node in G.nodes():
        node_weights[node] = sum(G.edges[node, neighbor]['weight'] for neighbor in G.neighbors(node))
    
    # Sort nodes by weight (descending)
    sorted_layer_nodes = sorted(layer_nodes, key=lambda n: node_weights.get(n, 0), reverse=True)
    sorted_cluster_nodes = sorted(cluster_nodes, key=lambda n: node_weights.get(n, 0), reverse=True)
    
    # Position layer nodes on the left
    total_layer_weight = sum(node_weights.get(n, 0) for n in layer_nodes)
    y_pos = 0.1
    for node in sorted_layer_nodes:
        weight = node_weights.get(node, 0)
        height = 0.8 * (weight / total_layer_weight) if total_layer_weight > 0 else 0.8 / len(layer_nodes)
        pos[node] = (0.2, y_pos + height/2)
        y_pos += height
    
    # Position cluster nodes on the right
    total_cluster_weight = sum(node_weights.get(n, 0) for n in cluster_nodes)
    y_pos = 0.1
    for node in sorted_cluster_nodes:
        weight = node_weights.get(node, 0)
        height = 0.8 * (weight / total_cluster_weight) if total_cluster_weight > 0 else 0.8 / len(cluster_nodes)
        pos[node] = (0.8, y_pos + height/2)
        y_pos += height
    
    # Apply aspect ratio
    for node in pos:
        x, y = pos[node]
        pos[node] = (x * aspect_ratio, y)
    
    return pos

def _create_circular_layout(G, layer_nodes, cluster_nodes):
    """Create a circular layout with layers and clusters grouped together"""
    # Create a circular layout
    pos = nx.circular_layout(G)
    
    # Adjust positions to group layers and clusters
    layer_angle = 0
    layer_step = np.pi / (len(layer_nodes) + 1)
    for i, node in enumerate(layer_nodes):
        angle = layer_angle + (i + 1) * layer_step
        pos[node] = (0.4 * np.cos(angle), 0.4 * np.sin(angle))
    
    cluster_angle = np.pi
    cluster_step = np.pi / (len(cluster_nodes) + 1)
    for i, node in enumerate(cluster_nodes):
        angle = cluster_angle + (i + 1) * cluster_step
        pos[node] = (0.4 * np.cos(angle), 0.4 * np.sin(angle))
    
    return pos

def _create_spectral_layout(G, layer_nodes, cluster_nodes, aspect_ratio=1.0):
    """Create a spectral layout based on graph Laplacian"""
    # Create initial spectral layout
    pos = nx.spectral_layout(G)
    
    # Adjust positions to ensure layers and clusters are separated
    layer_center = np.mean([pos[n] for n in layer_nodes], axis=0)
    cluster_center = np.mean([pos[n] for n in cluster_nodes], axis=0)
    
    # Calculate direction vector between centers
    direction = cluster_center - layer_center
    if np.linalg.norm(direction) < 1e-6:
        direction = np.array([1.0, 0.0])  # Default direction if centers are too close
    else:
        direction = direction / np.linalg.norm(direction)
    
    # Shift positions to separate layers and clusters
    for node in layer_nodes:
        pos[node] = pos[node] - 0.2 * direction
    
    for node in cluster_nodes:
        pos[node] = pos[node] + 0.2 * direction
    
    # Apply aspect ratio
    for node in pos:
        x, y = pos[node]
        pos[node] = (x * aspect_ratio, y)
    
    return pos

def _create_community_layout(G, layer_nodes, cluster_nodes, aspect_ratio=1.0):
    """Create a layout that emphasizes community structure"""
    # First, detect communities
    try:
        # Try to use community detection algorithms if available
        import community as community_louvain
        partition = community_louvain.best_partition(G, weight='weight')
    except ImportError:
        # Fallback to a simple partitioning based on node type
        partition = {node: 0 if node.startswith('L_') else 1 for node in G.nodes()}
    
    # Create a spring layout with community information
    pos = nx.spring_layout(G, k=0.5, iterations=100, weight='weight', seed=42)
    
    # Group nodes by community
    communities = {}
    for node, comm in partition.items():
        if comm not in communities:
            communities[comm] = []
        communities[comm].append(node)
    
    # Adjust positions to group nodes by community
    for comm, nodes in communities.items():
        # Calculate community center
        center = np.mean([pos[n] for n in nodes], axis=0)
        
        # Move nodes closer to their community center
        for node in nodes:
            pos[node] = 0.7 * pos[node] + 0.3 * center
    
    # Ensure layers and clusters are separated
    _adjust_spring_layout(pos, layer_nodes, cluster_nodes, aspect_ratio)
    
    return pos

def _adjust_spring_layout(pos, layer_nodes, cluster_nodes, aspect_ratio=1.0):
    """Adjust a spring layout to ensure layers are on the left and clusters on the right"""
    # Calculate the center of each group
    layer_center = np.mean([pos[n] for n in layer_nodes], axis=0)
    cluster_center = np.mean([pos[n] for n in cluster_nodes], axis=0)
    
    # Shift the positions to ensure separation
    for node in layer_nodes:
        pos[node][0] = pos[node][0] - layer_center[0] + 0.2
    
    for node in cluster_nodes:
        pos[node][0] = pos[node][0] - cluster_center[0] + 0.8
    
    # Scale the y-positions to fit better within the plot area
    all_y = [pos[n][1] for n in pos]
    y_min, y_max = min(all_y), max(all_y)
    y_range = y_max - y_min
    
    # Scale and center y-positions to use 80% of the available space
    for node in pos:
        pos[node][1] = 0.1 + 0.8 * (pos[node][1] - y_min) / y_range if y_range > 0 else 0.5
    
    # Apply aspect ratio
    for node in pos:
        x, y = pos[node]
        pos[node] = (x * aspect_ratio, y)

def _scale_layout(pos, padding=0.1):
    """Scale the layout to fit within the plot area with padding"""
    # Find current min/max coordinates
    x_values = [p[0] for p in pos.values()]
    y_values = [p[1] for p in pos.values()]
    
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Scale factor to fit within [padding, 1-padding] in both dimensions
    scale_x = (1 - 2 * padding) / x_range if x_range > 0 else 1
    scale_y = (1 - 2 * padding) / y_range if y_range > 0 else 1
    
    # Use the smaller scale to maintain aspect ratio
    scale = min(scale_x, scale_y)
    
    # Apply scaling and centering
    for node in pos:
        # Scale and shift to center
        x = padding + scale * (pos[node][0] - x_min)
        y = padding + scale * (pos[node][1] - y_min)
        pos[node] = (x, y) 