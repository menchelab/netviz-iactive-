import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

def create_cluster_network(
    ax,
    visible_links,
    node_ids,
    node_clusters,
    nodes_per_layer,
    layers,
    small_font,
    medium_font,
    visible_layer_indices=None,
):
    """
    Create a network diagram showing connections between clusters.
    Clusters are represented as nodes, with edges representing the number of connections
    between clusters across all layers.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating cluster co-occurrence network with nodes_per_layer={nodes_per_layer}")
    
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
    ax.set_title("Cluster Network\nEdges represent direct connections between nodes in different clusters", fontsize=medium_fontsize)
    
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
    
    # Map node indices to clusters
    node_cluster_map = {}
    cluster_counts = defaultdict(int)
    
    for node_idx in visible_node_indices:
        # Calculate layer index
        layer_idx = node_idx // nodes_per_layer
        
        # Skip if layer is not visible
        if layer_idx not in visible_layers:
            continue
            
        node_id = node_ids[node_idx]
        cluster = node_clusters.get(node_id, "Unknown")
        
        node_cluster_map[node_idx] = cluster
        cluster_counts[cluster] += 1
    
    # Count connections between clusters
    cluster_connections = {}
    for start_idx, end_idx in visible_links:
        if start_idx < len(node_ids) and end_idx < len(node_ids):
            start_id = node_ids[start_idx]
            end_id = node_ids[end_idx]
            
            if start_id in node_clusters and end_id in node_clusters:
                start_cluster = node_clusters[start_id]
                end_cluster = node_clusters[end_id]
                
                # Skip self-connections within the same cluster
                if start_cluster == end_cluster:
                    continue
                
                # Ensure we have a consistent order (smaller cluster ID first)
                if isinstance(start_cluster, np.integer):
                    start_cluster = int(start_cluster)
                if isinstance(end_cluster, np.integer):
                    end_cluster = int(end_cluster)
                
                cluster1 = min(start_cluster, end_cluster)
                cluster2 = max(start_cluster, end_cluster)
                
                # Initialize nested dictionaries if they don't exist
                if cluster1 not in cluster_connections:
                    cluster_connections[cluster1] = {}
                if cluster2 not in cluster_connections[cluster1]:
                    cluster_connections[cluster1][cluster2] = 0
                
                # Increment the connection count
                cluster_connections[cluster1][cluster2] += 1
    
    # Get unique clusters
    unique_clusters = sorted(cluster_counts.keys())
    
    # Check if we have any data
    if not unique_clusters or len(unique_clusters) < 2:
        ax.text(0.5, 0.5, "Not enough clusters to display", ha="center", va="center")
        ax.axis("off")
        return
    
    # Create a colormap for clusters
    colormap = plt.cm.tab20
    cluster_colors = {cluster: colormap(i % 20) for i, cluster in enumerate(unique_clusters)}
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes (clusters)
    for cluster in unique_clusters:
        # Count total nodes in this cluster
        cluster_size = sum(1 for c in node_clusters.values() if c == cluster)
        G.add_node(cluster, size=cluster_size)
    
    # Add edges between clusters that share connections
    for cluster1 in unique_clusters:
        for cluster2 in unique_clusters:
            if cluster1 < cluster2:  # Avoid duplicates
                # Convert numpy integers to Python integers if needed
                c1 = int(cluster1) if isinstance(cluster1, np.integer) else cluster1
                c2 = int(cluster2) if isinstance(cluster2, np.integer) else cluster2
                
                # Check if there's a connection between these clusters
                weight = 0
                if c1 in cluster_connections and c2 in cluster_connections[c1]:
                    weight = cluster_connections[c1][c2]
                
                if weight > 0:
                    G.add_edge(cluster1, cluster2, weight=weight)
    
    # Create a layout that emphasizes communities
    pos = nx.spring_layout(G, k=1.0, iterations=100, weight='weight', seed=42)
    
    # Draw the nodes with size based on cluster size
    node_sizes = [G.nodes[n]['size'] * 100 for n in G.nodes()]
    max_size = max(node_sizes) if node_sizes else 100
    normalized_sizes = [max(100, (size / max_size) * 500) for size in node_sizes]
    
    # Draw nodes with colors from the cluster_colors dictionary
    node_colors = [cluster_colors.get(n, 'gray') for n in G.nodes()]
    
    nx.draw_networkx_nodes(
        G, pos,
        node_size=normalized_sizes,
        node_color=node_colors,
        alpha=0.8,
        ax=ax
    )
    
    # Draw the edges with width based on weight
    edge_weights = [G.edges[edge]['weight'] for edge in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    
    for (u, v, data) in G.edges(data=True):
        weight = data.get('weight', 1)
        width = 0.5 + 2.5 * (weight / max_weight)
        # Use a color that reflects the weight
        edge_color = plt.cm.Purples(0.2 + 0.8 * (weight / max_weight))
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=[(u, v)], 
            width=width,
            alpha=0.7,
            edge_color=[edge_color],
            ax=ax
        )
        
        # Add edge weight labels with percentage
        edge_x = (pos[u][0] + pos[v][0]) / 2
        edge_y = (pos[u][1] + pos[v][1]) / 2
        
        # Calculate percentage of total connections
        percentage = (weight / sum(edge_weights)) * 100
        
        ax.text(edge_x, edge_y, f"{weight}\n({percentage:.1f}%)", 
                fontsize=small_fontsize-1, 
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    # Draw node labels
    nx.draw_networkx_labels(
        G, pos,
        labels={n: f"Cluster {n}\n({G.nodes[n]['size']} nodes)" for n in G.nodes()},
        font_size=small_fontsize,
        font_weight='bold',
        ax=ax
    )
    
    # Create a legend for clusters
    legend_elements = []
    for cluster in unique_clusters[:min(5, len(unique_clusters))]:  # Limit to top 5 clusters
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
    
    # Add a note about what the edges represent
    ax.text(
        0.5, -0.05,
        "Edge weights represent the number of direct connections between nodes in different clusters",
        transform=ax.transAxes,
        ha='center',
        fontsize=small_fontsize,
        bbox=dict(facecolor='lightyellow', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
    )
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    logger.info(f"Successfully created cluster co-occurrence network with {len(unique_clusters)} clusters") 