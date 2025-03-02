import logging
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from itertools import combinations

def create_interlayer_path_similarity(fig, visible_links, node_ids, node_clusters, nodes_per_layer, layers, 
                                     small_font, medium_font, visible_layer_indices, cluster_colors, 
                                     selected_cluster=None):
    """
    Create a visualization of path-based similarity between clusters using interlayer edges.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to draw on
    visible_links : list
        List of (start_idx, end_idx) tuples representing visible links
    node_ids : list
        List of node IDs
    node_clusters : dict
        Dictionary mapping node IDs to cluster assignments
    nodes_per_layer : int
        Number of nodes per layer
    layers : list
        List of layer names
    small_font : dict
        Dictionary with fontsize for small text
    medium_font : dict
        Dictionary with fontsize for medium text
    visible_layer_indices : list
        List of indices of visible layers
    cluster_colors : dict
        Dictionary mapping cluster IDs to colors
    selected_cluster : int or None
        If specified, focus on this cluster's similarity to others
    """
    logging.info(f"Creating interlayer path similarity visualization with {len(visible_links)} visible links")
    
    # Filter visible layers if specified
    if visible_layer_indices is not None:
        visible_layers = set(visible_layer_indices)
    else:
        visible_layers = set(range(len(layers)))
    
    # Create a graph from the visible links
    G = nx.Graph()
    
    # Add nodes with attributes
    for node_id, cluster in node_clusters.items():
        if node_id in node_ids:
            node_idx = node_ids.index(node_id)
            layer_idx = node_idx // nodes_per_layer
            
            if layer_idx in visible_layers:
                G.add_node(node_idx, cluster=cluster, layer=layer_idx, node_id=node_id)
    
    # Add edges
    interlayer_edges = []
    for start_idx, end_idx in visible_links:
        if start_idx in G.nodes and end_idx in G.nodes:
            start_layer = G.nodes[start_idx]['layer']
            end_layer = G.nodes[end_idx]['layer']
            
            # Only add interlayer edges
            if start_layer != end_layer:
                G.add_edge(start_idx, end_idx)
                interlayer_edges.append((start_idx, end_idx))
    
    logging.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} interlayer edges")
    
    # Get unique clusters
    unique_clusters = sorted(set(node_clusters.values()))
    
    # If no clusters or no interlayer edges, display a message and return
    if not unique_clusters or not interlayer_edges:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No data to display: No clusters or interlayer edges found", 
                ha='center', va='center', fontsize=medium_font['fontsize'])
        ax.axis('off')
        return
    
    # Group nodes by cluster
    nodes_by_cluster = defaultdict(list)
    for node, attrs in G.nodes(data=True):
        nodes_by_cluster[attrs['cluster']].append(node)
    
    # If a specific cluster is selected, focus on that cluster
    if selected_cluster is not None and selected_cluster in nodes_by_cluster:
        create_single_cluster_view(fig, G, selected_cluster, nodes_by_cluster, unique_clusters, 
                                  visible_layers, layers, cluster_colors, small_font, medium_font)
    else:
        create_all_clusters_view(fig, G, nodes_by_cluster, unique_clusters, cluster_colors, small_font, medium_font)

def create_all_clusters_view(fig, G, nodes_by_cluster, unique_clusters, cluster_colors, small_font, medium_font):
    """Create a view showing path-based similarity between all clusters"""
    logging.info("Creating all clusters view")
    
    # Create a single subplot
    ax = fig.add_subplot(111)
    
    # Calculate path-based similarity between all cluster pairs
    similarity_matrix = np.zeros((len(unique_clusters), len(unique_clusters)))
    
    # Compute shortest paths between all nodes
    try:
        # Use all_pairs_shortest_path_length for efficiency
        path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        
        # Calculate average path length between clusters
        for i, cluster1 in enumerate(unique_clusters):
            for j, cluster2 in enumerate(unique_clusters):
                if i == j:
                    # Self-similarity is 1.0
                    similarity_matrix[i, j] = 1.0
                else:
                    # Get nodes in each cluster
                    nodes1 = nodes_by_cluster[cluster1]
                    nodes2 = nodes_by_cluster[cluster2]
                    
                    # Calculate average path length
                    path_lengths_sum = 0
                    count = 0
                    
                    for node1 in nodes1:
                        for node2 in nodes2:
                            try:
                                # Get path length if nodes are connected
                                length = path_lengths[node1][node2]
                                path_lengths_sum += length
                                count += 1
                            except KeyError:
                                # Nodes are not connected, skip
                                pass
                    
                    # Calculate similarity as 1 / (1 + average path length)
                    if count > 0:
                        avg_path_length = path_lengths_sum / count
                        similarity_matrix[i, j] = 1.0 / (1.0 + avg_path_length)
                    else:
                        # No paths between clusters
                        similarity_matrix[i, j] = 0.0
        
        # Create a heatmap of the similarity matrix
        im = ax.imshow(similarity_matrix, cmap='Blues', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Similarity (1 / (1 + avg path length))', fontdict=small_font)
        
        # Add labels
        ax.set_xticks(np.arange(len(unique_clusters)))
        ax.set_yticks(np.arange(len(unique_clusters)))
        ax.set_xticklabels([f'C{c}' for c in unique_clusters], fontdict=small_font)
        ax.set_yticklabels([f'C{c}' for c in unique_clusters], fontdict=small_font)
        
        # Rotate x labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(unique_clusters)):
            for j in range(len(unique_clusters)):
                if similarity_matrix[i, j] > 0.2 or i == j:  # Only annotate significant values
                    text = ax.text(j, i, f"{similarity_matrix[i, j]:.2f}",
                                  ha="center", va="center", color="black" if similarity_matrix[i, j] < 0.5 else "white",
                                  fontsize=small_font['fontsize'] - 1)
        
        # Set title
        ax.set_title("Interlayer Path-based Similarity Between Clusters", fontdict=medium_font)
        
        # Add grid lines
        ax.set_xticks(np.arange(-.5, len(unique_clusters), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(unique_clusters), 1), minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        
    except nx.NetworkXNoPath:
        # Handle case where there are no paths between some nodes
        ax.text(0.5, 0.5, "No paths between some clusters", ha='center', va='center', fontsize=medium_font['fontsize'])
        ax.axis('off')

def create_single_cluster_view(fig, G, selected_cluster, nodes_by_cluster, unique_clusters, 
                              visible_layers, layers, cluster_colors, small_font, medium_font):
    """Create a view focusing on a single cluster's similarity to others, broken down by layer pairs"""
    logging.info(f"Creating single cluster view for cluster {selected_cluster}")
    
    # Get visible layer names
    visible_layer_names = [layers[idx] for idx in visible_layers if idx < len(layers)]
    
    # Calculate number of layer pairs
    layer_pairs = list(combinations(sorted(visible_layers), 2))
    num_pairs = len(layer_pairs)
    
    # Determine grid dimensions
    if num_pairs <= 3:
        rows, cols = 1, num_pairs
    elif num_pairs <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3
    
    # Create subplots
    axes = []
    for i in range(min(rows * cols, num_pairs)):
        ax = fig.add_subplot(rows, cols, i + 1)
        axes.append(ax)
    
    # Add a main title
    fig.suptitle(f"Interlayer Path Similarity for Cluster {selected_cluster}", 
                fontsize=medium_font['fontsize'] + 2, y=0.98)
    
    # Compute shortest paths between all nodes
    try:
        path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        
        # For each layer pair, calculate similarity between selected cluster and all others
        for i, (layer1_idx, layer2_idx) in enumerate(layer_pairs[:len(axes)]):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get layer names
            layer1_name = layers[layer1_idx] if layer1_idx < len(layers) else f"Layer {layer1_idx}"
            layer2_name = layers[layer2_idx] if layer2_idx < len(layers) else f"Layer {layer2_idx}"
            
            # Calculate similarity matrix for this layer pair
            similarity_matrix = np.zeros(len(unique_clusters))
            
            # Get nodes in selected cluster
            selected_nodes = nodes_by_cluster[selected_cluster]
            
            # Filter nodes by layer
            selected_nodes_layer1 = [n for n in selected_nodes if G.nodes[n]['layer'] == layer1_idx]
            selected_nodes_layer2 = [n for n in selected_nodes if G.nodes[n]['layer'] == layer2_idx]
            
            # For each other cluster, calculate similarity
            for j, other_cluster in enumerate(unique_clusters):
                if other_cluster == selected_cluster:
                    # Self-similarity is 1.0
                    similarity_matrix[j] = 1.0
                else:
                    # Get nodes in other cluster
                    other_nodes = nodes_by_cluster[other_cluster]
                    
                    # Filter nodes by layer
                    other_nodes_layer1 = [n for n in other_nodes if G.nodes[n]['layer'] == layer1_idx]
                    other_nodes_layer2 = [n for n in other_nodes if G.nodes[n]['layer'] == layer2_idx]
                    
                    # Calculate path lengths between layers
                    path_lengths_sum = 0
                    count = 0
                    
                    # Layer 1 of selected to Layer 2 of other
                    for node1 in selected_nodes_layer1:
                        for node2 in other_nodes_layer2:
                            try:
                                length = path_lengths[node1][node2]
                                path_lengths_sum += length
                                count += 1
                            except KeyError:
                                pass
                    
                    # Layer 2 of selected to Layer 1 of other
                    for node1 in selected_nodes_layer2:
                        for node2 in other_nodes_layer1:
                            try:
                                length = path_lengths[node1][node2]
                                path_lengths_sum += length
                                count += 1
                            except KeyError:
                                pass
                    
                    # Calculate similarity
                    if count > 0:
                        avg_path_length = path_lengths_sum / count
                        similarity_matrix[j] = 1.0 / (1.0 + avg_path_length)
                    else:
                        similarity_matrix[j] = 0.0
            
            # Create a bar chart of the similarity matrix
            bars = ax.bar(range(len(unique_clusters)), similarity_matrix, 
                         color=[cluster_colors.get(c, 'gray') for c in unique_clusters])
            
            # Add labels
            ax.set_xticks(range(len(unique_clusters)))
            ax.set_xticklabels([f'C{c}' for c in unique_clusters], fontdict=small_font, rotation=45, ha='right')
            
            # Set title and labels
            ax.set_title(f"{layer1_name} ↔ {layer2_name}", fontdict=small_font)
            ax.set_ylabel('Similarity', fontdict=small_font)
            
            # Add text annotations
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0.1:  # Only annotate significant values
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f"{height:.2f}", ha='center', va='bottom', fontsize=small_font['fontsize'] - 1)
            
            # Highlight the selected cluster
            selected_idx = unique_clusters.index(selected_cluster)
            bars[selected_idx].set_edgecolor('black')
            bars[selected_idx].set_linewidth(2)
            
            # Set y-axis limit
            ax.set_ylim(0, 1.1)
        
        # Hide any unused subplots
        for i in range(len(layer_pairs), len(axes)):
            axes[i].axis('off')
        
    except nx.NetworkXNoPath:
        # Handle case where there are no paths between some nodes
        for ax in axes:
            ax.clear()
            ax.text(0.5, 0.5, "No paths between some nodes", ha='center', va='center', fontsize=small_font['fontsize'])
            ax.axis('off') 