import logging
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

def create_cluster_bridging_analysis(ax, visible_links, node_ids, node_clusters, nodes_per_layer, 
                                    layers, visible_layer_indices, 
                                    cluster_colors, layer_colors, analysis_type='bridge_score'):
    """
    Create a visualization of how clusters bridge between different layers in the network.
    
    This visualization builds a custom network by duplicating each node for each layer it's in,
    using the naming convention <layer>_<node>. This creates a network where the duplicated nodes
    connect interlayer and intralayer edges.
    
    For interlayer edges, all possible connections are created between duplicated nodes.
    For intralayer edges, only existing edges from the original network are added.
    
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
    nodes_per_layer : dict or int
        Dictionary mapping layer indices to lists of node IDs, or integer for number of nodes per layer
    layers : list
        List of layer names
    visible_layer_indices : list
        List of indices of visible layers
    cluster_colors : dict
        Dictionary mapping cluster IDs to colors
    layer_colors : dict
        Dictionary mapping layer indices to colors
    analysis_type : str
        Type of bridging analysis to perform: 'bridge_score', 'flow_efficiency', 'layer_span',
        'centrality_distribution', 'cluster_cohesion', or 'information_flow'
    """
    try:
        logging.info(f"Visible links: {len(visible_links)}, Node IDs: {len(node_ids)}, Clusters: {len(node_clusters)}")
        logging.info(f"Layers: {layers}")
        logging.info(f"Visible layer indices: {visible_layer_indices}")
        
        # Clear the axis and set title
        ax.clear()
        ax.set_title(f"Cluster Bridging Analysis: {analysis_type.replace('_', ' ').title()}")
        
        # Filter to show only visible layers
        logging.info(f"Filtering bridging analysis to show only {len(visible_layer_indices)} visible layers")
        
        # Build a multilayer network with duplicated nodes
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
            if layer_idx in visible_layer_indices:
                for node_id in node_list:
                    if node_id in node_ids:
                        node_layers[node_id].append(layer_idx)
        
        # Track nodes by layer and cluster for analysis
        nodes_by_layer = defaultdict(list)
        nodes_by_cluster = defaultdict(list)
        
        # Create duplicated nodes for each layer a node appears in
        duplicated_nodes = {}  # Maps original node ID to list of duplicated node IDs
        
        for node_id, layers_list in node_layers.items():
            duplicated_nodes[node_id] = []
            cluster = node_clusters.get(node_id)
            if cluster is None:
                continue  # Skip nodes without cluster assignment
                
            for layer_idx in layers_list:
                # Create a new node ID in the format <layer>_<node>
                new_node_id = f"{layer_idx}_{node_id}"
                duplicated_nodes[node_id].append(new_node_id)
                
                # Add the node to the graph with attributes
                G.add_node(new_node_id, 
                          original_id=node_id,
                          cluster=cluster, 
                          layer=layer_idx)
                          
                # Track nodes by layer and cluster for analysis
                nodes_by_layer[layer_idx].append(new_node_id)
                nodes_by_cluster[cluster].append(new_node_id)
        
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
        
        # Get unique clusters
        unique_clusters = sorted(set(node_clusters.values()))
        logging.info(f"Found {len(unique_clusters)} unique clusters across {len(visible_layer_indices)} layers")
        
        # Perform the selected analysis
        if analysis_type == 'bridge_score':
            _analyze_bridge_score(ax, G, unique_clusters, nodes_by_cluster, nodes_by_layer, 
                                  cluster_colors, layers, visible_layer_indices)
        elif analysis_type == 'flow_efficiency':
            _analyze_flow_efficiency(ax, G, unique_clusters, nodes_by_cluster, nodes_by_layer, 
                                    cluster_colors, layers, visible_layer_indices)
        elif analysis_type == 'layer_span':
            _analyze_layer_span(ax, G, unique_clusters, nodes_by_cluster, nodes_by_layer, 
                               cluster_colors, layers, visible_layer_indices, layer_colors)
        elif analysis_type == 'centrality_distribution':
            _analyze_centrality_distribution(ax, G, unique_clusters, nodes_by_cluster, nodes_by_layer,
                                           cluster_colors, layers, visible_layer_indices)
        elif analysis_type == 'cluster_cohesion':
            _analyze_cluster_cohesion(ax, G, unique_clusters, nodes_by_cluster, nodes_by_layer,
                                     cluster_colors, layers, visible_layer_indices)
        elif analysis_type == 'information_flow':
            _analyze_information_flow(ax, G, unique_clusters, nodes_by_cluster, nodes_by_layer,
                                     cluster_colors, layers, visible_layer_indices)
        else:
            ax.text(0.5, 0.5, f"Unknown analysis type: {analysis_type}", 
                   ha='center', va='center')
            
        logging.info(f"Successfully created cluster bridging analysis for {analysis_type}")
        
    except Exception as e:
        logging.error(f"Error creating cluster bridging analysis: {str(e)}")
        ax.clear()
        ax.text(0.5, 0.5, f"Error creating visualization: {str(e)}", 
               ha='center', va='center')


def _analyze_bridge_score(ax, G, unique_clusters, nodes_by_cluster, nodes_by_layer, 
                         cluster_colors, layers, visible_layer_indices):
    """
    Analyze and visualize bridge scores for clusters.
    Bridge score measures how well a cluster connects different layers.
    """
    # Calculate bridge scores for each cluster
    bridge_scores = {}
    for cluster in unique_clusters:
        cluster_nodes = nodes_by_cluster[cluster]
        if not cluster_nodes:
            bridge_scores[cluster] = 0
            continue
            
        # Count interlayer vs. intralayer connections
        interlayer_connections = 0
        total_connections = 0
        
        for node in cluster_nodes:
            if node not in G.nodes:
                continue
                
            # Extract layer from the duplicated node name (format: <layer>_<node>)
            node_layer = G.nodes[node]['layer']
            
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['cluster'] == cluster:  # Only count connections within the same cluster
                    neighbor_layer = G.nodes[neighbor]['layer']
                    total_connections += 1
                    if node_layer != neighbor_layer:
                        interlayer_connections += 1
        
        # Bridge score is the ratio of interlayer to total connections
        bridge_scores[cluster] = interlayer_connections / max(total_connections, 1)
    
    # Sort clusters by bridge score
    sorted_clusters = sorted(unique_clusters, key=lambda c: bridge_scores[c], reverse=True)
    
    # Create bar chart
    y_pos = np.arange(len(sorted_clusters))
    scores = [bridge_scores[c] for c in sorted_clusters]
    colors = [cluster_colors.get(c, '#CCCCCC') for c in sorted_clusters]
    
    bars = ax.barh(y_pos, scores, color=colors)
    
    # Add labels and formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Cluster {c}" for c in sorted_clusters])
    ax.set_xlabel("Bridge Score (higher = better bridging)")
    ax.set_xlim(0, max(scores) * 1.1 if scores else 0.1)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               f"{width:.2f}", ha='left', va='center')
    
    # Add explanation
    ax.text(0.5, -0.1, 
           "Bridge Score: Ratio of interlayer to total connections within a cluster.\nHigher scores indicate clusters that effectively bridge between layers.",
           ha='center', va='center', transform=ax.transAxes)


def _analyze_flow_efficiency(ax, G, unique_clusters, nodes_by_cluster, nodes_by_layer, 
                            cluster_colors, layers, visible_layer_indices):
    """
    Analyze and visualize flow efficiency for clusters between layers.
    Flow efficiency measures how efficiently information can flow through clusters between layers.
    """
    # Create a matrix to store flow efficiency between layers through each cluster
    num_layers = len(visible_layer_indices)
    flow_matrix = np.zeros((len(unique_clusters), num_layers, num_layers))
    
    # For each cluster, calculate average shortest path length between layers
    for c_idx, cluster in enumerate(unique_clusters):
        cluster_nodes = nodes_by_cluster[cluster]
        
        # Skip clusters with no nodes
        if not cluster_nodes:
            continue
            
        # Create a subgraph for this cluster
        subgraph = G.subgraph(cluster_nodes)
        
        # For each pair of layers, calculate average shortest path length
        for i, layer1 in enumerate(visible_layer_indices):
            # Get nodes in this cluster and layer (format: <layer>_<node>)
            nodes1 = [n for n in cluster_nodes if G.nodes[n]['layer'] == layer1]
            
            for j, layer2 in enumerate(visible_layer_indices):
                if i == j:  # Same layer, set to 1.0 (perfect efficiency)
                    flow_matrix[c_idx, i, j] = 1.0
                    continue
                    
                # Get nodes in this cluster and layer
                nodes2 = [n for n in cluster_nodes if G.nodes[n]['layer'] == layer2]
                
                # Skip if either layer has no nodes in this cluster
                if not nodes1 or not nodes2:
                    flow_matrix[c_idx, i, j] = 0
                    continue
                
                # Calculate average shortest path length
                path_lengths = []
                for n1 in nodes1:
                    for n2 in nodes2:
                        try:
                            path_length = nx.shortest_path_length(subgraph, n1, n2)
                            path_lengths.append(path_length)
                        except nx.NetworkXNoPath:
                            # No path exists
                            pass
                
                # Calculate flow efficiency as inverse of average path length
                if path_lengths:
                    avg_path_length = sum(path_lengths) / len(path_lengths)
                    flow_matrix[c_idx, i, j] = 1.0 / max(avg_path_length, 1.0)
                else:
                    flow_matrix[c_idx, i, j] = 0
    
    # Find the best cluster for each layer pair
    best_clusters = np.zeros((num_layers, num_layers), dtype=int)
    for i in range(num_layers):
        for j in range(num_layers):
            if i != j:
                best_clusters[i, j] = np.argmax(flow_matrix[:, i, j])
    
    # Create a heatmap of the best flow efficiency between layers
    best_flow = np.zeros((num_layers, num_layers))
    for i in range(num_layers):
        for j in range(num_layers):
            best_flow[i, j] = flow_matrix[best_clusters[i, j], i, j]
    
    # Create the heatmap
    im = ax.imshow(best_flow, cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Flow Efficiency (higher = better)')
    
    # Add labels
    layer_names = [layers[idx] for idx in visible_layer_indices]
    ax.set_xticks(np.arange(num_layers))
    ax.set_yticks(np.arange(num_layers))
    ax.set_xticklabels(layer_names)
    ax.set_yticklabels(layer_names)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add annotations showing the best cluster and efficiency value
    for i in range(num_layers):
        for j in range(num_layers):
            if i != j:  # Skip diagonal
                best_cluster = unique_clusters[best_clusters[i, j]]
                text = ax.text(j, i, f"C{best_cluster}\n{best_flow[i, j]:.2f}",
                              ha="center", va="center", color="white" if best_flow[i, j] > 0.5 else "black")
            else:
                ax.text(j, i, "—", ha="center", va="center")
    
    # Add explanation
    ax.text(0.5, -0.15, 
           "Flow Efficiency: Inverse of average shortest path length between layers through each cluster.\nHigher values indicate more efficient information flow. Each cell shows the best cluster for that layer pair.",
           ha='center', va='center', transform=ax.transAxes)


def _analyze_layer_span(ax, G, unique_clusters, nodes_by_cluster, nodes_by_layer, 
                       cluster_colors, layers, visible_layer_indices, layer_colors):
    """
    Analyze and visualize how clusters span across different layers.
    This shows the distribution of cluster nodes across layers.
    """
    logging.info("Analyzing layer span for clusters")
    
    # Calculate layer distribution for each cluster
    layer_distribution = {}
    for cluster in unique_clusters:
        cluster_nodes = nodes_by_cluster[cluster]
        if not cluster_nodes:
            continue
            
        # Count nodes by layer
        layer_counts = Counter()
        for node in cluster_nodes:
            if node in G.nodes:
                layer_idx = G.nodes[node]['layer']
                if layer_idx in visible_layer_indices:
                    layer_counts[layer_idx] += 1
        
        # Skip clusters with no nodes in visible layers
        if sum(layer_counts.values()) == 0:
            continue
            
        # Calculate percentage distribution
        total_nodes = sum(layer_counts.values())
        distribution = {layer: count / total_nodes for layer, count in layer_counts.items()}
        layer_distribution[cluster] = distribution
    
    # Skip if no clusters have layer distribution
    if not layer_distribution:
        ax.text(0.5, 0.5, "No clusters with nodes in visible layers", 
               ha='center', va='center')
        return
    
    # Sort clusters by number of layers they span
    cluster_span = {c: len(dist) for c, dist in layer_distribution.items()}
    sorted_clusters = sorted(layer_distribution.keys(), key=lambda c: cluster_span[c], reverse=True)
    
    # Create stacked bar chart
    x_pos = np.arange(len(sorted_clusters))
    bottoms = np.zeros(len(sorted_clusters))
    
    # Get layer names for legend
    layer_names = [layers[idx] for idx in visible_layer_indices]
    
    # Create bars for each layer
    bars = []
    for layer_idx in visible_layer_indices:
        layer_values = []
        for cluster in sorted_clusters:
            dist = layer_distribution[cluster]
            layer_values.append(dist.get(layer_idx, 0))
        
        layer_color = layer_colors.get(layer_idx, '#CCCCCC')
        bar = ax.bar(x_pos, layer_values, bottom=bottoms, color=layer_color, label=layers[layer_idx])
        bars.append(bar)
        bottoms += layer_values
    
    # Add labels and formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"C{c}" for c in sorted_clusters])
    ax.set_ylabel("Proportion of Cluster Nodes")
    ax.set_ylim(0, 1.0)
    
    # Add legend with smaller font size
    ax.legend(title="Layers", loc='upper right', fontsize='small')
    
    # Add span values above bars
    for i, cluster in enumerate(sorted_clusters):
        span = cluster_span[cluster]
        ax.text(i, 1.02, f"Span: {span}", ha='center', va='bottom', fontsize=9)
    
    # Add explanation
    ax.text(0.5, -0.15, 
           "Layer Span: Distribution of a cluster's nodes across different layers.\nHigher span values indicate clusters that bridge more layers. Each color represents a different layer.",
           ha='center', va='center', transform=ax.transAxes)


def _analyze_centrality_distribution(ax, G, unique_clusters, nodes_by_cluster, nodes_by_layer,
                                   cluster_colors, layers, visible_layer_indices):
    """
    Analyze and visualize the distribution of different centrality measures across clusters.
    This helps identify which clusters serve as central hubs in the network.
    """
    logging.info("Analyzing centrality distribution across clusters")
    
    # Calculate different centrality measures
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    
    # We'll use only betweenness centrality for simplicity and to avoid eigenvector issues
    # Group centrality measures by cluster
    cluster_centrality = defaultdict(lambda: {'betweenness': []})
    
    for node, bc in betweenness.items():
        if node in G.nodes:
            cluster = G.nodes[node]['cluster']
            cluster_centrality[cluster]['betweenness'].append(bc)
    
    # Calculate average betweenness centrality for each cluster
    avg_betweenness = {}
    for cluster in unique_clusters:
        if cluster in cluster_centrality and cluster_centrality[cluster]['betweenness']:
            avg_betweenness[cluster] = np.mean(cluster_centrality[cluster]['betweenness'])
        else:
            avg_betweenness[cluster] = 0
    
    # Sort clusters by average betweenness centrality
    sorted_clusters = sorted(unique_clusters, key=lambda c: avg_betweenness[c], reverse=True)
    
    # Create bar chart of average betweenness centrality
    y_pos = np.arange(len(sorted_clusters))
    values = [avg_betweenness[c] for c in sorted_clusters]
    colors = [cluster_colors.get(c, '#CCCCCC') for c in sorted_clusters]
    
    bars = ax.barh(y_pos, values, color=colors)
    
    # Add labels and formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Cluster {c}" for c in sorted_clusters])
    ax.set_xlabel("Average Betweenness Centrality")
    ax.set_title("Cluster Centrality Distribution")
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               f"{width:.3f}", ha='left', va='center')
    
    # Add explanation
    ax.text(0.5, -0.1, 
           "Centrality Distribution: Shows the average betweenness centrality for each cluster.\n"
           "Clusters with higher values are more important for information flow in the network.",
           ha='center', va='center', transform=ax.transAxes)


def _analyze_cluster_cohesion(ax, G, unique_clusters, nodes_by_cluster, nodes_by_layer,
                             cluster_colors, layers, visible_layer_indices):
    """
    Analyze and visualize the cohesion of clusters within and between layers.
    Cohesion measures how tightly connected nodes are within a cluster.
    """
    logging.info("Analyzing cluster cohesion within and between layers")
    
    # Calculate cohesion metrics for each cluster
    intra_cohesion = {}  # Within-layer cohesion
    inter_cohesion = {}  # Between-layer cohesion
    
    for cluster in unique_clusters:
        cluster_nodes = nodes_by_cluster[cluster]
        if not cluster_nodes:
            intra_cohesion[cluster] = 0
            inter_cohesion[cluster] = 0
            continue
        
        # Create a subgraph for this cluster
        subgraph = G.subgraph(cluster_nodes)
        
        # Calculate within-layer cohesion (average clustering coefficient within layers)
        layer_cohesion = {}
        for layer_idx in visible_layer_indices:
            # Get nodes in this layer and cluster
            layer_nodes = [n for n in cluster_nodes if G.nodes[n]['layer'] == layer_idx]
            
            if len(layer_nodes) < 3:  # Need at least 3 nodes for clustering coefficient
                continue
                
            # Create a subgraph for this layer
            layer_subgraph = subgraph.subgraph(layer_nodes)
            
            # Calculate average clustering coefficient
            try:
                clustering = nx.average_clustering(layer_subgraph)
                layer_cohesion[layer_idx] = clustering
            except:
                layer_cohesion[layer_idx] = 0
        
        # Average within-layer cohesion across all layers
        intra_cohesion[cluster] = np.mean(list(layer_cohesion.values())) if layer_cohesion else 0
        
        # Calculate between-layer cohesion (ratio of interlayer to total edges)
        interlayer_edges = 0
        total_edges = 0
        
        for u, v in subgraph.edges():
            total_edges += 1
            u_layer = G.nodes[u]['layer']
            v_layer = G.nodes[v]['layer']
            if u_layer != v_layer:
                interlayer_edges += 1
        
        inter_cohesion[cluster] = interlayer_edges / max(total_edges, 1)
    
    # Create a bar chart of combined cohesion score
    # Calculate combined score as weighted average of intra and inter cohesion
    combined_scores = {}
    for cluster in unique_clusters:
        intra = intra_cohesion.get(cluster, 0)
        inter = inter_cohesion.get(cluster, 0)
        # Weight inter-layer cohesion more heavily as it's more important for bridging
        combined_scores[cluster] = 0.4 * intra + 0.6 * inter
    
    # Sort clusters by combined score
    sorted_clusters = sorted(unique_clusters, key=lambda c: combined_scores[c], reverse=True)
    
    # Create bar chart
    y_pos = np.arange(len(sorted_clusters))
    values = [combined_scores[c] for c in sorted_clusters]
    colors = [cluster_colors.get(c, '#CCCCCC') for c in sorted_clusters]
    
    bars = ax.barh(y_pos, values, color=colors)
    
    # Add labels and formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Cluster {c}" for c in sorted_clusters])
    ax.set_xlabel("Cohesion Score")
    ax.set_title("Cluster Cohesion Analysis")
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        intra = intra_cohesion.get(sorted_clusters[i], 0)
        inter = inter_cohesion.get(sorted_clusters[i], 0)
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               f"{width:.2f} (I:{intra:.2f}, E:{inter:.2f})", ha='left', va='center', fontsize=8)
    
    # Add explanation
    ax.text(0.5, -0.1, 
           "Cluster Cohesion: Combined score of within-layer (I) and between-layer (E) cohesion.\n"
           "Higher scores indicate clusters that effectively bridge between layers while maintaining internal structure.",
           ha='center', va='center', transform=ax.transAxes)


def _analyze_information_flow(ax, G, unique_clusters, nodes_by_cluster, nodes_by_layer,
                             cluster_colors, layers, visible_layer_indices):
    """
    Simulate information flow through the network to identify key bridging clusters.
    This uses a diffusion model to see how information spreads across layers through clusters.
    """
    logging.info("Analyzing information flow through clusters")
    
    # Create a matrix to store information flow between layers through each cluster
    num_layers = len(visible_layer_indices)
    flow_matrix = np.zeros((len(unique_clusters), num_layers))
    
    # For each layer, simulate information flow starting from that layer
    for start_idx, start_layer in enumerate(visible_layer_indices):
        # For each cluster, measure how well it transmits information from the start layer
        for c_idx, cluster in enumerate(unique_clusters):
            cluster_nodes = nodes_by_cluster[cluster]
            
            # Skip clusters with no nodes
            if not cluster_nodes:
                continue
                
            # Get nodes in the start layer for this cluster
            start_nodes = [n for n in cluster_nodes if G.nodes[n]['layer'] == start_layer]
            
            if not start_nodes:
                continue
                
            # Create a subgraph for this cluster
            subgraph = G.subgraph(cluster_nodes)
            
            # Initialize information values (1.0 for start nodes, 0.0 for others)
            info_values = {node: 1.0 if node in start_nodes else 0.0 for node in cluster_nodes}
            
            # Simulate diffusion for a few steps
            num_steps = 3
            damping = 0.85
            
            for _ in range(num_steps):
                new_values = {}
                
                # For each node, update its value based on neighbors
                for node in cluster_nodes:
                    if node not in subgraph:
                        continue
                        
                    # Get neighbors
                    neighbors = list(subgraph.neighbors(node))
                    
                    if not neighbors:
                        new_values[node] = info_values[node]
                        continue
                    
                    # Calculate new value as weighted average of neighbors
                    neighbor_sum = sum(info_values[neigh] for neigh in neighbors)
                    new_value = (1 - damping) * info_values[node] + damping * neighbor_sum / len(neighbors)
                    new_values[node] = new_value
                
                # Update all values at once
                info_values = new_values
            
            # Measure information flow to each layer
            for end_idx, end_layer in enumerate(visible_layer_indices):
                if end_layer == start_layer:
                    continue
                    
                # Get nodes in the end layer for this cluster
                end_nodes = [n for n in cluster_nodes if G.nodes[n]['layer'] == end_layer]
                
                if not end_nodes:
                    continue
                
                # Calculate average information value at end nodes
                avg_info = sum(info_values[node] for node in end_nodes) / len(end_nodes)
                
                # Store in the flow matrix
                flow_matrix[c_idx, end_idx] += avg_info
    
    # Normalize the flow matrix by the number of source layers
    flow_matrix = flow_matrix / num_layers
    
    # Calculate overall flow score for each cluster (average across destination layers)
    flow_scores = np.mean(flow_matrix, axis=1)
    
    # Clear the original axis and create a new figure with a polar projection
    ax.clear()
    
    # Create a bar chart of flow scores instead of a radar chart
    # Sort clusters by flow score
    sorted_indices = np.argsort(flow_scores)[::-1]
    sorted_clusters = [unique_clusters[i] for i in sorted_indices]
    sorted_scores = [flow_scores[i] for i in sorted_indices]
    
    # Create bar chart
    y_pos = np.arange(len(sorted_clusters))
    colors = [cluster_colors.get(c, '#CCCCCC') for c in sorted_clusters]
    
    bars = ax.barh(y_pos, sorted_scores, color=colors)
    
    # Add labels and formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Cluster {c}" for c in sorted_clusters])
    ax.set_xlabel("Information Flow Score")
    ax.set_title("Information Flow Through Clusters")
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               f"{width:.2f}", ha='left', va='center')
    
    # Add explanation
    ax.text(0.5, -0.15, 
           "Information Flow: Shows how effectively clusters transmit information to different layers.\n"
           "Higher scores indicate clusters that better bridge information across the network.",
           ha='center', va='center', transform=ax.transAxes) 