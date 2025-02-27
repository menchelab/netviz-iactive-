import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.gridspec as gridspec

def identify_critical_layers(layer_connections, layers, metric="Connectivity Impact"):
    """Identify critical layers in the network"""
    n_layers = len(layers)
    criticality_scores = np.zeros(n_layers)
    impact_data = []
    
    # Calculate baseline network properties
    total_connections = np.sum(layer_connections)
    
    if metric == "Connectivity Impact":
        # Assess impact of removing each layer
        for i in range(n_layers):
            # Create modified connection matrix without layer i
            mod_connections = layer_connections.copy()
            mod_connections[i, :] = 0
            mod_connections[:, i] = 0
            
            # Calculate impact
            remaining_connections = np.sum(mod_connections)
            impact = (total_connections - remaining_connections) / total_connections if total_connections > 0 else 0
            criticality_scores[i] = impact
            
            # Store detailed impact data
            impact_data.append({
                'layer_index': i,
                'layer_name': layers[i],
                'connections_removed': total_connections - remaining_connections,
                'impact_score': impact
            })
            
    elif metric == "Information Flow":
        # Create network graph
        G = nx.Graph()
        for i in range(n_layers):
            G.add_node(i)
        for i in range(n_layers):
            for j in range(i+1, n_layers):
                if layer_connections[i,j] > 0:
                    G.add_edge(i, j, weight=layer_connections[i,j])
        
        # Calculate baseline centrality
        baseline_centrality = nx.betweenness_centrality(G, weight='weight')
        
        # Assess impact of removing each layer
        for i in range(n_layers):
            # Create modified graph without layer i
            H = G.copy()
            H.remove_node(i)
            
            # Calculate new centralities
            if H.number_of_nodes() > 1:
                new_centrality = nx.betweenness_centrality(H, weight='weight')
                
                # Calculate impact as average change in centrality
                impact = sum(abs(baseline_centrality.get(n, 0) - new_centrality.get(n, 0)) 
                           for n in H.nodes()) / H.number_of_nodes()
            else:
                impact = 1.0  # Maximum impact if removing node disconnects graph
                
            criticality_scores[i] = impact
            
            # Store detailed impact data
            impact_data.append({
                'layer_index': i,
                'layer_name': layers[i],
                'centrality_change': impact,
                'impact_score': impact
            })
    
    return criticality_scores, impact_data

def detect_anomalies(layer_connections, layers, threshold=1.0):
    """Detect anomalous connections between layers"""
    n_layers = len(layers)
    anomalies = []
    anomaly_scores = np.zeros((n_layers, n_layers))
    
    # Calculate mean and std of connection counts
    connection_counts = layer_connections[~np.eye(n_layers, dtype=bool)]
    mean_connections = np.mean(connection_counts)
    std_connections = np.std(connection_counts)
    
    # Detect anomalies using z-score
    if std_connections > 0:
        for i in range(n_layers):
            for j in range(n_layers):
                if i != j:
                    z_score = (layer_connections[i,j] - mean_connections) / std_connections
                    anomaly_scores[i,j] = abs(z_score)
                    
                    if abs(z_score) > threshold:
                        anomalies.append((i, j, layer_connections[i,j], z_score))
    
    return anomalies, anomaly_scores

def create_critical_structure_charts(criticality_bar_ax, impact_ax, anomaly_ax, network_ax, layer_connections, layers, medium_font=None, large_font=None, layer_colors=None):
    """Create visualizations for critical structure analysis including all four charts"""
    if medium_font is None:
        medium_font = {'fontsize': 8}
    if large_font is None:
        large_font = {'fontsize': 10}
    if layer_colors is None:
        layer_colors = {layer: 'skyblue' for layer in layers}

    # Calculate criticality scores and impact data
    criticality_scores, impact_data = identify_critical_layers(
        layer_connections, 
        layers,
        metric="Connectivity Impact"
    )

    # Create criticality bar chart (top left)
    sorted_indices = np.argsort(criticality_scores)[::-1]
    sorted_layers = [layers[i] for i in sorted_indices]
    sorted_scores = criticality_scores[sorted_indices]
    colors = [layer_colors.get(layers[i], 'skyblue') for i in sorted_indices]

    bars = criticality_bar_ax.barh(sorted_layers, sorted_scores, color=colors)
    criticality_bar_ax.set_title('Layer Criticality', **large_font)
    criticality_bar_ax.set_xlabel('Criticality Score', **medium_font)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        criticality_bar_ax.text(width * 1.01, bar.get_y() + bar.get_height()/2,
                              f'{width:.2f}', va='center', **medium_font)

    # Create impact visualization (top right)
    top_layers = 5
    x = np.arange(min(top_layers, len(sorted_indices)))
    width = 0.35

    before_values = [1.0] * len(x)  # Baseline connectivity
    after_values = [1.0 - criticality_scores[i] for i in sorted_indices[:len(x)]]
    
    impact_ax.bar(x - width/2, before_values, width, label='Before Removal', color='skyblue')
    impact_ax.bar(x + width/2, after_values, width, label='After Removal', color='salmon')
    
    impact_ax.set_ylabel('Network Connectivity', **medium_font)
    impact_ax.set_title('Impact of Layer Removal', **large_font)
    impact_ax.set_xticks(x)
    impact_ax.set_xticklabels([layers[i] for i in sorted_indices[:len(x)]], 
                             rotation=45, ha='right', **medium_font)
    impact_ax.legend(fontsize=8)

    # Detect anomalies
    anomalies, anomaly_scores = detect_anomalies(layer_connections, layers, threshold=1.0)

    # Create anomaly heatmap (bottom left)
    im = anomaly_ax.imshow(anomaly_scores, cmap='RdBu_r')
    anomaly_ax.set_title('Connection Anomalies', **large_font)
    
    # Add colorbar using the correct figure
    fig = anomaly_ax.get_figure()
    fig.colorbar(im, ax=anomaly_ax, label='Anomaly Score (Z-score)')
    
    # Add labels
    anomaly_ax.set_xticks(range(len(layers)))
    anomaly_ax.set_yticks(range(len(layers)))
    anomaly_ax.set_xticklabels(layers, rotation=90, **medium_font)
    anomaly_ax.set_yticklabels(layers, **medium_font)

    # Create network visualization (bottom right)
    G = nx.Graph()
    for i, layer in enumerate(layers):
        G.add_node(i, name=layer)
    
    # Add edges with weights based on connection counts
    for i in range(len(layers)):
        for j in range(i+1, len(layers)):
            if layer_connections[i, j] > 0:
                G.add_edge(i, j, weight=layer_connections[i, j])
    
    # Create layout
    pos = nx.spring_layout(G)
    
    # Draw nodes with layer colors
    node_colors = [layer_colors.get(layers[i], 'skyblue') for i in range(len(layers))]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.7, ax=network_ax)
    
    # Draw edges with width based on weight and color based on anomaly
    edge_colors = []
    edge_widths = []
    for (i, j) in G.edges():
        # Convert anomalies to tuples for comparison
        anomaly_pairs = [(a[0], a[1]) for a in anomalies]
        is_anomalous = (i, j) in anomaly_pairs or (j, i) in anomaly_pairs
        edge_colors.append('red' if is_anomalous else 'gray')
        edge_widths.append(3 if is_anomalous else 1)
    
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, ax=network_ax)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, {i: layers[i] for i in range(len(layers))}, 
                           font_size=8, ax=network_ax)
    
    network_ax.set_title('Layer Network Structure', **large_font)
    network_ax.axis('off') 