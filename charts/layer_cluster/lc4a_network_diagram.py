import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba
import sys
import os

# Add the project root to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.calc_community import detect_communities, get_node_community_metrics, AVAILABLE_COMMUNITY_ALGORITHMS

# Import Qt components for UI
from PyQt5.QtWidgets import (
    QGroupBox, QGridLayout, QLabel, QComboBox, 
    QDoubleSpinBox, QCheckBox, QPushButton
)

def create_enhanced_network_ui():
    """
    Create UI components for the enhanced network diagram.
    
    Returns:
    --------
    dict
        Dictionary containing all UI components
    """
    # Create group box for enhanced network options
    enhanced_network_group = QGroupBox("Enhanced Network Diagram Options")
    enhanced_network_layout = QGridLayout()
    
    # Edge counting method
    enhanced_network_layout.addWidget(QLabel("Edge Counting:"), 0, 0)
    edge_counting_combo = QComboBox()
    edge_counting_combo.addItems(["all", "unique", "weighted"])
    edge_counting_combo.setToolTip("Method to count edges between nodes:\n"
                                   "- all: Count all edges (default)\n"
                                   "- unique: Count unique connections (binary)\n"
                                   "- weighted: Use edge weights for counting")
    enhanced_network_layout.addWidget(edge_counting_combo, 0, 1)
    
    # Community detection algorithm
    enhanced_network_layout.addWidget(QLabel("Community Detection:"), 1, 0)
    community_algorithm_combo = QComboBox()
    community_algorithm_combo.addItem("None")
    community_algorithm_combo.addItems(AVAILABLE_COMMUNITY_ALGORITHMS)
    community_algorithm_combo.setToolTip("Community detection algorithm to use")
    enhanced_network_layout.addWidget(community_algorithm_combo, 1, 1)
    
    # Resolution parameter for community detection
    enhanced_network_layout.addWidget(QLabel("Resolution:"), 2, 0)
    community_resolution_spin = QDoubleSpinBox()
    community_resolution_spin.setRange(0.1, 5.0)
    community_resolution_spin.setSingleStep(0.1)
    community_resolution_spin.setValue(1.0)
    community_resolution_spin.setToolTip("Resolution parameter for community detection algorithms")
    enhanced_network_layout.addWidget(community_resolution_spin, 2, 1)
    
    # Show edge weights
    show_edge_weights_check = QCheckBox("Show Edge Weights")
    show_edge_weights_check.setChecked(True)
    enhanced_network_layout.addWidget(show_edge_weights_check, 3, 0)
    
    # Show node labels
    show_node_labels_check = QCheckBox("Show Node Labels")
    show_node_labels_check.setChecked(True)
    enhanced_network_layout.addWidget(show_node_labels_check, 3, 1)
    
    # Show legend
    show_legend_check = QCheckBox("Show Legend")
    show_legend_check.setChecked(True)
    enhanced_network_layout.addWidget(show_legend_check, 4, 0, 1, 2)
    
    # Apply button for enhanced network
    update_enhanced_network_btn = QPushButton("Update Enhanced Network")
    enhanced_network_layout.addWidget(update_enhanced_network_btn, 5, 0, 1, 2)
    
    enhanced_network_group.setLayout(enhanced_network_layout)
    
    # Return all UI components in a dictionary
    return {
        "group": enhanced_network_group,
        "edge_counting_combo": edge_counting_combo,
        "community_algorithm_combo": community_algorithm_combo,
        "community_resolution_spin": community_resolution_spin,
        "show_edge_weights_check": show_edge_weights_check,
        "show_node_labels_check": show_node_labels_check,
        "show_legend_check": show_legend_check,
        "update_btn": update_enhanced_network_btn
    }

def update_enhanced_network(figure, ax, canvas, data_manager, ui_components, layout_algorithm="community", aspect_ratio=1.0):
    """
    Update the enhanced network diagram with current settings.
    
    Parameters:
    -----------
    figure : matplotlib.figure.Figure
        The figure to draw on
    ax : matplotlib.axes.Axes
        The axis to draw on
    canvas : FigureCanvas
        The canvas to update
    data_manager : DataManager
        The data manager containing the data to visualize
    ui_components : dict
        Dictionary of UI components from create_enhanced_network_ui()
    layout_algorithm : str
        The layout algorithm to use
    aspect_ratio : float
        The aspect ratio for the layout
    """
    if not figure or not data_manager:
        return
        
    # Clear the figure and recreate the axis
    figure.clear()
    new_ax = figure.add_subplot(111)
    
    # Get the selected options from UI components
    edge_counting = ui_components["edge_counting_combo"].currentText()
    
    community_algorithm = ui_components["community_algorithm_combo"].currentText()
    if community_algorithm == "None":
        community_algorithm = None
        
    community_resolution = ui_components["community_resolution_spin"].value()
    show_edge_weights = ui_components["show_edge_weights_check"].isChecked()
    show_node_labels = ui_components["show_node_labels_check"].isChecked()
    show_legend = ui_components["show_legend_check"].isChecked()
    
    # Get visible links
    visible_links = []
    if data_manager.current_edge_mask is not None:
        visible_links = [data_manager.link_pairs[i] for i in range(len(data_manager.link_pairs)) 
                        if data_manager.current_edge_mask[i]]
    
    # Get visible layer indices
    visible_layer_indices = []
    if hasattr(data_manager, 'visible_layer_indices'):
        visible_layer_indices = data_manager.visible_layer_indices
    else:
        visible_layer_indices = list(range(len(data_manager.layers)))
    
    # Create the enhanced network diagram
    create_enhanced_layer_cluster_network_diagram(
        new_ax,
        visible_links,
        data_manager.node_ids,
        data_manager.node_clusters,
        data_manager.nodes_per_layer,
        data_manager.layers,
        visible_layer_indices,
        data_manager.cluster_colors,
        layout_algorithm=layout_algorithm,
        aspect_ratio=aspect_ratio,
        edge_counting=edge_counting,
        community_algorithm=community_algorithm,
        community_resolution=community_resolution,
        show_edge_weights=show_edge_weights,
        show_node_labels=show_node_labels,
        show_legend=show_legend
    )
    
    # Draw the canvas
    canvas.draw()
    
    return new_ax

def create_enhanced_layer_cluster_network_diagram(
    ax,
    visible_links,
    node_ids,
    node_clusters,
    nodes_per_layer,
    layers,
    visible_layer_indices=None,
    cluster_colors=None,
    layout_algorithm="community",  # Layout algorithm
    aspect_ratio=1.0,  # Aspect ratio control
    edge_counting="all",  # Edge counting method: "all", "unique", "weighted"
    community_algorithm=None,  # Community detection algorithm (None = no communities)
    community_resolution=1.0,  # Resolution parameter for community detection
    show_edge_weights=True,  # Whether to show edge weights
    show_node_labels=True,  # Whether to show node labels
    show_legend=True,  # Whether to show the legend
):
    """
    Create an enhanced network diagram showing connections between layers and clusters
    with additional features for community detection and edge counting options.
    
    Parameters:
    -----------
    edge_counting : str
        Method to count edges between nodes:
        - "all": Count all edges (default)
        - "unique": Count unique connections (binary)
        - "weighted": Use edge weights for counting
    community_algorithm : str or None
        Community detection algorithm to use (None = no communities)
        Options: "louvain", "leiden", "girvan_newman", "label_propagation", etc.
    community_resolution : float
        Resolution parameter for community detection algorithms
    show_edge_weights : bool
        Whether to show edge weights on the diagram
    show_node_labels : bool
        Whether to show node labels
    show_legend : bool
        Whether to show the legend
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating enhanced layer-cluster network diagram with layout={layout_algorithm}, edge_counting={edge_counting}")
    
    # Clear the axis and set title
    ax.clear()
    
    # Set title based on configuration
    title = f"Layer-Cluster Network ({layout_algorithm.title()} Layout)"
    if community_algorithm:
        title += f" with {community_algorithm.title()} Communities"
    ax.set_title(title)
    
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
    
    # Count nodes by cluster and layer based on the selected edge counting method
    cluster_layer_counts = defaultdict(lambda: defaultdict(float))
    
    # Track unique connections for the "unique" counting method
    if edge_counting == "unique":
        unique_connections = set()
    
    # Process nodes and count based on the selected method
    for node_idx in visible_node_indices:
        # Calculate layer index
        layer_idx = node_idx // nodes_per_layer
        
        # Skip if layer is not visible
        if layer_idx not in visible_layers:
            continue
            
        node_id = node_ids[node_idx]
        cluster = node_clusters.get(node_id, "Unknown")
        
        if edge_counting == "unique":
            # Only count each unique node once
            connection_key = (cluster, layer_idx)
            if connection_key not in unique_connections:
                unique_connections.add(connection_key)
                cluster_layer_counts[cluster][layer_idx] += 1
        elif edge_counting == "weighted":
            # Count with weights (will be adjusted later with edge weights)
            # For now, just count the node
            cluster_layer_counts[cluster][layer_idx] += 1
        else:  # "all" (default)
            # Count all nodes
            cluster_layer_counts[cluster][layer_idx] += 1
    
    # If using weighted counting, adjust counts based on edge weights
    if edge_counting == "weighted":
        # Create a mapping of node_idx to (cluster, layer)
        node_to_cluster_layer = {}
        for node_idx in visible_node_indices:
            layer_idx = node_idx // nodes_per_layer
            if layer_idx in visible_layers:
                node_id = node_ids[node_idx]
                cluster = node_clusters.get(node_id, "Unknown")
                node_to_cluster_layer[node_idx] = (cluster, layer_idx)
        
        # Process edges and add weights
        edge_weights = defaultdict(float)
        for start_idx, end_idx in visible_links:
            if start_idx in node_to_cluster_layer and end_idx in node_to_cluster_layer:
                start_cluster, start_layer = node_to_cluster_layer[start_idx]
                end_cluster, end_layer = node_to_cluster_layer[end_idx]
                
                # Add weight to the connection (both directions)
                edge_weights[(start_cluster, start_layer, end_cluster, end_layer)] += 1
                edge_weights[(end_cluster, end_layer, start_cluster, start_layer)] += 1
        
        # Reset counts and use edge weights
        cluster_layer_counts = defaultdict(lambda: defaultdict(float))
        for (start_cluster, start_layer, end_cluster, end_layer), weight in edge_weights.items():
            # Only count connections between different cluster-layer pairs
            if start_cluster != end_cluster or start_layer != end_layer:
                cluster_layer_counts[start_cluster][start_layer] += weight / 2
                cluster_layer_counts[end_cluster][end_layer] += weight / 2
    
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
    
    # Apply community detection if requested
    node_communities = {}
    community_colors = {}
    
    if community_algorithm:
        try:
            # Detect communities
            communities = detect_communities(G, algorithm=community_algorithm, resolution=community_resolution)
            
            # Get community metrics for nodes
            node_metrics = get_node_community_metrics(G, communities)
            
            # Store community assignments
            node_communities = communities
            
            # Create colors for communities
            unique_communities = sorted(set(communities.values()))
            community_cmap = plt.cm.nipy_spectral
            community_colors = {comm: community_cmap(i / max(1, len(unique_communities) - 1)) 
                               for i, comm in enumerate(unique_communities)}
            
            logger.info(f"Detected {len(unique_communities)} communities using {community_algorithm}")
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            community_algorithm = None
    
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
        pos = _create_community_layout(G, layer_nodes, cluster_nodes, aspect_ratio, node_communities)
    
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
        
        # Add edge weight labels if requested
        if show_edge_weights:
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
                ax.text(edge_x, edge_y, f"{weight:.1f}\n({percentage:.1f}%)", 
                        fontsize=8, 
                        ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    # Draw nodes with community colors if community detection was used
    if community_algorithm and node_communities:
        # Draw nodes with community colors
        for node in G.nodes():
            comm = node_communities.get(node, -1)
            color = community_colors.get(comm, "gray")
            
            # Adjust color based on node type (layer vs cluster)
            if node.startswith("L_"):
                # For layer nodes, use a lighter version of the community color
                base_color = to_rgba(color)
                # Mix with light blue
                mixed_color = [0.7 * base_color[0] + 0.3 * 0.7, 
                              0.7 * base_color[1] + 0.3 * 0.9, 
                              0.7 * base_color[2] + 0.3 * 1.0, 
                              1.0]
                node_color = mixed_color
                size = layer_sizes.get(node, 500)
            else:  # Cluster node
                # For cluster nodes, use the cluster color but adjust saturation based on community
                base_color = to_rgba(cluster_colors.get(node[2:], "gray"))
                comm_color = to_rgba(color)
                # Mix cluster color with community color
                mixed_color = [0.7 * base_color[0] + 0.3 * comm_color[0], 
                              0.7 * base_color[1] + 0.3 * comm_color[1], 
                              0.7 * base_color[2] + 0.3 * comm_color[2], 
                              1.0]
                node_color = mixed_color
                size = cluster_sizes.get(node, 500)
            
            # Draw the node
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=[node],
                node_color=[node_color],
                node_size=size,
                alpha=0.8,
                edgecolors='black',
                ax=ax
            )
    else:
        # Standard drawing without communities
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
    
    # Add node labels if requested
    if show_node_labels:
        layer_labels = {}
        for layer in unique_layers:
            layer_count = sum(cluster_layer_counts[cluster][layers.index(layer)] 
                            for cluster in unique_clusters 
                            if layers.index(layer) in cluster_layer_counts[cluster])
            layer_labels[f"L_{layer}"] = f"{layer}\n({layer_count:.1f})"
        
        nx.draw_networkx_labels(
            G, pos, 
            labels=layer_labels,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
        
        cluster_labels = {}
        for cluster in unique_clusters:
            cluster_count = sum(cluster_layer_counts[cluster].values())
            cluster_labels[f"C_{cluster}"] = f"{cluster}\n({cluster_count:.1f})"
        
        nx.draw_networkx_labels(
            G, pos, 
            labels=cluster_labels,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
    
    # Create legends if requested
    if show_legend:
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
        
        # Add the cluster legend
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=8,
            title="Clusters",
            title_fontsize=9
        )
        
        # Add a community legend if community detection was used
        if community_algorithm and node_communities:
            # Create a separate legend for communities
            community_legend_elements = []
            for comm in sorted(set(node_communities.values())):
                community_legend_elements.append(
                    plt.Line2D([0], [0], 
                            marker='s', 
                            color='w',
                            markerfacecolor=community_colors.get(comm, "gray"),
                            markersize=10,
                            label=f"Comm {comm}")
                )
            
            # Add the community legend in a different location
            community_legend = ax.legend(
                handles=community_legend_elements,
                loc='lower right',
                fontsize=8,
                title="Communities",
                title_fontsize=9
            )
            
            # Add the first legend back
            ax.add_artist(community_legend)
        
        # Add a colorbar for edge weights
        sm = ScalarMappable(cmap=edge_cmap, norm=edge_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Connection Strength', fontsize=8)
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add padding around the plot
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Apply tight layout with padding
    plt.tight_layout(pad=1.5)
    
    logger.info(f"Successfully created enhanced layer-cluster network with {len(unique_layers)} layers and {len(unique_clusters)} clusters") 
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

def _create_community_layout(G, layer_nodes, cluster_nodes, aspect_ratio=1.0, node_communities=None):
    """Create a layout that emphasizes community structure"""
    # First, detect communities if not provided
    if node_communities is None:
        try:
            # Try to use community detection algorithms if available
            import community as community_louvain
            partition = community_louvain.best_partition(G, weight='weight')
        except ImportError:
            # Fallback to a simple partitioning based on node type
            partition = {node: 0 if node.startswith('L_') else 1 for node in G.nodes()}
    else:
        partition = node_communities
    
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