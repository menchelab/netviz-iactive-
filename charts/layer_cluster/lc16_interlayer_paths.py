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
from PyQt5.QtWidgets import (QComboBox, QCheckBox, QLabel, QHBoxLayout, 
                            QVBoxLayout, QGroupBox, QWidget, QGridLayout)
from PyQt5.QtCore import Qt

def create_lc16_ui_elements(parent=None):
    """
    Create UI elements for the LC16 interlayer path analysis visualization.
    
    Parameters:
    -----------
    parent : QWidget, optional
        The parent widget for the UI elements
        
    Returns:
    --------
    dict
        A dictionary containing the UI elements and layouts
    """
    # Create a group box to contain all the UI elements
    group_box = QGroupBox("LC16 Visualization Options")
    
    # Create a grid layout for the UI elements
    grid_layout = QGridLayout()
    
    # Create a dropdown for visualization style
    viz_style_label = QLabel("LC16 Visualization:")
    viz_style_combo = QComboBox()
    viz_style_combo.addItems(["Standard", "Simplified", "Detailed", "Classic Circle", "Layer-Focused"])
    viz_style_combo.setToolTip("Select a visualization style for the bottleneck analysis")
    
    # Create checkboxes for various options
    show_labels_checkbox = QCheckBox("Labels")
    show_labels_checkbox.setChecked(True)
    show_labels_checkbox.setToolTip("Show node and edge labels")
    
    show_nodes_checkbox = QCheckBox("Nodes")
    show_nodes_checkbox.setChecked(True)
    show_nodes_checkbox.setToolTip("Show nodes in the visualization")
    
    color_by_centrality_checkbox = QCheckBox("Color by Centrality")
    color_by_centrality_checkbox.setChecked(False)
    color_by_centrality_checkbox.setToolTip("Color edges by betweenness centrality instead of edge type")
    
    # Add new UI elements for improved visualization
    hide_unconnected_checkbox = QCheckBox("Hide Unconnected Nodes")
    hide_unconnected_checkbox.setChecked(False)
    hide_unconnected_checkbox.setToolTip("Hide nodes with no significant connections to improve layout")
    
    emphasize_layers_checkbox = QCheckBox("Emphasize Layer Labels")
    emphasize_layers_checkbox.setChecked(True)
    emphasize_layers_checkbox.setToolTip("Add prominent layer labels to the visualization")
    
    # Create a slider or dropdown for node size scaling
    node_size_label = QLabel("Node Size:")
    node_size_combo = QComboBox()
    node_size_combo.addItems(["Small", "Medium", "Large", "Dynamic"])
    node_size_combo.setCurrentText("Medium")
    node_size_combo.setToolTip("Adjust the size of nodes in the visualization")
    
    # Create a dropdown for layout spacing
    layout_spacing_label = QLabel("Layout Spacing:")
    layout_spacing_combo = QComboBox()
    layout_spacing_combo.addItems(["Compact", "Standard", "Expanded"])
    layout_spacing_combo.setCurrentText("Standard")
    layout_spacing_combo.setToolTip("Adjust the spacing between nodes in the layout")
    
    # Add the UI elements to the grid layout
    grid_layout.addWidget(viz_style_label, 0, 0)
    grid_layout.addWidget(viz_style_combo, 0, 1)
    
    grid_layout.addWidget(show_labels_checkbox, 1, 0)
    grid_layout.addWidget(show_nodes_checkbox, 1, 1)
    
    grid_layout.addWidget(color_by_centrality_checkbox, 2, 0)
    grid_layout.addWidget(hide_unconnected_checkbox, 2, 1)
    
    grid_layout.addWidget(emphasize_layers_checkbox, 3, 0)
    
    grid_layout.addWidget(node_size_label, 4, 0)
    grid_layout.addWidget(node_size_combo, 4, 1)
    
    grid_layout.addWidget(layout_spacing_label, 5, 0)
    grid_layout.addWidget(layout_spacing_combo, 5, 1)
    
    # Set the layout for the group box
    group_box.setLayout(grid_layout)
    
    # Create a dictionary to store the UI elements
    ui_elements = {
        "group": group_box,
        "viz_style_combo": viz_style_combo,
        "show_labels_checkbox": show_labels_checkbox,
        "show_nodes_checkbox": show_nodes_checkbox,
        "color_by_centrality_checkbox": color_by_centrality_checkbox,
        "hide_unconnected_checkbox": hide_unconnected_checkbox,
        "emphasize_layers_checkbox": emphasize_layers_checkbox,
        "node_size_combo": node_size_combo,
        "layout_spacing_combo": layout_spacing_combo
    }
    
    return ui_elements

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
    small_fontsize=9,
    selected_cluster=None,  # New parameter to filter by cluster
    viz_style="Standard",  # Visualization style: "Standard", "Simplified", "Detailed", "Classic Circle"
    show_labels=True,      # Whether to show node and edge labels
    show_nodes=True,       # Whether to show nodes
    color_by_centrality=False,  # Whether to color edges by centrality instead of type
    hide_unconnected=False,     # Whether to hide nodes with no significant connections
    emphasize_layers=True,      # Whether to emphasize layer labels
    node_size="Medium",         # Node size: "Small", "Medium", "Large", "Dynamic"
    layout_spacing="Standard"   # Layout spacing: "Compact", "Standard", "Expanded"
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
    selected_cluster : int, optional
        If specified, only include nodes from this cluster in the analysis
    viz_style : str, optional
        Visualization style to use: "Standard", "Simplified", "Detailed", or "Classic Circle"
    show_labels : bool, optional
        Whether to show node and edge labels
    show_nodes : bool, optional
        Whether to show nodes
    color_by_centrality : bool, optional
        Whether to color edges by centrality instead of type
    hide_unconnected : bool, optional
        Whether to hide nodes with no significant connections
    emphasize_layers : bool, optional
        Whether to emphasize layer labels
    node_size : str, optional
        Node size: "Small", "Medium", "Large", "Dynamic"
    layout_spacing : str, optional
        Layout spacing: "Compact", "Standard", "Expanded"
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
                        # If a cluster is selected, only include nodes from that cluster
                        if selected_cluster is not None:
                            node_cluster = node_clusters.get(node_id, None)
                            if node_cluster != selected_cluster:
                                continue
                        node_layers[node_id].append(layer_idx)
        
        # Create duplicated nodes for each layer a node appears in
        duplicated_nodes = {}
        
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
            
            # If a cluster is selected, only include edges where both nodes are in the selected cluster
            if selected_cluster is not None:
                source_cluster = node_clusters.get(source_id, None)
                target_cluster = node_clusters.get(target_id, None)
                if source_cluster != selected_cluster or target_cluster != selected_cluster:
                    continue
                    
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
            
            # If a cluster is selected, only include edges where both nodes are in the selected cluster
            if selected_cluster is not None:
                source_cluster = node_clusters.get(source_id, None)
                target_cluster = node_clusters.get(target_id, None)
                if source_cluster != selected_cluster or target_cluster != selected_cluster:
                    continue
            
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
            _analyze_bottlenecks(ax, G, visible_layer_indices, layers, node_clusters, cluster_colors,
                                viz_style=viz_style, show_labels=show_labels, show_nodes=show_nodes,
                                color_by_centrality=color_by_centrality, hide_unconnected=hide_unconnected,
                                emphasize_layers=emphasize_layers, node_size=node_size, 
                                layout_spacing=layout_spacing)
        else:
            ax.text(0.5, 0.5, f"Unknown analysis type: {analysis_type}", 
                   ha='center', va='center')
            
        # Set title based on analysis type
        title_map = {
            "path_length": "Interlayer Path Analysis (Duplicated Nodes)",
            "betweenness": "Node Betweenness in Duplicated Network",
            "bottleneck": "Critical Connections in Duplicated Network"
        }
        title = title_map.get(analysis_type, f"Duplicated Network Analysis: {analysis_type}")
        
        # Add cluster information to the title if a cluster is selected
        if selected_cluster is not None:
            title += f" - Cluster {selected_cluster}"
            
        ax.set_title(title)
        
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

def _analyze_bottlenecks(ax, G, visible_layer_indices, layers, node_clusters, cluster_colors,
                         viz_style="Standard", show_labels=True, show_nodes=True, color_by_centrality=False,
                         hide_unconnected=False, emphasize_layers=True, node_size="Medium", layout_spacing="Standard"):
    """Analyze and visualize bottleneck connections in the duplicated node network"""
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing bottleneck connections in the duplicated node network with style: {viz_style}")
    
    # Calculate edge betweenness centrality to identify bottlenecks
    edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight')
    
    # Normalize edge betweenness by the maximum value
    max_betweenness = max(edge_betweenness.values()) if edge_betweenness else 1.0
    normalized_betweenness = {edge: bc / max_betweenness for edge, bc in edge_betweenness.items()}
    
    # Calculate node betweenness for filtering and sizing
    node_betweenness = nx.betweenness_centrality(G, weight='weight')
    max_node_betweenness = max(node_betweenness.values()) if node_betweenness else 1.0
    
    # Filter out unconnected nodes if requested
    if hide_unconnected:
        # Define a threshold for "significant" connections
        betweenness_threshold = 0.05  # Nodes with betweenness < 5% of max are considered unconnected
        
        # Create a subgraph with only the connected nodes
        connected_nodes = [node for node, bc in node_betweenness.items() 
                          if bc > max_node_betweenness * betweenness_threshold]
        G = G.subgraph(connected_nodes).copy()
        
        logger.info(f"Filtered graph to {len(G.nodes)} connected nodes from original {len(node_betweenness)} nodes")
    
    # Group nodes by layer and cluster for better layout
    layer_nodes = defaultdict(list)
    cluster_nodes = defaultdict(list)
    for node in G.nodes:
        layer = G.nodes[node]['layer']
        cluster = G.nodes[node]['cluster']
        layer_nodes[layer].append(node)
        cluster_nodes[cluster].append(node)
    
    # Create a layout based on the selected visualization style
    if viz_style == "Layer-Focused":
        # Create a layout that emphasizes layers
        pos = _create_layer_focused_layout(G, layer_nodes)
    elif viz_style == "Classic Circle":
        # Use a circular layout
        pos = nx.circular_layout(G, scale=0.9)
    elif viz_style == "Simplified":
        # Use a simpler spring layout with fewer iterations
        pos = nx.spring_layout(G, k=0.3, iterations=30, seed=42)
    elif viz_style == "Detailed":
        # Use a more detailed layout with community detection
        try:
            # Try to use community detection for better grouping
            from community import best_partition
            partition = best_partition(G)
            # Add community information to nodes
            for node, community in partition.items():
                G.nodes[node]['community'] = community
            
            # Position nodes using the community structure
            pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42)
            
            # Adjust positions to group by community
            community_centers = defaultdict(lambda: [0, 0])
            community_counts = Counter()
            
            for node, position in pos.items():
                community = G.nodes[node].get('community', 0)
                community_centers[community][0] += position[0]
                community_centers[community][1] += position[1]
                community_counts[community] += 1
            
            # Calculate average position for each community
            for community in community_centers:
                if community_counts[community] > 0:
                    community_centers[community][0] /= community_counts[community]
                    community_centers[community][1] /= community_counts[community]
            
            # Adjust node positions to be closer to their community center
            for node in pos:
                community = G.nodes[node].get('community', 0)
                pos[node][0] = 0.8 * pos[node][0] + 0.2 * community_centers[community][0]
                pos[node][1] = 0.8 * pos[node][1] + 0.2 * community_centers[community][1]
        except ImportError:
            # Fall back to standard layout if community detection is not available
            pos = nx.kamada_kawai_layout(G)
    else:  # Standard
        # Use the Kamada-Kawai layout which tends to show structure well
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            # Fall back to spring layout if kamada_kawai fails
            pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Apply layout spacing adjustment
    spacing_factor = 1.0  # Default for "Standard" spacing
    if layout_spacing == "Compact":
        spacing_factor = 0.8
    elif layout_spacing == "Expanded":
        spacing_factor = 1.5
    
    # Scale the layout to fill the available space
    pos_array = np.array(list(pos.values()))
    if len(pos_array) > 0:
        # Find the current min/max coordinates
        min_x, min_y = pos_array.min(axis=0)
        max_x, max_y = pos_array.max(axis=0)
        
        # Scale to ensure we use most of the available space
        scale_factor = 0.8 * spacing_factor  # Adjust scale based on spacing preference
        for node in pos:
            pos[node] = [
                (pos[node][0] - min_x) / (max_x - min_x + 1e-10) * scale_factor * 2 - scale_factor,
                (pos[node][1] - min_y) / (max_y - min_y + 1e-10) * scale_factor * 2 - scale_factor
            ]
    
    # Clear the axis for a clean visualization
    ax.clear()
    
    # Draw the graph
    # 1. First draw all edges with minimal styling to show the structure
    for u, v in G.edges():
        edge_type = G.edges[u, v].get('edge_type', 'unknown')
        # Use very light colors for the background structure
        color = 'lightblue' if edge_type == "interlayer" else 'mistyrose'
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
               linewidth=0.5, color=color, alpha=0.3,
               zorder=0)  # Lowest zorder to draw in the background
    
    # 2. Draw the significant edges with width and color based on betweenness
    for edge, betweenness in normalized_betweenness.items():
        if edge[0] not in G.nodes or edge[1] not in G.nodes:
            continue  # Skip edges that involve filtered nodes
            
        if betweenness < 0.1 and viz_style != "Detailed":  # Skip low betweenness edges for clarity, except in detailed mode
            continue
            
        u, v = edge
        edge_type = G.edges.get(edge, {}).get('edge_type', G.edges.get((v, u), {}).get('edge_type', 'unknown'))
        
        # Calculate edge width based on betweenness
        width = 0.5 + 4 * betweenness
        
        # Calculate edge color based on configuration
        if color_by_centrality:
            # Color by betweenness centrality (red = high, blue = low)
            color = plt.cm.RdYlBu_r(betweenness)
        else:
            # Color by edge type (blue for interlayer, red for intralayer)
            if edge_type == "interlayer":
                color = plt.cm.Blues(0.5 + 0.5 * betweenness)  # Blue for interlayer
            else:
                color = plt.cm.Reds(0.5 + 0.5 * betweenness)   # Red for intralayer
        
        # Draw the edge
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
               linewidth=width, color=color, alpha=0.8,
               zorder=1)  # Higher zorder to draw on top of background edges
        
        # Add edge label only for the most significant bottlenecks
        if show_labels and betweenness > 0.7:
            # Position the label at the middle of the edge
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            
            # Add a small offset to avoid overlapping with the edge
            dx = pos[v][0] - pos[u][0]
            dy = pos[v][1] - pos[u][1]
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 0:
                offset_x = -dy / norm * 0.05
                offset_y = dx / norm * 0.05
            else:
                offset_x = offset_y = 0
            
            # Add the label with simplified text
            edge_label = "Inter" if edge_type == "interlayer" else "Intra"
            ax.text(x + offset_x, y + offset_y, 
                   f"{edge_label}\nBC={betweenness:.2f}", 
                   fontsize=8,
                   ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=1, boxstyle='round'),
                   zorder=3)
    
    if show_nodes:
        # Determine node size scaling based on network size, visualization style, and user preference
        if node_size == "Small":
            size_factor = 0.6
        elif node_size == "Large":
            size_factor = 1.5
        elif node_size == "Dynamic":
            # Dynamic sizing will be handled differently
            size_factor = 1.0
        else:  # Medium (default)
            size_factor = 1.0
            
        if viz_style == "Simplified":
            base_size = max(30, 300 / np.sqrt(len(G))) * size_factor
        elif viz_style == "Detailed":
            base_size = max(70, 600 / np.sqrt(len(G))) * size_factor
        else:  # Standard or Classic Circle
            base_size = max(50, 500 / np.sqrt(len(G))) * size_factor
        
        # First draw all nodes with minimal styling to show the structure
        for node in G.nodes:
            cluster = G.nodes[node]['cluster']
            
            # Get node color based on cluster
            if cluster_colors and cluster in cluster_colors:
                color = cluster_colors[cluster]
            else:
                color = plt.cm.tab10(cluster % 10)
                
            # Draw all nodes with a small size for context
            ax.scatter(pos[node][0], pos[node][1], 
                      s=base_size * 0.5, color=color, alpha=0.3, edgecolor='none',
                      zorder=1.5)  # Draw between edges and significant nodes
        
        # Then draw significant nodes with more emphasis
        for node in G.nodes:
            betweenness = node_betweenness.get(node, 0)
            if betweenness < 0.1 and viz_style != "Detailed" and node_size != "Dynamic":
                # Skip low betweenness nodes for clarity, except in detailed mode or with dynamic sizing
                continue
                
            # Get node attributes
            layer = G.nodes[node]['layer']
            cluster = G.nodes[node]['cluster']
            original_id = G.nodes[node].get('original_id', node)
            
            # Calculate node size based on betweenness and user preference
            if node_size == "Dynamic":
                # Dynamic sizing: size directly proportional to betweenness
                size = base_size * 0.5 + base_size * 2 * (betweenness / max_node_betweenness)
            else:
                # Standard sizing with a boost based on betweenness
                size = base_size + 300 * (betweenness / max_node_betweenness) * size_factor
            
            # Get node color based on cluster
            if cluster_colors and cluster in cluster_colors:
                color = cluster_colors[cluster]
            else:
                color = plt.cm.tab10(cluster % 10)
            
            # Draw the significant node
            ax.scatter(pos[node][0], pos[node][1], 
                      s=size, color=color, edgecolor='black', linewidth=0.8,
                      zorder=2)  # Higher zorder to draw on top of edges
            
            # Add node label for significant nodes
            if show_labels and (betweenness > 0.3 or node_size == "Dynamic"):
                # Extract layer and original node ID from the node name
                node_parts = node.split('_', 1)
                if len(node_parts) == 2:
                    layer_str, orig_id = node_parts
                else:
                    layer_str, orig_id = str(layer), str(original_id)
                    
                # Create label
                label = f"L{layer_str}_N{orig_id}"
                if betweenness > 0.5:
                    label += f"\nBC={betweenness:.2f}"
                    
                ax.text(pos[node][0], pos[node][1], 
                       label, 
                       fontsize=8,
                       ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=1, boxstyle='round'),
                       zorder=4)  # Highest zorder to draw on top of everything
    
    # Add layer labels if requested
    if emphasize_layers:
        _add_layer_labels(ax, G, pos, layer_nodes, layers)
    
    # Add a legend based on the coloring mode
    if color_by_centrality:
        # Create a colormap for edge betweenness
        sm = ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Edge Betweenness Centrality')
    else:
        # Add a legend for edge types
        inter_line = plt.Line2D([0], [0], color=plt.cm.Blues(0.8), linewidth=3, label='Interlayer Edge')
        intra_line = plt.Line2D([0], [0], color=plt.cm.Reds(0.8), linewidth=3, label='Intralayer Edge')
        ax.legend(handles=[inter_line, intra_line], loc='upper right', framealpha=0.9)
    
    # Add a title with the visualization style
    title = f'Critical Connections in Duplicated Network ({viz_style})'
    ax.set_title(title, fontsize=12)
    
    # Remove axis ticks and spines for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Set equal aspect ratio to prevent distortion
    ax.set_aspect('equal')
    
    # Add a subtle grid for better orientation
    ax.grid(alpha=0.1)
    
    logger.info(f"Created bottleneck visualization with {len(G.nodes)} nodes and {len(G.edges)} edges")

def _create_layer_focused_layout(G, layer_nodes):
    """
    Create a layout that emphasizes the layer structure of the network.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to create a layout for
    layer_nodes : dict
        Dictionary mapping layer indices to lists of node IDs
        
    Returns:
    --------
    dict
        Dictionary mapping node IDs to (x, y) positions
    """
    # Create a position dictionary
    pos = {}
    
    # Get unique layers and sort them
    unique_layers = sorted(layer_nodes.keys())
    num_layers = len(unique_layers)
    
    # Calculate the radius for the outer circle
    radius = 1.0
    
    # Place nodes in concentric circles based on their layer
    for i, layer_idx in enumerate(unique_layers):
        # Get nodes in this layer
        nodes = layer_nodes[layer_idx]
        num_nodes = len(nodes)
        
        # Calculate the radius for this layer's circle
        layer_radius = radius * (0.3 + 0.7 * i / max(1, num_layers - 1))
        
        # Place nodes evenly around the circle
        for j, node in enumerate(nodes):
            angle = 2 * np.pi * j / max(1, num_nodes)
            x = layer_radius * np.cos(angle)
            y = layer_radius * np.sin(angle)
            pos[node] = np.array([x, y])
    
    return pos

def _add_layer_labels(ax, G, pos, layer_nodes, layers):
    """
    Add prominent layer labels to the visualization.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to draw on
    G : networkx.Graph
        The graph being visualized
    pos : dict
        Dictionary mapping node IDs to (x, y) positions
    layer_nodes : dict
        Dictionary mapping layer indices to lists of node IDs
    layers : list
        List of layer names
    """
    # Get unique layers
    unique_layers = sorted(layer_nodes.keys())
    
    # Calculate the center position for each layer
    layer_centers = {}
    for layer_idx in unique_layers:
        nodes = layer_nodes[layer_idx]
        if not nodes:
            continue
            
        # Calculate the average position of nodes in this layer
        x_sum = sum(pos[node][0] for node in nodes)
        y_sum = sum(pos[node][1] for node in nodes)
        
        # Calculate the center
        center_x = x_sum / len(nodes)
        center_y = y_sum / len(nodes)
        
        layer_centers[layer_idx] = (center_x, center_y)
    
    # Add layer labels with background boxes
    for layer_idx, (center_x, center_y) in layer_centers.items():
        # Get the layer name
        if layer_idx < len(layers):
            layer_name = layers[layer_idx]
        else:
            layer_name = f"Layer {layer_idx}"
            
        # Add a background box for the label
        ax.text(center_x, center_y, layer_name,
               fontsize=10, fontweight='bold',
               ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', 
                        boxstyle='round,pad=0.5'),
               zorder=5)  # Highest zorder to draw on top of everything

def integrate_lc16_ui_with_panel(panel, parent_layout, analysis_combo=None, cluster_combo=None):
    """
    Integrate the LC16 UI elements with the visualization panel.
    
    Parameters:
    -----------
    panel : QWidget
        The panel to integrate the UI elements with
    parent_layout : QLayout
        The layout to add the UI elements to
    analysis_combo : QComboBox, optional
        An existing analysis type combo box to connect with the UI elements
    cluster_combo : QComboBox, optional
        An existing cluster selection combo box to connect with the UI elements
        
    Returns:
    --------
    dict
        A dictionary containing the UI elements and layouts
    """
    # Create the UI elements
    ui_elements = create_lc16_ui_elements(panel)
    
    # Add the group box to the parent layout
    parent_layout.addWidget(ui_elements["group"])
    
    # Connect signals to slots if the panel has an update method
    if hasattr(panel, "update_lc16_path_analysis") and callable(panel.update_lc16_path_analysis):
        # Connect the UI elements to the update method
        ui_elements["viz_style_combo"].currentIndexChanged.connect(
            lambda: panel.update_lc16_path_analysis(panel._current_data) if hasattr(panel, "_current_data") else None
        )
        ui_elements["show_labels_checkbox"].stateChanged.connect(
            lambda: panel.update_lc16_path_analysis(panel._current_data) if hasattr(panel, "_current_data") else None
        )
        ui_elements["show_nodes_checkbox"].stateChanged.connect(
            lambda: panel.update_lc16_path_analysis(panel._current_data) if hasattr(panel, "_current_data") else None
        )
        ui_elements["color_by_centrality_checkbox"].stateChanged.connect(
            lambda: panel.update_lc16_path_analysis(panel._current_data) if hasattr(panel, "_current_data") else None
        )
        
        # Connect the new UI elements
        ui_elements["hide_unconnected_checkbox"].stateChanged.connect(
            lambda: panel.update_lc16_path_analysis(panel._current_data) if hasattr(panel, "_current_data") else None
        )
        ui_elements["emphasize_layers_checkbox"].stateChanged.connect(
            lambda: panel.update_lc16_path_analysis(panel._current_data) if hasattr(panel, "_current_data") else None
        )
        ui_elements["node_size_combo"].currentIndexChanged.connect(
            lambda: panel.update_lc16_path_analysis(panel._current_data) if hasattr(panel, "_current_data") else None
        )
        ui_elements["layout_spacing_combo"].currentIndexChanged.connect(
            lambda: panel.update_lc16_path_analysis(panel._current_data) if hasattr(panel, "_current_data") else None
        )
        
        # Connect the analysis combo box if provided
        if analysis_combo is not None:
            analysis_combo.currentIndexChanged.connect(
                lambda: panel.update_lc16_path_analysis(panel._current_data) if hasattr(panel, "_current_data") else None
            )
            
        # Connect the cluster combo box if provided
        if cluster_combo is not None:
            cluster_combo.currentIndexChanged.connect(
                lambda: panel.update_lc16_path_analysis(panel._current_data) if hasattr(panel, "_current_data") else None
            )
    
    # Store references to the UI elements in the panel for later access
    panel.lc16_ui_elements = ui_elements
    
    return ui_elements

def get_lc16_visualization_settings(panel):
    """
    Get the current visualization settings from the LC16 UI elements.
    
    Parameters:
    -----------
    panel : QWidget
        The panel containing the LC16 UI elements
        
    Returns:
    --------
    dict
        A dictionary containing the current visualization settings
    """
    settings = {}
    
    # Check if the panel has the UI elements
    if hasattr(panel, "lc16_ui_elements"):
        ui_elements = panel.lc16_ui_elements
        
        # Get the visualization style
        settings["viz_style"] = ui_elements["viz_style_combo"].currentText()
        
        # Get the checkbox states
        settings["show_labels"] = ui_elements["show_labels_checkbox"].isChecked()
        settings["show_nodes"] = ui_elements["show_nodes_checkbox"].isChecked()
        settings["color_by_centrality"] = ui_elements["color_by_centrality_checkbox"].isChecked()
        
        # Get the new UI element states
        settings["hide_unconnected"] = ui_elements["hide_unconnected_checkbox"].isChecked()
        settings["emphasize_layers"] = ui_elements["emphasize_layers_checkbox"].isChecked()
        settings["node_size"] = ui_elements["node_size_combo"].currentText()
        settings["layout_spacing"] = ui_elements["layout_spacing_combo"].currentText()
    else:
        # Default settings if UI elements are not available
        settings["viz_style"] = "Standard"
        settings["show_labels"] = True
        settings["show_nodes"] = True
        settings["color_by_centrality"] = False
        settings["hide_unconnected"] = False
        settings["emphasize_layers"] = True
        settings["node_size"] = "Medium"
        settings["layout_spacing"] = "Standard"
        
    return settings

def get_selected_cluster(panel):
    """
    Get the selected cluster from the cluster combo box.
    
    Parameters:
    -----------
    panel : QWidget
        The panel containing the cluster combo box
        
    Returns:
    --------
    int or None
        The selected cluster number, or None if "All Clusters" is selected
    """
    selected_cluster = None
    
    # Check if the panel has the cluster combo box
    if hasattr(panel, "path_similarity_cluster_combo"):
        cluster_selection = panel.path_similarity_cluster_combo.currentText()
        try:
            # Handle different possible formats of cluster selection text
            if cluster_selection == "All Clusters":
                selected_cluster = None
            else:
                # Try to extract the cluster number, handling different formats
                parts = cluster_selection.split()
                if len(parts) > 1 and parts[0].lower() == "cluster":
                    selected_cluster = int(parts[1])
                elif len(parts) > 1:
                    # Try to find a number in the parts
                    for part in parts:
                        if part.isdigit():
                            selected_cluster = int(part)
                            break
                    else:
                        selected_cluster = None
                else:
                    # If it's just a number
                    selected_cluster = (
                        int(cluster_selection) if cluster_selection.isdigit() else None
                    )
        except (ValueError, IndexError):
            # If parsing fails, default to All Clusters
            import logging
            logging.warning(
                f"Could not parse cluster selection: {cluster_selection}, defaulting to All Clusters"
            )
            selected_cluster = None
            
    return selected_cluster 