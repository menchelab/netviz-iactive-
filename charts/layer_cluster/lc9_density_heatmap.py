import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_layer_cluster_density_heatmap(
    ax,
    visible_links,
    node_ids,
    node_clusters,
    nodes_per_layer,  # This is now an integer
    layers,
    small_font,
    medium_font,
    visible_layer_indices=None,
    cluster_colors=None,
):
    """
    Create a heatmap showing the density of nodes per layer and cluster.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to draw the heatmap on
    visible_links : list of tuples
        List of (start_idx, end_idx) tuples representing visible links
    node_ids : list
        List of node IDs
    node_clusters : dict
        Dictionary mapping node IDs to cluster labels
    nodes_per_layer : int
        Number of nodes in each layer
    layers : list
        List of layer names
    small_font : dict
        Dictionary with font properties for small text
    medium_font : dict
        Dictionary with font properties for medium text
    visible_layer_indices : list, optional
        List of indices of visible layers
    cluster_colors : dict, optional
        Dictionary mapping cluster labels to colors
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating layer-cluster density heatmap with nodes_per_layer={nodes_per_layer}")
    logger.info(f"Visible links: {len(visible_links)}, Node IDs: {len(node_ids)}, Clusters: {len(node_clusters)}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Visible layer indices: {visible_layer_indices}")
    
    # Handle font parameters correctly
    if isinstance(medium_font, dict):
        medium_fontsize = medium_font.get("fontsize", 12)
    else:
        medium_fontsize = medium_font
        
    if isinstance(small_font, dict):
        small_fontsize = small_font.get("fontsize", 10)
    else:
        small_fontsize = small_font
    
    logger.info(f"Using medium_fontsize={medium_fontsize}, small_fontsize={small_fontsize}")

    # Clear the axis
    ax.clear()
    
    # Create a figure with 2x2 grid layout (we'll use the provided ax as the main container)
    fig = ax.figure
    ax.set_axis_off()  # Turn off the main axis
    
    # Create 2x2 grid of subplots within the main axis area
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.4, bottom=0.05, top=0.9, left=0.1, right=0.9)
    ax_original = fig.add_subplot(gs[0, 0])  # Original Combined Heatmap
    ax_count = fig.add_subplot(gs[0, 1])     # Node Count Matrix
    ax_conn = fig.add_subplot(gs[1, 0])      # Connection Matrix
    ax_density = fig.add_subplot(gs[1, 1])   # Density Matrix
    
    # Add a title to the entire figure
    fig.suptitle("Layer-Cluster Analysis for Multilayer Networks", fontsize=medium_fontsize+2)

    # Filter visible layers if specified
    if visible_layer_indices is not None:
        visible_layers = set(visible_layer_indices)
        logger.info(f"Filtering heatmap to show only {len(visible_layers)} visible layers")
    else:
        visible_layers = set(range(len(layers)))
        logger.info("No layer filtering applied")

    # Get visible node indices and build a mapping of node_idx to layer and cluster
    visible_node_indices = set()
    node_layer_map = {}
    node_cluster_map = {}
    
    for start_idx, end_idx in visible_links:
        visible_node_indices.add(start_idx)
        visible_node_indices.add(end_idx)
    
    logger.info(f"Found {len(visible_node_indices)} visible nodes")

    # Count nodes by cluster and layer
    cluster_layer_counts = defaultdict(lambda: defaultdict(int))
    
    # Track nodes by cluster and layer for connection density calculation
    nodes_by_cluster_layer = defaultdict(lambda: defaultdict(set))
    
    for node_idx in visible_node_indices:
        # Calculate layer index using integer division
        layer_idx = node_idx // nodes_per_layer
        
        # Skip if layer is not visible
        if layer_idx not in visible_layers:
            continue
            
        node_id = node_ids[node_idx]
        cluster = node_clusters.get(node_id, "Unknown")
        
        # Store the mapping
        node_layer_map[node_idx] = layer_idx
        node_cluster_map[node_idx] = cluster
        
        # Count nodes by cluster and layer
        cluster_layer_counts[cluster][layer_idx] += 1
        
        # Track nodes by cluster and layer
        nodes_by_cluster_layer[cluster][layer_idx].add(node_idx)
    
    logger.info(f"Counted nodes by cluster and layer: {dict(cluster_layer_counts)}")

    # Check if we have any data
    if not cluster_layer_counts:
        logger.warning("No cluster-layer data to display")
        for subplot_ax in [ax_original, ax_count, ax_conn, ax_density]:
            subplot_ax.text(
                0.5,
                0.5,
                "No cluster-layer data to display",
                horizontalalignment="center",
                verticalalignment="center",
            )
            subplot_ax.axis("off")
        return

    # Get unique layers and clusters
    unique_clusters = sorted(cluster_layer_counts.keys())
    unique_layer_indices = sorted(visible_layers)
    unique_layers = [layers[i] for i in unique_layer_indices if i < len(layers)]
    
    logger.info(f"Found {len(unique_clusters)} unique clusters and {len(unique_layers)} unique layers")
    
    # Create matrices for the heatmap
    count_matrix = np.zeros((len(unique_clusters), len(unique_layers)))
    connection_matrix = np.zeros((len(unique_clusters), len(unique_layers)))
    max_possible_matrix = np.zeros((len(unique_clusters), len(unique_layers)))
    
    # Fill the count matrix with node counts
    for i, cluster in enumerate(unique_clusters):
        for j, layer_idx in enumerate(unique_layer_indices):
            if layer_idx < len(layers):  # Ensure layer index is valid
                count_matrix[i, j] = cluster_layer_counts[cluster][layer_idx]
    
    # Count actual connections within each cluster-layer combination
    for start_idx, end_idx in visible_links:
        # Skip if either node is not in our maps
        if start_idx not in node_layer_map or end_idx not in node_layer_map:
            continue
        
        start_layer = node_layer_map[start_idx]
        end_layer = node_layer_map[end_idx]
        start_cluster = node_cluster_map[start_idx]
        end_cluster = node_cluster_map[end_idx]
        
        # For multilayer networks, we count connections differently:
        # 1. If both nodes are in the same layer and same cluster, count as intra-cluster connection
        # 2. If nodes are in different layers but same cluster, count as inter-layer connection
        
        if start_cluster == end_cluster:
            # Find the index of the cluster and layer in our matrices
            cluster_idx = unique_clusters.index(start_cluster)
            
            # For same layer connections
            if start_layer == end_layer and start_layer in unique_layer_indices:
                layer_idx = unique_layer_indices.index(start_layer)
                connection_matrix[cluster_idx, layer_idx] += 1
            
            # For inter-layer connections, we count them for both layers
            elif start_layer in unique_layer_indices and end_layer in unique_layer_indices:
                start_layer_idx = unique_layer_indices.index(start_layer)
                end_layer_idx = unique_layer_indices.index(end_layer)
                # Add 0.5 to each layer to avoid double counting
                connection_matrix[cluster_idx, start_layer_idx] += 0.5
                connection_matrix[cluster_idx, end_layer_idx] += 0.5
    
    # Calculate maximum possible connections for each cluster-layer combination
    for i, cluster in enumerate(unique_clusters):
        for j, layer_idx in enumerate(unique_layer_indices):
            nodes_in_cluster_layer = len(nodes_by_cluster_layer[cluster][layer_idx])
            
            if nodes_in_cluster_layer > 1:
                # Maximum possible connections within a layer = n(n-1)/2
                max_possible_matrix[i, j] = (nodes_in_cluster_layer * (nodes_in_cluster_layer - 1)) / 2
            else:
                max_possible_matrix[i, j] = 0
    
    # Calculate density as actual connections / maximum possible connections
    # Avoid division by zero
    density_matrix = np.zeros_like(connection_matrix)
    for i in range(len(unique_clusters)):
        for j in range(len(unique_layers)):
            if max_possible_matrix[i, j] > 0:
                density_matrix[i, j] = connection_matrix[i, j] / max_possible_matrix[i, j]
            elif count_matrix[i, j] > 0:
                # If there are nodes but no possible connections (e.g., only 1 node),
                # we use the normalized count instead
                density_matrix[i, j] = count_matrix[i, j] / np.sum(count_matrix[:, j])
    
    logger.info(f"Created matrices with shape {count_matrix.shape}")
    
    # Create custom colormaps
    count_cmap = plt.cm.Blues
    connection_cmap = plt.cm.Greens
    density_cmap = LinearSegmentedColormap.from_list('WhiteToBlue', ['#FFFFFF', '#0343DF'])
    original_cmap = plt.cm.Blues
    
    # 1. Plot the Original Combined Heatmap (top-left)
    im_orig = ax_original.imshow(count_matrix, cmap=original_cmap, aspect='auto')
    ax_original.set_title("Original Density Heatmap", fontsize=medium_fontsize)
    
    # Add colorbar
    cbar_orig = plt.colorbar(im_orig, ax=ax_original, shrink=0.8)
    cbar_orig.set_label('Node count', fontsize=small_fontsize)
    
    # Add text annotations for raw counts
    for i in range(len(unique_clusters)):
        for j in range(len(unique_layers)):
            count = int(count_matrix[i, j])
            if count > 0:
                # Use white text for dark cells, black for light cells
                text_color = 'white' if count > np.max(count_matrix) * 0.7 else 'black'
                ax_original.text(j, i, str(count), 
                        ha="center", va="center", 
                        color=text_color, fontsize=small_fontsize)
    
    # 2. Plot the Node Count Matrix (top-right)
    im_count = ax_count.imshow(count_matrix, cmap=count_cmap, aspect='auto')
    ax_count.set_title("Node Count Matrix", fontsize=medium_fontsize)
    
    # Add colorbar
    cbar_count = plt.colorbar(im_count, ax=ax_count, shrink=0.8)
    cbar_count.set_label('Number of nodes', fontsize=small_fontsize)
    
    # Add text annotations for raw counts
    for i in range(len(unique_clusters)):
        for j in range(len(unique_layers)):
            count = int(count_matrix[i, j])
            if count > 0:
                # Use white text for dark cells, black for light cells
                text_color = 'white' if count > np.max(count_matrix) * 0.7 else 'black'
                ax_count.text(j, i, str(count), 
                        ha="center", va="center", 
                        color=text_color, fontsize=small_fontsize)
    
    # 3. Plot the Connection Matrix (bottom-left)
    im_conn = ax_conn.imshow(connection_matrix, cmap=connection_cmap, aspect='auto')
    ax_conn.set_title("Connection Matrix", fontsize=medium_fontsize)
    
    # Add colorbar
    cbar_conn = plt.colorbar(im_conn, ax=ax_conn, shrink=0.8)
    cbar_conn.set_label('Number of connections', fontsize=small_fontsize)
    
    # Add text annotations for connection counts
    for i in range(len(unique_clusters)):
        for j in range(len(unique_layers)):
            conn = connection_matrix[i, j]
            if conn > 0:
                # Use white text for dark cells, black for light cells
                text_color = 'white' if conn > np.max(connection_matrix) * 0.7 else 'black'
                ax_conn.text(j, i, f"{conn:.1f}", 
                        ha="center", va="center", 
                        color=text_color, fontsize=small_fontsize)
    
    # 4. Plot the Density Matrix (bottom-right)
    im_density = ax_density.imshow(density_matrix, cmap=density_cmap, aspect='auto', vmin=0, vmax=1.0)
    ax_density.set_title("Density Matrix", fontsize=medium_fontsize)
    
    # Add colorbar
    cbar_density = plt.colorbar(im_density, ax=ax_density, shrink=0.8)
    cbar_density.set_label('Density (0-1)', fontsize=small_fontsize)
    
    # Add text annotations for density values
    for i in range(len(unique_clusters)):
        for j in range(len(unique_layers)):
            density = density_matrix[i, j] / 2
            if density > 0:
                # Use white text for dark cells, black for light cells
                text_color = 'white' if density > 0.5 else 'black'
                ax_density.text(j, i, f"{density:.2f}", 
                        ha="center", va="center", 
                        color=text_color, fontsize=small_fontsize)
    
    # Add common labels and formatting to all subplots
    for subplot_ax in [ax_original, ax_count, ax_conn, ax_density]:
        # Add labels
        subplot_ax.set_xticks(np.arange(len(unique_layers)))
        subplot_ax.set_yticks(np.arange(len(unique_clusters)))
        subplot_ax.set_xticklabels(unique_layers, rotation=45, ha='right', fontsize=small_fontsize-1)
        subplot_ax.set_yticklabels(unique_clusters, fontsize=small_fontsize-1)
        
        # Add grid lines
        subplot_ax.set_xticks(np.arange(-.5, len(unique_layers), 1), minor=True)
        subplot_ax.set_yticks(np.arange(-.5, len(unique_clusters), 1), minor=True)
        subplot_ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        # Add axis labels
        subplot_ax.set_xlabel("Layers", fontsize=small_fontsize)
        subplot_ax.set_ylabel("Clusters", fontsize=small_fontsize)
    
    # Add a subtitle with more information
    fig.text(0.5, 0.95, f"Multilayer Network Analysis ({len(unique_clusters)} clusters, {len(unique_layers)} layers)", 
             fontsize=small_fontsize, ha='center')
    
    logger.info(f"Created density heatmap with {len(unique_clusters)} clusters and {len(unique_layers)} layers") 