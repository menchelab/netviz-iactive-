import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_cluster_connectivity_matrix(
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
    edge_type="all",  # New parameter to filter by edge type: "all", "same_layer", or "interlayer"
):
    """
    Create a matrix showing the connectivity between clusters.
    Each cell represents the number of connections between two clusters.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to draw the matrix on
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
    edge_type : str, optional
        Type of edges to include: "all", "same_layer", or "interlayer"
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating cluster connectivity matrix for edge type '{edge_type}' with nodes_per_layer={nodes_per_layer}")
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
    
    # Set title based on edge type
    if edge_type == "all":
        title = "Cluster Connectivity Matrix (All Edges)"
    elif edge_type == "same_layer":
        title = "Cluster Connectivity Matrix (Same Layer)"
    elif edge_type == "interlayer":
        title = "Cluster Connectivity Matrix (Interlayer)"
    else:
        title = "Cluster Connectivity Matrix"
    
    ax.set_title(title, fontsize=medium_fontsize)

    # Filter visible layers if specified
    if visible_layer_indices is not None:
        visible_layers = set(visible_layer_indices)
        logger.info(f"Filtering connectivity matrix to show only {len(visible_layers)} visible layers")
    else:
        visible_layers = set(range(len(layers)))
        logger.info("No layer filtering applied")

    # Get visible node indices
    visible_node_indices = set()
    for start_idx, end_idx in visible_links:
        visible_node_indices.add(start_idx)
        visible_node_indices.add(end_idx)
    
    logger.info(f"Found {len(visible_node_indices)} visible nodes")

    # Count connections between clusters
    cluster_connections = defaultdict(lambda: defaultdict(int))
    
    # Track unique clusters
    unique_clusters = set()
    
    # Process each link
    for start_idx, end_idx in visible_links:
        # Get layer indices using integer division
        start_layer = start_idx // nodes_per_layer
        end_layer = end_idx // nodes_per_layer
        
        # Skip if either layer is not visible
        if start_layer not in visible_layers or end_layer not in visible_layers:
            continue
        
        # Filter by edge type
        if edge_type == "same_layer" and start_layer != end_layer:
            continue
        elif edge_type == "interlayer" and start_layer == end_layer:
            continue
            
        # Get node IDs
        start_id = node_ids[start_idx]
        end_id = node_ids[end_idx]
        
        # Get clusters
        start_cluster = node_clusters.get(start_id, "Unknown")
        end_cluster = node_clusters.get(end_id, "Unknown")
        
        # Skip self-connections within the same node
        if start_idx == end_idx:
            continue
            
        # Add to unique clusters
        unique_clusters.add(start_cluster)
        unique_clusters.add(end_cluster)
        
        # Count connection between clusters
        cluster_connections[start_cluster][end_cluster] += 1
        
        # Also count in the opposite direction for undirected graph
        if start_cluster != end_cluster:
            cluster_connections[end_cluster][start_cluster] += 1
    
    # Sort clusters for consistent ordering
    sorted_clusters = sorted(unique_clusters)
    
    logger.info(f"Found {len(sorted_clusters)} unique clusters with connections for edge type '{edge_type}'")
    
    # Check if we have any data
    if not sorted_clusters:
        logger.warning(f"No cluster connectivity data to display for edge type '{edge_type}'")
        ax.text(
            0.5,
            0.5,
            f"No cluster connectivity data to display for edge type '{edge_type}'",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off")
        return
    
    # Create the connectivity matrix
    matrix_size = len(sorted_clusters)
    connectivity_matrix = np.zeros((matrix_size, matrix_size))
    
    # Fill the matrix
    for i, cluster1 in enumerate(sorted_clusters):
        for j, cluster2 in enumerate(sorted_clusters):
            connectivity_matrix[i, j] = cluster_connections[cluster1][cluster2]
    
    logger.info(f"Created connectivity matrix of size {matrix_size}x{matrix_size} for edge type '{edge_type}'")
    logger.info(f"Matrix values range: min={np.min(connectivity_matrix)}, max={np.max(connectivity_matrix)}")
    
    try:
        # Create a custom colormap from white to purple
        cmap = LinearSegmentedColormap.from_list(
            "WhitePurple", [(1, 1, 1), (0.5, 0, 0.5)], N=100
        )
        
        # Plot the matrix
        im = ax.imshow(connectivity_matrix, cmap=cmap, interpolation='nearest')
        
        # Add a colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Number of Connections", fontsize=small_fontsize)
        cbar.ax.tick_params(labelsize=small_fontsize)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(matrix_size))
        ax.set_yticks(np.arange(matrix_size))
        ax.set_xticklabels(sorted_clusters, rotation=45, ha="right", fontsize=small_fontsize)
        ax.set_yticklabels(sorted_clusters, fontsize=small_fontsize)
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, matrix_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, matrix_size, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.2)
        
        # Add text annotations for connection counts
        for i in range(matrix_size):
            for j in range(matrix_size):
                value = connectivity_matrix[i, j]
                if value > 0:
                    text_color = "black" if value < np.max(connectivity_matrix) * 0.7 else "white"
                    ax.text(
                        j, i, int(value),
                        ha="center", va="center",
                        color=text_color,
                        fontsize=small_fontsize
                    )
        
        # Add axis labels
        ax.set_xlabel("Clusters", fontsize=small_fontsize)
        ax.set_ylabel("Clusters", fontsize=small_fontsize)
        
        # Add a title with more information
        edge_type_str = "all edges" if edge_type == "all" else f"{edge_type} edges"
        ax.set_title(f"Cluster Connectivity Matrix\n({len(sorted_clusters)} clusters, {edge_type_str})", 
                    fontsize=medium_fontsize)
        
        logger.info(f"Successfully created cluster connectivity matrix with {len(sorted_clusters)} clusters for edge type '{edge_type}'")
        
    except Exception as e:
        logger.error(f"Error creating cluster connectivity matrix for edge type '{edge_type}': {str(e)}")
        ax.clear()
        ax.text(
            0.5,
            0.5,
            f"Error creating matrix: {str(e)}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off")

def create_layer_connectivity_matrix(
    ax,
    visible_links,
    node_ids,
    node_clusters,
    nodes_per_layer,
    layers,
    small_font,
    medium_font,
    visible_layer_indices=None,
    layer_colors=None,
    edge_type="all",  # "all", "same_cluster", or "different_cluster"
):
    """
    Create a matrix showing the connectivity between layers.
    Each cell represents the number of connections between two layers.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to draw the matrix on
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
    layer_colors : dict, optional
        Dictionary mapping layer indices to colors
    edge_type : str, optional
        Type of edges to include: "all", "same_cluster", or "different_cluster"
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating layer connectivity matrix for edge type '{edge_type}' with nodes_per_layer={nodes_per_layer}")
    logger.info(f"Visible links: {len(visible_links)}, Node IDs: {len(node_ids)}, Layers: {len(layers)}")
    
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
    
    # Set title based on edge type
    if edge_type == "all":
        title = "Layer Connectivity Matrix (All Edges)"
    elif edge_type == "same_cluster":
        title = "Layer Connectivity Matrix (Same Cluster)"
    elif edge_type == "different_cluster":
        title = "Layer Connectivity Matrix (Different Clusters)"
    else:
        title = "Layer Connectivity Matrix"
    
    ax.set_title(title, fontsize=medium_fontsize)

    # Filter visible layers if specified
    if visible_layer_indices is not None:
        visible_layers = set(visible_layer_indices)
        logger.info(f"Filtering layer connectivity matrix to show only {len(visible_layers)} visible layers")
    else:
        visible_layers = set(range(len(layers)))
        logger.info("No layer filtering applied")

    # Count connections between layers
    layer_connections = defaultdict(lambda: defaultdict(int))
    
    # Process each link
    for start_idx, end_idx in visible_links:
        # Get layer indices using integer division
        start_layer = start_idx // nodes_per_layer
        end_layer = end_idx // nodes_per_layer
        
        # Skip if either layer is not visible
        if start_layer not in visible_layers or end_layer not in visible_layers:
            continue
        
        # Get node IDs
        start_id = node_ids[start_idx]
        end_id = node_ids[end_idx]
        
        # Get clusters
        start_cluster = node_clusters.get(start_id, "Unknown")
        end_cluster = node_clusters.get(end_id, "Unknown")
        
        # Skip self-connections within the same node
        if start_idx == end_idx:
            continue
        
        # Filter by edge type
        if edge_type == "same_cluster" and start_cluster != end_cluster:
            continue
        elif edge_type == "different_cluster" and start_cluster == end_cluster:
            continue
        
        # Count connection between layers
        layer_connections[start_layer][end_layer] += 1
        
        # Also count in the opposite direction for undirected graph
        if start_layer != end_layer:
            layer_connections[end_layer][start_layer] += 1
    
    # Get visible layer indices in sorted order
    sorted_layers = sorted(visible_layers)
    
    logger.info(f"Found {len(sorted_layers)} visible layers with connections for edge type '{edge_type}'")
    
    # Check if we have any data
    if not sorted_layers:
        logger.warning(f"No layer connectivity data to display for edge type '{edge_type}'")
        ax.text(
            0.5,
            0.5,
            f"No layer connectivity data to display for edge type '{edge_type}'",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off")
        return
    
    # Create the connectivity matrix
    matrix_size = len(sorted_layers)
    connectivity_matrix = np.zeros((matrix_size, matrix_size))
    
    # Fill the matrix
    for i, layer1_idx in enumerate(sorted_layers):
        for j, layer2_idx in enumerate(sorted_layers):
            connectivity_matrix[i, j] = layer_connections[layer1_idx][layer2_idx]
    
    logger.info(f"Created layer connectivity matrix of size {matrix_size}x{matrix_size} for edge type '{edge_type}'")
    logger.info(f"Matrix values range: min={np.min(connectivity_matrix)}, max={np.max(connectivity_matrix)}")
    
    try:
        # Create a custom colormap from white to blue
        cmap = LinearSegmentedColormap.from_list(
            "WhiteBlue", [(1, 1, 1), (0, 0, 0.8)], N=100
        )
        
        # Plot the matrix
        im = ax.imshow(connectivity_matrix, cmap=cmap, interpolation='nearest')
        
        # Add a colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Number of Connections", fontsize=small_fontsize)
        cbar.ax.tick_params(labelsize=small_fontsize)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(matrix_size))
        ax.set_yticks(np.arange(matrix_size))
        
        # Use layer names for labels if available
        x_labels = [layers[idx] if idx < len(layers) else f"Layer {idx}" for idx in sorted_layers]
        y_labels = [layers[idx] if idx < len(layers) else f"Layer {idx}" for idx in sorted_layers]
        
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=small_fontsize)
        ax.set_yticklabels(y_labels, fontsize=small_fontsize)
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, matrix_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, matrix_size, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.2)
        
        # Add text annotations for connection counts
        for i in range(matrix_size):
            for j in range(matrix_size):
                value = connectivity_matrix[i, j]
                if value > 0:
                    text_color = "black" if value < np.max(connectivity_matrix) * 0.7 else "white"
                    ax.text(
                        j, i, int(value),
                        ha="center", va="center",
                        color=text_color,
                        fontsize=small_fontsize
                    )
        
        # Add axis labels
        ax.set_xlabel("Layers", fontsize=small_fontsize)
        ax.set_ylabel("Layers", fontsize=small_fontsize)
        
        # Add a title with more information
        edge_type_str = "all edges" if edge_type == "all" else f"{edge_type} edges"
        ax.set_title(f"Layer Connectivity Matrix\n({len(sorted_layers)} layers, {edge_type_str})", 
                    fontsize=medium_fontsize)
        
        logger.info(f"Successfully created layer connectivity matrix with {len(sorted_layers)} layers for edge type '{edge_type}'")
        
    except Exception as e:
        logger.error(f"Error creating layer connectivity matrix for edge type '{edge_type}': {str(e)}")
        ax.clear()
        ax.text(
            0.5,
            0.5,
            f"Error creating matrix: {str(e)}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off") 