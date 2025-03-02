import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import logging
from collections import defaultdict
import matplotlib.path as mpath
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import squarify
import networkx as nx
from itertools import combinations
import math

def get_visible_nodes(visible_links):
    """
    Extract the set of visible node indices from the visible links.
    
    Args:
        visible_links: List of (start_idx, end_idx) tuples representing visible links
        
    Returns:
        Set of visible node indices
    """
    visible_node_indices = set()
    for start_idx, end_idx in visible_links:
        visible_node_indices.add(start_idx)
        visible_node_indices.add(end_idx)
    return visible_node_indices

def count_nodes_by_cluster_and_layer(visible_node_indices, node_ids, node_clusters, nodes_per_layer, visible_layers=None):
    """
    Count the number of nodes in each cluster and layer combination.
    
    Args:
        visible_node_indices: Set of visible node indices
        node_ids: List of node IDs
        node_clusters: Dictionary mapping node IDs to cluster IDs
        nodes_per_layer: Integer representing the number of nodes in each layer
        visible_layers: Set of visible layer indices (optional)
        
    Returns:
        counts: Dictionary mapping (cluster, layer) tuples to node counts
        unique_clusters: Set of unique cluster IDs
        layer_indices: List of layer indices (filtered by visibility if applicable)
    """
    # Initialize counts
    counts = defaultdict(int)
    
    # Track unique clusters
    unique_clusters = set()
    
    # Determine which layers to include
    if visible_layers is not None:
        layer_indices = sorted(visible_layers)
    else:
        # Calculate the number of layers based on the maximum node index
        if visible_node_indices:
            max_node_idx = max(visible_node_indices)
            num_layers = (max_node_idx // nodes_per_layer) + 1
            layer_indices = list(range(num_layers))
        else:
            layer_indices = []
    
    # Count nodes by cluster and layer
    for node_idx in visible_node_indices:
        layer_idx = node_idx // nodes_per_layer
        
        if visible_layers is not None and layer_idx not in visible_layers:
            continue
            
        node_id = node_ids[node_idx]
        if node_id in node_clusters:
            cluster = node_clusters[node_id]
            unique_clusters.add(cluster)
            counts[(cluster, layer_idx)] += 1
    
    return counts, unique_clusters, layer_indices

def blend_colors(color1, color2, alpha=0.5):
    """
    Blend two colors together.
    
    Args:
        color1: First color (as RGBA tuple or hex string)
        color2: Second color (as RGBA tuple or hex string)
        alpha: Blending factor (0.0 to 1.0)
        
    Returns:
        Blended color as RGBA tuple
    """
    # Convert hex strings to RGBA if necessary
    if isinstance(color1, str):
        color1 = to_rgba(color1)
    if isinstance(color2, str):
        color2 = to_rgba(color2)
    
    # Blend the colors
    r = color1[0] * alpha + color2[0] * (1 - alpha)
    g = color1[1] * alpha + color2[1] * (1 - alpha)
    b = color1[2] * alpha + color2[2] * (1 - alpha)
    a = color1[3] * alpha + color2[3] * (1 - alpha)
    
    return (r, g, b, a)

def get_base_node_id(node_id):
    """
    Extract the base node ID from a node ID that may include layer information.
    
    Args:
        node_id: Node ID (possibly with layer information)
        
    Returns:
        Base node ID (without layer information)
    """
    # If the node ID contains a colon, extract the part before the colon
    if isinstance(node_id, str) and ':' in node_id:
        return node_id.split(':')[0]
    return node_id 