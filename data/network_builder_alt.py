import pandas as pd
import numpy as np
import networkx as nx
import logging
from tqdm import tqdm
from utils.color_utils import generate_distinct_colors
from utils.calc_layout import get_layout_position


def calculate_layer_layout(G, layer_nodes, layout_algorithm="kamada_kawai"):
    """Calculate layout for a specific layer's nodes"""
    subgraph = G.subgraph(layer_nodes)
    return get_layout_position(subgraph, layout_algorithm=layout_algorithm)


def build_multilayer_network(
    node_file_path,
    edge_file_path,
    add_interlayer_edges=False,
    use_ml_layout=False,
    layout_algorithm="kamada_kawai",
    z_offset=2.0,
):
    logger = logging.getLogger(__name__)
    logger.info(f"Building multilayer network from Hetionet files")

    # Read the node metadata and edge list
    node_metadata = pd.read_table(node_file_path, sep="\t")
    edgelist = pd.read_table(edge_file_path, sep="\t")
    logger.info(f"Loaded {len(node_metadata)} nodes and {len(edgelist)} edges")

    # Get layers and count nodes per layer
    layers = pd.unique(node_metadata["kind"]).tolist()
    nodes_per_layer = {layer: len(node_metadata[node_metadata["kind"] == layer]) 
                      for layer in layers}
    logger.info(f"Found {len(layers)} layers with node counts: {nodes_per_layer}")

    # Configuration for circular arrangement
    cylinder_radius = 80  # Distance from center to each layer
    grid_spacing = 0.5   # Spacing between nodes in grid

    def create_grid_layout(num_nodes, spacing):
        """Calculate grid dimensions and positions for a layer"""
        if num_nodes == 0:
            return [], 0, 0
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(num_nodes)))
        
        # Calculate actual positions (vertical grid)
        positions = []
        max_y = max_z = 0
        for i in range(num_nodes):
            # Now creating a vertical grid (y-z plane)
            col = i // grid_size
            row = i % grid_size
            # Center the grid around origin
            y = (col - grid_size/2) * spacing
            z = (row - grid_size/2) * spacing
            positions.append((y, z))
            max_y = max(max_y, abs(y))
            max_z = max(max_z, abs(z))
        
        return positions, max_y * 2, max_z * 2

    # Generate colors
    kind_colors = {kind: color for kind, color in zip(layers, generate_distinct_colors(len(layers)))}
    edge_colors = {edge_type: color for edge_type, color in 
                  zip(edgelist["metaedge"].unique(), generate_distinct_colors(len(edgelist["metaedge"].unique())))}

    # Create node positions and metadata
    node_positions = []
    node_ids = []
    node_colors = []
    node_clusters = {}
    node_origins = {}
    
    # Calculate angle between layers
    num_layers = len(layers)
    angle_step = 2 * np.pi / num_layers
    
    for layer_idx, layer in enumerate(tqdm(layers, desc="Processing layers")):
        # Calculate layer angle in the circle
        layer_angle = layer_idx * angle_step
        
        # Get nodes for this layer
        layer_nodes = node_metadata[node_metadata["kind"] == layer]
        num_nodes = len(layer_nodes)
        
        # Calculate vertical grid layout for this layer
        grid_positions, width, height = create_grid_layout(num_nodes, grid_spacing)
        layer_diameter = max(width, height)
        
        logger.info(f"Layer {layer}: {num_nodes} nodes, grid size: {width:.2f}x{height:.2f}, angle={layer_angle:.2f}")
        
        # Position nodes in vertical grid and transform to cylinder coordinates
        for (y, z), (_, node) in zip(grid_positions, layer_nodes.iterrows()):
            node_id = node["id"]
            node_ids.append(node_id)
            
            # Add small random offset to y and z
            offset_y = np.random.normal(0, grid_spacing * 0.1)
            offset_z = np.random.normal(0, grid_spacing * 0.1)
            
            # Transform grid coordinates to cylinder coordinates
            # x and y form the circle, z remains vertical
            final_y = y + offset_y
            final_z = z + offset_z
            
            # Calculate position on the cylinder
            x = cylinder_radius * np.cos(layer_angle)
            y = cylinder_radius * np.sin(layer_angle)
            
            # Add the vertical grid offset to the cylinder position
            final_x = x - final_y * np.sin(layer_angle)  # Offset perpendicular to radius
            final_y = y + final_y * np.cos(layer_angle)  # Offset perpendicular to radius
            
            node_positions.append([final_x, final_y, final_z])
            
            node_colors.append(kind_colors[layer])
            node_clusters[node_id] = layer
            node_origins[node_id] = layer

    # Convert to numpy array
    node_positions = np.array(node_positions)

    # Create node ID to position index mapping
    node_to_pos_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    # Create edge pairs and colors
    link_pairs = []
    link_colors = []

    for _, edge in tqdm(edgelist.iterrows(), desc="Processing edges"):
        if edge["source"] in node_to_pos_idx and edge["target"] in node_to_pos_idx:
            source_idx = node_to_pos_idx[edge["source"]]
            target_idx = node_to_pos_idx[edge["target"]]
            link_pairs.append([source_idx, target_idx])
            link_colors.append(edge_colors[edge["metaedge"]])

    # Convert to numpy arrays
    link_pairs = np.array(link_pairs, dtype=np.int32)
    link_colors = np.array(link_colors)

    logger.info("Network building complete:")
    logger.info(f"- {len(node_positions)} nodes")
    logger.info(f"- {len(link_pairs)} edges")
    logger.info(f"- {len(layers)} layers")
    logger.info(f"- Z range: {node_positions[:,2].min():.2f} to {node_positions[:,2].max():.2f}")
    logger.info(f"- Max layer diameter: {layer_diameter:.2f}")
    logger.info(f"- Nodes per layer: {nodes_per_layer}")

    return (
        node_positions,
        link_pairs,
        link_colors,
        node_ids,
        layers,
        node_clusters,
        layers,
        node_colors,
        node_origins,
        layers,
        edge_colors,
    ) 