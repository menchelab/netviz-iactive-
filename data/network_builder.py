import pandas as pd
import numpy as np
import networkx as nx
import logging
from tqdm import tqdm
from utils.color_utils import generate_distinct_colors, generate_colors_for_dark_background
from utils.calc_layout import get_layout_position


def calculate_layer_layout(G, layer_nodes, layout_algorithm="kamada_kawai"):
    """Calculate layout for a specific layer's nodes"""
    subgraph = G.subgraph(layer_nodes)
    return get_layout_position(subgraph, layout_algorithm=layout_algorithm)


def build_multilayer_network(
    edge_list_path,
    node_metadata_path,
    add_interlayer_edges=True,
    use_ml_layout=False,
    layout_algorithm="kamada_kawai",
    z_offset=0.5,
):
    """
    Build a multilayer network from edge list and node metadata files.
    Following the logic from multiCore_DataDiVR.ipynb.

    Parameters:
    -----------
    edge_list_path : str
        Path to edge list file
    node_metadata_path : str
        Path to node metadata file
    add_interlayer_edges : bool
        Whether to add edges between layers
    use_ml_layout : bool
        Whether to use multilayer layout
    layout_algorithm : str
        The layout algorithm to use
    z_offset : float
        The vertical offset between network layers (default: 0.5)
        If 0, will auto-scale so that total height is 2.0

    Returns numpy arrays ready for visualization.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Building multilayer network from {edge_list_path}")

    # Read the edge list
    edgelist_with_att = pd.read_table(
        edge_list_path, sep="\t", header=0, index_col=False
    )
    # Remove .tsv string from the column names
    edgelist_with_att.columns = edgelist_with_att.columns.str.replace(".tsv", "")

    # Read the node metadata first
    node_metadata = pd.read_table(
        node_metadata_path, sep="\t", header=0, index_col=False
    )

    # Rename columns in node metadata
    node_metadata.rename(
        columns={"Cluster": "cluster", "Color": "nodecolor"}, inplace=True
    )

    # Get all unique base nodes
    unique_base_nodes = pd.unique(
        np.concatenate(
            [
                [node.split("_")[0] for node in edgelist_with_att["V1"]],
                [node.split("_")[0] for node in edgelist_with_att["V2"]],
            ]
        )
    )
    logger.info(f"Found {len(unique_base_nodes)} unique base nodes")

    # Get all layers (columns starting from the third column)
    layers = edgelist_with_att.columns[2:].tolist()
    logger.info(f"Found {len(layers)} layers: {layers}")

    # Create a mapping from node ID to index
    base_node_to_index = {node: idx for idx, node in enumerate(unique_base_nodes)}

    # Create node positions array
    # First, create a base graph for layout calculation
    G_base = nx.Graph()

    # Add nodes with cluster information
    for base_node in unique_base_nodes:
        cluster = "Unknown"
        if base_node in node_metadata["Node"].values:
            node_data = node_metadata[node_metadata["Node"] == base_node].iloc[0]
            cluster = node_data["cluster"]
        G_base.add_node(base_node, cluster=cluster)

    # Add edges from the first layer for layout purposes
    first_layer = layers[0]
    for _, row in edgelist_with_att.iterrows():
        source = row["V1"].split("_")[0]
        target = row["V2"].split("_")[0]
        G_base.add_edge(source, target)

    # Create node-to-layer mapping before layout calculation
    node_layers = {}
    for layer in layers:
        for base_node in unique_base_nodes:
            node_id = f"{base_node}_{layer}"
            node_layers[node_id] = layer

    # Calculate layout
    logger.info("Calculating layout...")
    # Create an aggregated graph with all nodes and edges from all layers
    G_aggregated = nx.Graph()

    # Add all nodes
    for base_node in unique_base_nodes:
        cluster = "Unknown"
        if base_node in node_metadata["Node"].values:
            node_data = node_metadata[node_metadata["Node"] == base_node].iloc[0]
            cluster = node_data["cluster"]
        G_aggregated.add_node(base_node, cluster=cluster)

    # Add all edges from all layers
    for layer in layers:
        layer_edges = edgelist_with_att[edgelist_with_att[layer] == 1][
            ["V1", "V2"]
        ].values
        for edge in layer_edges:
            source = edge[0].split("_")[0]
            target = edge[1].split("_")[0]
            G_aggregated.add_edge(source, target)

    # Calculate layout once for the aggregated network
    aggregated_layout = get_layout_position(
        G_aggregated, layout_algorithm=layout_algorithm
    )

    # Use the same x,y coordinates for each node across all layers
    positions = {}

    # If using multilayer layout, apply scaling to layers
    if use_ml_layout:
        # Find center layer index
        center_layer_idx = len(layers) // 2

        # Calculate center point of the layout
        all_x = [pos[0] for pos in aggregated_layout.values()]
        all_y = [pos[1] for pos in aggregated_layout.values()]
        center_x = sum(all_x) / len(all_x) if all_x else 0
        center_y = sum(all_y) / len(all_y) if all_y else 0

        logger.info(
            f"Using multilayer layout with center at ({center_x:.2f}, {center_y:.2f})"
        )

        # Apply scaling to each layer
        for layer_idx, layer in enumerate(layers):
            # Calculate distance from center layer
            distance = abs(layer_idx - center_layer_idx)
            # Scale factor increases by 10% for each layer away from center
            scale_factor = 1.0 + (distance * 0.3)

            for node in unique_base_nodes:
                node_id = f"{node}_{layer}"
                x, y = aggregated_layout.get(node, (0, 0))

                # Scale coordinates while preserving center
                scaled_x = center_x + (x - center_x) * scale_factor
                scaled_y = center_y + (y - center_y) * scale_factor

                positions[node_id] = (scaled_x, scaled_y)

            if distance > 0:
                logger.info(
                    f"Layer {layer} (distance {distance}): scaled by {scale_factor:.2f}x"
                )
    else:
        # Original behavior - use same coordinates for all layers
        for layer in layers:
            for node in unique_base_nodes:
                node_id = f"{node}_{layer}"
                positions[node_id] = aggregated_layout.get(node, (0, 0))

    # Calculate network width if auto z_offset (0)
    if z_offset == 0 and positions:
        # Get all x,y positions from the first layer to calculate width
        first_layer_positions = [
            positions[f"{node}_{first_layer}"]
            for node in unique_base_nodes
            if f"{node}_{first_layer}" in positions
        ]

        if first_layer_positions:
            # Calculate the width (max x - min x) and height (max y - min y)
            x_values = [pos[0] for pos in first_layer_positions]
            y_values = [pos[1] for pos in first_layer_positions]

            width = max(x_values) - min(x_values) if x_values else 1.0
            height = max(y_values) - min(y_values) if y_values else 1.0

            # Use the larger of width or height as the network extent
            network_extent = max(width, height)

            # Set z_offset so that total height is 2 * network_extent
            # This makes the vertical spacing proportional to the network width
            z_offset = (
                (1.3 * network_extent) / (len(layers) - 1)
                if len(layers) > 1
                else network_extent
            )
            logger.info(
                f"Auto z_offset: {z_offset:.2f} (based on network extent: {network_extent:.2f})"
            )
        else:
            # Fallback if no positions found
            z_offset = 1.3 / (len(layers) - 1) if len(layers) > 1 else 0.5
            logger.info(f"Auto z_offset fallback: {z_offset:.2f}")

    # Create node positions for all layers
    node_positions = []
    node_ids = []
    node_colors = []
    node_clusters = {}
    node_origins = {}  # Add dictionary to store node origins

    for z, layer in enumerate(layers):
        for base_node in unique_base_nodes:
            node_id = f"{base_node}_{layer}"
            node_ids.append(node_id)

            # Get position from layout
            x, y = positions[node_id]
            node_positions.append([x, y, z * z_offset])  # Use configurable z_offset

            # Get node metadata
            if base_node in node_metadata["Node"].values:
                node_data = node_metadata[node_metadata["Node"] == base_node].iloc[0]
                color = node_data["nodecolor"]
                cluster = node_data["cluster"]
                origin = node_data.get("Origin", "Unknown")  # Get origin if available
                node_colors.append(color)
                node_clusters[node_id] = cluster
                node_origins[node_id] = origin  # Store the origin
            else:
                node_colors.append("#CCCCCC")  # Default gray
                node_clusters[node_id] = "Unknown"
                node_origins[node_id] = "Unknown"  # Default origin

    # Convert to numpy arrays
    node_positions = np.array(node_positions)

    # Create link pairs and colors
    link_pairs = []
    link_colors = []

    # Generate distinct colors for each layer using our palette
    layer_colors = {
        layer: color
        for layer, color in zip(layers, generate_colors_for_dark_background(len(layers)))
    }

    # Add intra-layer edges
    for z, layer in enumerate(tqdm(layers, desc="Processing layers")):
        layer_edges = edgelist_with_att[edgelist_with_att[layer] == 1][
            ["V1", "V2"]
        ].values
        for edge in layer_edges:
            source_base = edge[0].split("_")[0]
            target_base = edge[1].split("_")[0]

            source_idx = z * len(unique_base_nodes) + base_node_to_index[source_base]
            target_idx = z * len(unique_base_nodes) + base_node_to_index[target_base]

            link_pairs.append([source_idx, target_idx])
            link_colors.append(layer_colors[layer])

    # Add inter-layer edges if requested
    if add_interlayer_edges:
        # First, determine which nodes exist in which layers
        node_layer_presence = {}
        for z, layer in enumerate(layers):
            layer_edges = edgelist_with_att[edgelist_with_att[layer] == 1][
                ["V1", "V2"]
            ].values
            for edge in layer_edges:
                source_base = edge[0].split("_")[0]
                target_base = edge[1].split("_")[0]

                if source_base not in node_layer_presence:
                    node_layer_presence[source_base] = set()
                if target_base not in node_layer_presence:
                    node_layer_presence[target_base] = set()

                node_layer_presence[source_base].add(z)
                node_layer_presence[target_base].add(z)

        # Now connect nodes between layers where they exist
        for base_node, layer_indices in tqdm(
            node_layer_presence.items(), desc="Adding inter-layer edges"
        ):
            if base_node in base_node_to_index and len(layer_indices) > 1:
                # Sort the layer indices to ensure we process them in order
                layer_indices = sorted(layer_indices)

                # Connect each layer to all other layers where this node exists
                for i, z1 in enumerate(layer_indices):
                    for z2 in layer_indices[i + 1 :]:
                        source_idx = (
                            z1 * len(unique_base_nodes) + base_node_to_index[base_node]
                        )
                        target_idx = (
                            z2 * len(unique_base_nodes) + base_node_to_index[base_node]
                        )
                        link_pairs.append([source_idx, target_idx])
                        # Use the color of the source layer
                        source_layer = layers[z1]
                        link_colors.append(layer_colors[source_layer])

    # Convert to numpy arrays
    link_pairs = np.array(link_pairs)

    # Get unique clusters and origins
    unique_clusters = list(set(node_clusters.values()))
    unique_origins = list(set(node_origins.values()))  # Get unique origins

    logger.info(
        f"Created multilayer network with {len(node_positions)} nodes and {len(link_pairs)} edges"
    )
    logger.info(f"Found {len(unique_clusters)} clusters: {unique_clusters}")
    logger.info(f"Found {len(unique_origins)} origins: {unique_origins}")

    return (
        node_positions,  # numpy array we already created
        link_pairs,  # numpy array we already created
        link_colors,  # list we already created
        node_ids,  # list we already created
        layers,  # list we already have
        node_clusters,  # dict we already created
        unique_clusters,  # list we already created
        node_colors,  # list we already created
        node_origins,  # dict we already created
        unique_origins,  # list we already created
        layer_colors,  # dict we already created
    )
