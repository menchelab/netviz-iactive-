import pandas as pd
import numpy as np
import networkx as nx
import logging
from tqdm import tqdm
from utils.color_utils import generate_distinct_colors


def build_multilayer_network(
    edge_list_path, node_metadata_path, add_interlayer_edges=True
):
    """
    Build a multilayer network from edge list and node metadata files.
    Following the logic from multiCore_DataDiVR.ipynb.

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

    # Read the node metadata
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
    G_base.add_nodes_from(unique_base_nodes)

    # Add edges from the first layer for layout purposes
    first_layer = layers[0]
    for _, row in edgelist_with_att.iterrows():
        source = row["V1"].split("_")[0]
        target = row["V2"].split("_")[0]
        G_base.add_edge(source, target)
    logger.info("Calculating layout...")
    try:
        layout = nx.kamada_kawai_layout(G_base)
    except:
        logger.warning("Kamada-Kawai layout failed, falling back to spring layout")
        layout = nx.spring_layout(G_base)

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
            x, y = layout[base_node]
            node_positions.append([x, y, z / 20])

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
        for layer, color in zip(layers, generate_distinct_colors(len(layers)))
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
        node_positions,
        link_pairs,
        link_colors,
        node_ids,
        layers,
        node_clusters,
        unique_clusters,
        node_colors,
        node_origins,
        unique_origins,
        layer_colors,
    )
