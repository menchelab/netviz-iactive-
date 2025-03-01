import os
import logging
import networkx as nx
from data.network_builder import build_multilayer_network
from utils.calc_layout import get_layout_position


def calculate_layer_layout(G, layer_nodes):
    """Calculate layout for a specific layer's nodes"""
    # Create subgraph with only nodes in this layer
    subgraph = G.subgraph(layer_nodes)
    # Calculate spring layout for this layer
    return get_layout_position(subgraph, layout_algorithm="kamada_kawai")


def get_available_diseases(data_dir):
    """Get list of available disease datasets from the data directory"""
    diseases = set()
    for filename in os.listdir(data_dir):
        if filename.endswith("_Multiplex_Network.tsv"):
            disease_name = filename.replace("_Multiplex_Network.tsv", "")
            diseases.add(disease_name)
    return sorted(diseases)


def load_disease_data(data_dir, disease_name, use_ml_layout=False):
    """
    Load the selected disease dataset
    
    Parameters:
    -----------
    data_dir : str
        Path to data directory
    disease_name : str
        Name of disease dataset to load
    use_ml_layout : bool
        If True, calculate separate layout for each layer
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading disease: {disease_name}")

    edge_list_path = os.path.join(data_dir, f"{disease_name}_Multiplex_Network.tsv")
    node_metadata_path = os.path.join(
        data_dir, f"{disease_name}_Multiplex_Metadata.tsv"
    )

    print(edge_list_path)

    if not os.path.exists(edge_list_path) or not os.path.exists(node_metadata_path):
        logger.error(f"Data files for {disease_name} not found")
        return None

    # Build network with layout preference
    return build_multilayer_network(
        edge_list_path, 
        node_metadata_path, 
        add_interlayer_edges=True,
        use_ml_layout=use_ml_layout
    )
