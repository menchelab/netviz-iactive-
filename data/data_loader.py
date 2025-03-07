import os
import logging
import networkx as nx
from data.network_builder import build_multilayer_network
from data.network_builder_alt import build_multilayer_network as build_hetionet_network
from utils.calc_layout import get_layout_position


def calculate_layer_layout(G, layer_nodes, layout_algorithm="kamada_kawai"):
    """Calculate layout for a specific layer's nodes"""
    # Create subgraph with only nodes in this layer
    subgraph = G.subgraph(layer_nodes)
    # Calculate spring layout for this layer
    return get_layout_position(subgraph, layout_algorithm=layout_algorithm)


def get_available_datasets(data_dir):
    """Get list of available datasets from the data directory"""
    datasets = []
    
    # Check for Hetionet files
    if (os.path.exists(os.path.join(data_dir, "hetionet-v1.0-nodes.tsv")) and 
        os.path.exists(os.path.join(data_dir, "hetionet-v1.0-edges.sif"))):
        datasets.append("Hetionet")
    
    # Check for disease datasets (original format)
    for filename in os.listdir(data_dir):
        if filename.endswith("_Multiplex_Network.tsv"):
            disease_name = filename.replace("_Multiplex_Network.tsv", "")
            datasets.append(disease_name)
    
    return sorted(datasets)


def load_dataset(data_dir, dataset_name, use_ml_layout=False, layout_algorithm="kamada_kawai", z_offset=0.5):
    """Load the selected dataset"""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading dataset: {dataset_name}")

    if dataset_name == "Hetionet":
        # Load Hetionet format
        node_file = os.path.join(data_dir, "hetionet-v1.0-nodes.tsv")
        edge_file = os.path.join(data_dir, "hetionet-v1.0-edges.sif")
        
        if not os.path.exists(node_file) or not os.path.exists(edge_file):
            logger.error("Hetionet files not found")
            return None
            
        return build_hetionet_network(
            node_file,
            edge_file,
            add_interlayer_edges=True,
            use_ml_layout=use_ml_layout,
            layout_algorithm=layout_algorithm,
            z_offset=z_offset
        )
    else:
        # Load original disease format
        edge_list_path = os.path.join(data_dir, f"{dataset_name}_Multiplex_Network.tsv")
        node_metadata_path = os.path.join(data_dir, f"{dataset_name}_Multiplex_Metadata.tsv")

        if not os.path.exists(edge_list_path) or not os.path.exists(node_metadata_path):
            logger.error(f"Data files for {dataset_name} not found")
            return None

        return build_multilayer_network(
            edge_list_path,
            node_metadata_path,
            add_interlayer_edges=True,
            use_ml_layout=use_ml_layout,
            layout_algorithm=layout_algorithm,
            z_offset=z_offset
        )
