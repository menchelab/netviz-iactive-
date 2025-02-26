import os
import logging
from data.network_builder import build_multilayer_network

def get_available_diseases(data_dir):
    """Get list of available disease datasets from the data directory"""
    diseases = set()
    for filename in os.listdir(data_dir):
        if filename.endswith("_Multiplex_Network.tsv"):
            disease_name = filename.replace("_Multiplex_Network.tsv", "")
            diseases.add(disease_name)
    return sorted(diseases)

def load_disease_data(data_dir, disease_name):
    """Load the selected disease dataset"""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading disease: {disease_name}")
    
    edge_list_path = os.path.join(data_dir, f"{disease_name}_Multiplex_Network.tsv")
    node_metadata_path = os.path.join(data_dir, f"{disease_name}_Multiplex_Metadata.tsv")
    
    print(edge_list_path)

    if not os.path.exists(edge_list_path) or not os.path.exists(node_metadata_path):
        logger.error(f"Data files for {disease_name} not found")
        return None
    
    # Build network
    return build_multilayer_network(
        edge_list_path, node_metadata_path, add_interlayer_edges=True
    ) 