import os
import logging
import networkx as nx
from data.network_builder import build_multilayer_network
from utils.calc_layout import get_layout_position


def calculate_layer_layout(G, layer_nodes, layout_algorithm="kamada_kawai"):
    """Calculate layout for a specific layer's nodes"""
    # Create subgraph with only nodes in this layer
    subgraph = G.subgraph(layer_nodes)
    # Calculate spring layout for this layer
    return get_layout_position(subgraph, layout_algorithm=layout_algorithm)


def get_available_diseases(data_dir):
    """Get list of available disease datasets from the data directory"""
    diseases = set()
    for filename in os.listdir(data_dir):
        if filename.endswith("_Multiplex_Network.tsv"):
            disease_name = filename.replace("_Multiplex_Network.tsv", "")
            diseases.add(disease_name)
    return sorted(diseases)


def load_disease_data(
    disease_name, 
    data_dir, 
    prefilter_layers=None, 
    prefilter_clusters=None, 
    prefilter_max_nodes=None,
    use_ml_layout=False, 
    layout_algorithm="kamada_kawai", 
    z_offset=0.5
):
    """
    Load the selected disease dataset
    
    Parameters:
    -----------
    disease_name : str
        Name of disease dataset to load
    data_dir : str
        Path to data directory
    prefilter_layers : list, optional
        List of layer names to include (for prefiltering large networks)
    prefilter_clusters : list, optional
        List of cluster names to include (for prefiltering large networks)
    prefilter_max_nodes : int, optional
        Maximum number of nodes to include (for prefiltering large networks)
    use_ml_layout : bool
        If True, calculate separate layout for each layer
    layout_algorithm : str
        The layout algorithm to use for node positioning
    z_offset : float
        The vertical offset between network layers (default: 0.5)
        
    Returns:
    --------
    dict
        Dictionary containing network data with keys:
        - node_positions: NumPy array of node positions
        - link_pairs: NumPy array of edge connections
        - link_colors: List of edge colors
        - node_ids: List of node IDs
        - layers: List of layer names
        - node_clusters: List of node clusters
        - unique_clusters: List of unique cluster values
        - node_colors: List of node colors (optional)
        - node_origins: List of node origins (optional)
        - unique_origins: List of unique origin values (optional)
        - layer_colors: Dictionary mapping layer names to colors (optional)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading disease: {disease_name}")

    try:
        # Construct file paths
        edge_list_path = os.path.join(data_dir, f"{disease_name}_Multiplex_Network.tsv")
        node_metadata_path = os.path.join(
            data_dir, f"{disease_name}_Multiplex_Metadata.tsv"
        )

        logger.info(f"Looking for files at: {edge_list_path}")

        # Check if files exist
        if not os.path.exists(edge_list_path):
            logger.error(f"Edge list file not found: {edge_list_path}")
            return None
            
        if not os.path.exists(node_metadata_path):
            logger.error(f"Node metadata file not found: {node_metadata_path}")
            return None

        # Build network with layout preference and prefiltering options
        prefilter_options = {}
        if prefilter_layers:
            prefilter_options['layers'] = prefilter_layers
        if prefilter_clusters:
            prefilter_options['clusters'] = prefilter_clusters
        if prefilter_max_nodes:
            prefilter_options['max_nodes'] = prefilter_max_nodes
            
        # Build the network and get the result as a tuple
        if prefilter_options:
            logger.info(f"Building network with prefiltering: {prefilter_options}")
            result = build_multilayer_network(
                edge_list_path, 
                node_metadata_path, 
                add_interlayer_edges=True,
                use_ml_layout=use_ml_layout,
                layout_algorithm=layout_algorithm,
                z_offset=z_offset,
                prefilter_options=prefilter_options
            )
        else:
            result = build_multilayer_network(
                edge_list_path, 
                node_metadata_path, 
                add_interlayer_edges=True,
                use_ml_layout=use_ml_layout,
                layout_algorithm=layout_algorithm,
                z_offset=z_offset
            )
            
        # Check if result is None
        if result is None:
            logger.error("Network builder returned None")
            return None
            
        # Convert tuple to dictionary
        # The order of elements in the tuple is defined by the build_multilayer_network function
        try:
            # Unpack the tuple
            (
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
            ) = result
            
            # Create a dictionary with named keys
            return {
                "node_positions": node_positions,
                "link_pairs": link_pairs,
                "link_colors": link_colors,
                "node_ids": node_ids,
                "layers": layers,
                "node_clusters": node_clusters,
                "unique_clusters": unique_clusters,
                "node_colors": node_colors,
                "node_origins": node_origins,
                "unique_origins": unique_origins,
                "layer_colors": layer_colors,
            }
        except ValueError as e:
            # Handle case where tuple doesn't have expected number of elements
            logger.error(f"Error unpacking network data: {e}")
            logger.error(f"Expected 11 elements, got {len(result) if isinstance(result, tuple) else 'not a tuple'}")
            return None
            
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        return None
    except PermissionError as e:
        logger.error(f"Permission error accessing files: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading disease data: {e}")
        import traceback
        traceback.print_exc()
        return None
