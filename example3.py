import pandas as pd
import numpy as np
import networkx as nx
from vispy import scene, app
from vispy.scene.visuals import Markers, Line
import logging
from datetime import datetime
from tqdm import tqdm
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QApplication, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def load_tsv_data(file_path):
    logger = logging.getLogger(__name__)
    start_time = datetime.now()

    logger.info(f"Starting to load data from {file_path}")
    # Read TSV file
    df = pd.read_csv(file_path, sep='\t')
    
    # Read metadata file
    metadata_path = file_path.replace('_Network.tsv', '_Metadata.tsv')
    metadata_df = pd.read_csv(metadata_path, sep='\t')
    logger.info(f"Loaded metadata file with shape: {metadata_df.shape}")
    
    # Create node metadata dictionaries
    node_colors = dict(zip(metadata_df['Node'], metadata_df['Color'].apply(lambda x: [int(x[i:i+2], 16)/255 for i in (1,3,5,7)])))
    node_clusters = dict(zip(metadata_df['Node'], metadata_df['Cluster']))
    node_origins = dict(zip(metadata_df['Node'], metadata_df['Origin']))
    
    # Get all unique nodes
    unique_nodes = pd.unique(df[['V1', 'V2']].values.ravel())
    logger.info(f"Found {len(unique_nodes)} unique nodes")
    node_id_to_index = {node: idx for idx, node in enumerate(unique_nodes)}

    # Get layer names (excluding V1 and V2 columns)
    original_layers = df.columns[2:].tolist()
    # Create cleaned layer names (for display)
    display_layers = [col.replace('.tsv', '') for col in original_layers]
    logger.info(f"Found {len(display_layers)} layers: {display_layers}")

    # Create a base graph for layout calculation using only first layer
    logger.info("Creating base graph for layout calculation...")
    G = nx.Graph()
    G.add_nodes_from(unique_nodes)
    
    # Add edges from first layer only for layout purposes
    first_layer = original_layers[0]
    layer_edges = df[df[first_layer] == 1][['V1', 'V2']].values
    G.add_edges_from(layer_edges)
    
    logger.info(f"Base graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Calculate layout using base graph
    logger.info("Calculating Kamada-Kawai layout...")
    try:
        layout = nx.kamada_kawai_layout(G)
        logger.info("Kamada-Kawai layout calculation completed successfully")
    except Exception as e:
        logger.error(f"Error in Kamada-Kawai layout: {e}")
        logger.info("Falling back to spring layout...")
        layout = nx.spring_layout(G)

    # Create node positions array with z-coordinates
    logger.info("Creating 3D node positions...")
    node_positions = []
    for node in unique_nodes:
        x, y = layout[node]
        for z, layer in enumerate(display_layers):
            node_positions.append([x, y, z/30])

    node_positions = np.array(node_positions)
    logger.info(f"Created position array with shape: {node_positions.shape}")

    # Create link pairs and colors for each layer
    logger.info("Processing edges for each layer...")
    link_pairs = []
    link_colors = []

    for z, (orig_layer, display_layer) in enumerate(tqdm(zip(original_layers, display_layers), desc="Processing layers")):
        # Only get edges where the layer value is 1
        layer_edges = df[df[orig_layer] == 1][['V1', 'V2']].values
        logger.info(f"Layer {display_layer}: found {len(layer_edges)} edges")
        
        # Add edges only for connections that exist in this layer
        for v1, v2 in layer_edges:
            source_idx = node_id_to_index[v1] + (z * len(unique_nodes))
            target_idx = node_id_to_index[v2] + (z * len(unique_nodes))
            link_pairs.append([source_idx, target_idx])
            link_colors.append([0, 0, 1, 0.8])

    link_pairs = np.array(link_pairs)
    logger.info(f"Created edge array with shape: {link_pairs.shape}")

    # Create repeated node_ids for each layer
    node_ids = []
    for z in range(len(display_layers)):
        node_ids.extend(unique_nodes)
    logger.info(f"Created node_ids array with length: {len(node_ids)}")

    # Create node colors array
    node_colors_array = []
    for node in node_ids:
        node_colors_array.append(node_colors.get(node, [1, 1, 1, 1]))  # White as default
    node_colors_array = np.array(node_colors_array)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("\nFinal Network statistics:")
    logger.info(f"  - Number of nodes per layer: {len(unique_nodes)}")
    logger.info(f"  - Number of layers: {len(display_layers)}")
    logger.info(f"  - Total number of links: {len(link_pairs)}")
    logger.info(f"  - Memory usage:")
    logger.info(f"    * Node positions: {node_positions.nbytes / 1024:.2f} KB")
    logger.info(f"    * Link pairs: {link_pairs.nbytes / 1024:.2f} KB")
    logger.info(f"Data loading completed in {duration:.2f} seconds")

    # Return additional metadata
    return (node_positions, link_pairs, link_colors, node_ids, display_layers, 
            node_colors_array, node_clusters, node_origins)

class netviz(QWidget):
    def __init__(self, node_positions, link_pairs, link_colors, node_ids, layer_names, 
                 node_colors, node_clusters, node_origins):
        super().__init__()
        logger = logging.getLogger(__name__)
        logger.info("Initializing visualization...")

        # Store the data
        self.node_positions = node_positions
        self.link_pairs = link_pairs
        self.link_colors = link_colors
        self.node_ids = node_ids
        self.layer_names = layer_names
        self.node_colors = node_colors
        self.node_clusters = node_clusters
        self.node_origins = node_origins

        logger.info("Setting up Qt layout...")
        # Create main horizontal layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Create left control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setMaximumWidth(300)  # Limit control panel width
        main_layout.addWidget(control_panel)
        
        # Create right visualization panel
        viz_panel = QWidget()
        viz_layout = QVBoxLayout()
        viz_panel.setLayout(viz_layout)
        main_layout.addWidget(viz_panel)
        
        logger.info("Creating canvas...")
        # Create canvas
        canvas = scene.SceneCanvas(keys='interactive', size=(800, 600))
        viz_layout.addWidget(canvas.native)
        
        # Create view
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'
        view.camera.fov = 45
        view.camera.distance = 8
        
        logger.info("Adding node markers...")
        # Add scatter plot for nodes
        self.scatter = Markers()
        self.scatter.set_data(self.node_positions, edge_color='black', 
                            face_color=self.node_colors, size=3)
        view.add(self.scatter)
        
        logger.info("Adding edge lines...")
        # Add lines for edges
        self.lines = Line(pos=np.zeros((len(self.link_pairs)*2, 3)), 
                         color=np.tile([(0, 0, 1, 0.8)], (len(self.link_pairs)*2, 1)),
                         connect='segments', width=2)
        self.update_lines()
        view.add(self.lines)
        
        logger.info("Creating control sections...")
        
        # Layer controls
        layer_group = QWidget()
        layer_layout = QVBoxLayout()
        layer_layout.addWidget(QLabel("Layers:"))
        layer_group.setLayout(layer_layout)
        control_layout.addWidget(layer_group)
        
        # Cluster controls
        cluster_group = QWidget()
        cluster_layout = QVBoxLayout()
        cluster_layout.addWidget(QLabel("Clusters:"))
        cluster_group.setLayout(cluster_layout)
        control_layout.addWidget(cluster_group)
        
        # Origin controls
        origin_group = QWidget()
        origin_layout = QVBoxLayout()
        origin_layout.addWidget(QLabel("Origins:"))
        origin_group.setLayout(origin_layout)
        control_layout.addWidget(origin_group)

        # Create checkboxes
        self.layer_checkboxes = []
        self.cluster_checkboxes = {}
        self.origin_checkboxes = {}

        # Layer checkboxes
        for layer_name in self.layer_names:
            cb = QCheckBox(layer_name)
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_visibility)
            layer_layout.addWidget(cb)
            self.layer_checkboxes.append(cb)

        # Cluster checkboxes
        unique_clusters = sorted(set(self.node_clusters.values()))
        for cluster in unique_clusters:
            cb = QCheckBox(f"Cluster {cluster}")
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_visibility)
            cluster_layout.addWidget(cb)
            self.cluster_checkboxes[cluster] = cb

        # Origin checkboxes
        unique_origins = sorted(set(self.node_origins.values()))
        for origin in unique_origins:
            cb = QCheckBox(origin)
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_visibility)
            origin_layout.addWidget(cb)
            self.origin_checkboxes[origin] = cb

        # Add stretch to push checkboxes to the top
        control_layout.addStretch()
        
        logger.info("Visualization setup complete")
        self.show()

    def update_lines(self):
        logger = logging.getLogger(__name__)
        logger.debug(f"Updating {len(self.link_pairs)} edge positions...")
        pos = np.zeros((len(self.link_pairs)*2, 3))
        for i, (start_idx, end_idx) in enumerate(self.link_pairs):
            pos[i*2] = self.node_positions[start_idx]
            pos[i*2 + 1] = self.node_positions[end_idx]
        self.lines.set_data(pos=pos)

    def update_visibility(self):
        logger = logging.getLogger(__name__)
        # Get visible layers, clusters, and origins
        visible_layers = [i for i, cb in enumerate(self.layer_checkboxes) if cb.isChecked()]
        visible_clusters = [cluster for cluster, cb in self.cluster_checkboxes.items() if cb.isChecked()]
        visible_origins = [origin for origin, cb in self.origin_checkboxes.items() if cb.isChecked()]

        # Create node mask based on all criteria
        node_mask = np.zeros(len(self.node_positions), dtype=bool)
        for i, node_id in enumerate(self.node_ids):
            layer_idx = i // (len(self.node_ids) // len(self.layer_names))
            is_visible = (layer_idx in visible_layers and
                         self.node_clusters[node_id] in visible_clusters and
                         self.node_origins[node_id] in visible_origins)
            node_mask[i] = is_visible

        # Get visible nodes and their colors
        visible_nodes = self.node_positions[node_mask]
        visible_colors = self.node_colors[node_mask]

        # Handle case when no nodes are visible
        if len(visible_nodes) == 0:
            self.scatter.set_data(np.zeros((1, 3)), edge_color='black',
                                face_color=np.array([[0, 0, 0, 0]]), size=0)
            self.lines.set_data(pos=np.zeros((0, 3)),
                              color=np.zeros((0, 4)),
                              width=2)
            return

        # Update scatter plot with visible nodes and their colors
        self.scatter.set_data(visible_nodes, edge_color='black',
                             face_color=visible_colors, size=3)

        # Update edge visibility and colors
        edge_mask = np.zeros(len(self.link_pairs), dtype=bool)
        edge_colors = []
        
        for i, (start_idx, end_idx) in enumerate(self.link_pairs):
            if node_mask[start_idx] and node_mask[end_idx]:
                start_z = self.node_positions[start_idx, 2] * 30
                end_z = self.node_positions[end_idx, 2] * 30
                
                if int(start_z) in visible_layers and int(end_z) in visible_layers:
                    edge_mask[i] = True
                    z_diff = abs(end_z - start_z)
                    if z_diff == 0:
                        color = [0, 0, 1, 0.8]
                    else:
                        intensity = min(z_diff / len(self.layer_checkboxes), 1.0)
                        color = [intensity, 1 - intensity, 0, 0.8]
                    edge_colors.extend([color, color])

        visible_edges = self.link_pairs[edge_mask]
        
        if len(visible_edges) > 0:
            pos = np.zeros((len(visible_edges)*2, 3))
            for i, (start_idx, end_idx) in enumerate(visible_edges):
                pos[i*2] = self.node_positions[start_idx]
                pos[i*2 + 1] = self.node_positions[end_idx]
            self.lines.set_data(pos=pos,
                              color=np.array(edge_colors),
                              width=2)
        else:
            self.lines.set_data(pos=np.zeros((0, 3)),
                              color=np.zeros((0, 4)),
                              width=2)

def main():
    app_instance = QApplication([])
    logger = setup_logging()
    start_time = datetime.now()

    file_path = "Multiplex_DataDiVR/Multiplex_Net_Files/Other Myopathies_Multiplex_Network.tsv"
    (node_positions, link_pairs, link_colors, node_ids, layers, 
     node_colors, node_clusters, node_origins) = load_tsv_data(file_path)
    
    main_widget = netviz(node_positions, link_pairs, link_colors, node_ids, layers,
                        node_colors, node_clusters, node_origins)

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    logger.info(f"Total setup time: {total_duration:.2f} seconds")

    logger.info("Starting viz loop")
    app_instance.exec_()

if __name__ == "__main__":
    main() 