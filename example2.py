import pandas as pd
import numpy as np
import networkx as nx
from vispy import scene, app
from vispy.scene.visuals import Markers, Line
import logging
from datetime import datetime
from tqdm import tqdm
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QApplication, QHBoxLayout
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
    logger.info(f"Loaded TSV file with shape: {df.shape}")

    # Get all unique nodes
    unique_nodes = pd.unique(df[['V1', 'V2']].values.ravel())
    logger.info(f"Found {len(unique_nodes)} unique nodes")
    node_id_to_index = {node: idx for idx, node in enumerate(unique_nodes)}

    # Get layer names (excluding V1 and V2 columns)
    layers = df.columns[2:].tolist()
    logger.info(f"Found {len(layers)} layers: {layers}")

    # Create a base graph for layout
    logger.info("Creating base graph for layout calculation...")
    G = nx.Graph()
    G.add_nodes_from(unique_nodes)
    G.add_edges_from(df[['V1', 'V2']].values)
    logger.info(f"Base graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

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
        for z, layer in enumerate(layers):
            node_positions.append([x, y, z])

    node_positions = np.array(node_positions)
    logger.info(f"Created position array with shape: {node_positions.shape}")

    # Create link pairs and colors for each layer
    logger.info("Processing edges for each layer...")
    link_pairs = []
    link_colors = []

    for z, layer in enumerate(tqdm(layers, desc="Processing layers")):
        layer_edges = df[df[layer] == 1][['V1', 'V2']].values
        logger.info(f"Layer {layer}: found {len(layer_edges)} edges")
        for edge in layer_edges:
            source_idx = node_id_to_index[edge[0]] + (z * len(unique_nodes))
            target_idx = node_id_to_index[edge[1]] + (z * len(unique_nodes))
            link_pairs.append([source_idx, target_idx])
            link_colors.append('#1f77b4')

    link_pairs = np.array(link_pairs)
    logger.info(f"Created edge array with shape: {link_pairs.shape}")

    # Create repeated node_ids for each layer
    node_ids = []
    for z in range(len(layers)):
        node_ids.extend(unique_nodes)
    logger.info(f"Created node_ids array with length: {len(node_ids)}")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("\nFinal Network statistics:")
    logger.info(f"  - Number of nodes per layer: {len(unique_nodes)}")
    logger.info(f"  - Number of layers: {len(layers)}")
    logger.info(f"  - Total number of links: {len(link_pairs)}")
    logger.info(f"  - Memory usage:")
    logger.info(f"    * Node positions: {node_positions.nbytes / 1024:.2f} KB")
    logger.info(f"    * Link pairs: {link_pairs.nbytes / 1024:.2f} KB")
    logger.info(f"Data loading completed in {duration:.2f} seconds")

    return node_positions, link_pairs, link_colors, node_ids

class netviz(QWidget):
    def __init__(self, node_positions, link_pairs, link_colors, node_ids):
        super().__init__()
        logger = logging.getLogger(__name__)
        logger.info("Initializing visualization...")

        # Store the data
        self.node_positions = node_positions
        self.link_pairs = link_pairs
        self.link_colors = link_colors
        self.node_ids = node_ids

        logger.info("Setting up Qt layout...")
        # Create layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        logger.info("Creating canvas...")
        # Create canvas
        canvas = scene.SceneCanvas(keys='interactive', size=(800, 600))
        layout.addWidget(canvas.native)
        
        # Create view
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'
        view.camera.fov = 45
        view.camera.distance = 8
        
        logger.info("Adding node markers...")
        # Add scatter plot for nodes
        self.scatter = Markers()
        self.scatter.set_data(self.node_positions, edge_color='black', 
                            face_color=(1, 1, 1, .9), size=3)
        view.add(self.scatter)
        
        logger.info("Adding edge lines...")
        # Add lines for edges
        self.lines = Line(pos=np.zeros((len(self.link_pairs)*2, 3)), 
                         color=np.tile([(0, 0, 1, 0.8)], (len(self.link_pairs)*2, 1)),
                         connect='segments', width=2)
        self.update_lines()
        view.add(self.lines)
        
        logger.info("Creating layer controls...")
        # Create checkboxes layout
        checkbox_layout = QHBoxLayout()
        layout.addLayout(checkbox_layout)
        
        # Add layer visibility controls
        self.layer_checkboxes = []
        num_layers = len(np.unique(self.node_positions[:, 2]))
        for i in range(num_layers):
            cb = QCheckBox(f"Layer {i}")
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_visibility)
            checkbox_layout.addWidget(cb)
            self.layer_checkboxes.append(cb)
        
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
        # Get visible layers
        visible_layers = [i for i, cb in enumerate(self.layer_checkboxes) 
                         if cb.isChecked()]
        logger.debug(f"Updating visibility for layers: {visible_layers}")
        
        # Update node visibility
        node_mask = np.zeros(len(self.node_positions), dtype=bool)
        for layer in visible_layers:
            node_mask |= (self.node_positions[:, 2] == layer)
        
        # Update edge visibility
        edge_mask = np.zeros(len(self.link_pairs), dtype=bool)
        for i, (start_idx, end_idx) in enumerate(self.link_pairs):
            start_layer = self.node_positions[start_idx, 2]
            end_layer = self.node_positions[end_idx, 2]
            if start_layer in visible_layers and end_layer in visible_layers:
                edge_mask[i] = True
        
        # Apply visibility masks
        visible_nodes = self.node_positions[node_mask]
        visible_edges = self.link_pairs[edge_mask]
        visible_colors = np.array(self.link_colors)[edge_mask]
        
        # Update visualizations
        self.scatter.set_data(visible_nodes, edge_color='black', 
                            face_color=(1, 1, 1, .9), size=3)

        pos = np.zeros((len(visible_edges)*2, 3))
        for i, (start_idx, end_idx) in enumerate(visible_edges):
            pos[i*2] = self.node_positions[start_idx]
            pos[i*2 + 1] = self.node_positions[end_idx]
        self.lines.set_data(pos=pos, 
                           color=np.tile([(0, 0, 1, 0.8)], (len(visible_edges)*2, 1)),
                           width=2)

def main():
    app_instance = QApplication([])
    logger = setup_logging()
    start_time = datetime.now()

    file_path = "Multiplex_DataDiVR/Multiplex_Net_Files/Other Myopathies_Multiplex_Network.tsv"
    node_positions, link_pairs, link_colors, node_ids = load_tsv_data(file_path)
    main_widget = netviz(node_positions, link_pairs, link_colors, node_ids)

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    logger.info(f"Total setup time: {total_duration:.2f} seconds")

    logger.info("Starting viz loop")
    app_instance.exec_()

if __name__ == "__main__":
    main() 