import pandas as pd
import numpy as np
import networkx as nx
from vispy import scene, app
from vispy.scene.visuals import Markers, Line
import logging
from datetime import datetime
from tqdm import tqdm
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QApplication, QHBoxLayout, QLabel, QGroupBox, QComboBox
from PyQt5.QtCore import Qt
import random
import os

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def generate_random_colors(num_colors):
    """
    Generate a list of random bright HTML colors.
    
    Parameters:
    - num_colors: Number of colors to generate
    
    Returns:
    - colors: List of random bright HTML colors
    """
    colors = []
    for _ in range(num_colors):
        r = random.randint(128, 255)
        g = random.randint(128, 255)
        b = random.randint(128, 255)
        color = '#%02x%02x%02x' % (r, g, b)
        colors.append(color)
    return colors

def build_multilayer_network(edge_list_path, node_metadata_path, add_interlayer_edges=True):
    """
    Build a multilayer network from edge list and node metadata files.
    Following the logic from multiCore_DataDiVR.ipynb.
    
    Returns numpy arrays ready for visualization.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Building multilayer network from {edge_list_path}")
    
    # Read the edge list
    edgelist_with_att = pd.read_table(edge_list_path, sep='\t', header=0, index_col=False)
    # Remove .tsv string from the column names
    edgelist_with_att.columns = edgelist_with_att.columns.str.replace('.tsv', '')
    
    # Read the node metadata
    node_metadata = pd.read_table(node_metadata_path, sep='\t', header=0, index_col=False)
    
    # Rename columns in node metadata
    node_metadata.rename(columns={'Cluster': 'cluster', 'Color': 'nodecolor'}, inplace=True)
    
    # Get all unique base nodes
    unique_base_nodes = pd.unique(
        np.concatenate([
            [node.split('_')[0] for node in edgelist_with_att['V1']],
            [node.split('_')[0] for node in edgelist_with_att['V2']]
        ])
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
    
    # Add edges from the first layer for layout purposesthe inter layer edges only go
    first_layer = layers[0]
    for _, row in edgelist_with_att[edgelist_with_att[first_layer] == 1].iterrows():
        source = row['V1'].split('_')[0]
        target = row['V2'].split('_')[0]
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
            node_positions.append([x, y, z/20])

            # Get node metadata
            if base_node in node_metadata['Node'].values:
                node_data = node_metadata[node_metadata['Node'] == base_node].iloc[0]
                color = node_data['nodecolor']
                cluster = node_data['cluster']
                origin = node_data.get('Origin', 'Unknown')  # Get origin if available
                node_colors.append(color)
                node_clusters[node_id] = cluster
                node_origins[node_id] = origin  # Store the origin
            else:
                node_colors.append('#CCCCCC')  # Default gray
                node_clusters[node_id] = 'Unknown'
                node_origins[node_id] = 'Unknown'  # Default origin

    # Convert to numpy arrays
    node_positions = np.array(node_positions)

    # Create link pairs and colors
    link_pairs = []
    link_colors = []

    # Generate random colors for each layer
    layer_colors = {layer: color for layer, color in zip(layers, generate_random_colors(len(layers)))}

    # Add intra-layer edges
    for z, layer in enumerate(tqdm(layers, desc="Processing layers")):
        layer_edges = edgelist_with_att[edgelist_with_att[layer] == 1][['V1', 'V2']].values
        for edge in layer_edges:
            source_base = edge[0].split('_')[0]
            target_base = edge[1].split('_')[0]
            
            source_idx = z * len(unique_base_nodes) + base_node_to_index[source_base]
            target_idx = z * len(unique_base_nodes) + base_node_to_index[target_base]
            
            link_pairs.append([source_idx, target_idx])
            link_colors.append(layer_colors[layer])
    
    # Add inter-layer edges if requested
    if add_interlayer_edges:
        # First, determine which nodes exist in which layers
        node_layer_presence = {}
        for z, layer in enumerate(layers):
            layer_edges = edgelist_with_att[edgelist_with_att[layer] == 1][['V1', 'V2']].values
            for edge in layer_edges:
                source_base = edge[0].split('_')[0]
                target_base = edge[1].split('_')[0]
                
                if source_base not in node_layer_presence:
                    node_layer_presence[source_base] = set()
                if target_base not in node_layer_presence:
                    node_layer_presence[target_base] = set()
                    
                node_layer_presence[source_base].add(z)
                node_layer_presence[target_base].add(z)
        
        # Now connect nodes between layers where they exist
        for base_node, layer_indices in tqdm(node_layer_presence.items(), desc="Adding inter-layer edges"):
            if base_node in base_node_to_index and len(layer_indices) > 1:
                # Sort the layer indices to ensure we process them in order
                layer_indices = sorted(layer_indices)
                
                # Connect each layer to all other layers where this node exists
                for i, z1 in enumerate(layer_indices):
                    for z2 in layer_indices[i+1:]:
                        source_idx = z1 * len(unique_base_nodes) + base_node_to_index[base_node]
                        target_idx = z2 * len(unique_base_nodes) + base_node_to_index[base_node]
                        link_pairs.append([source_idx, target_idx])
                        link_colors.append('#07885f')  # Green for inter-layer edges
    
    # Convert to numpy arrays
    link_pairs = np.array(link_pairs)
    
    # Get unique clusters and origins
    unique_clusters = list(set(node_clusters.values()))
    unique_origins = list(set(node_origins.values()))  # Get unique origins
    
    logger.info(f"Created multilayer network with {len(node_positions)} nodes and {len(link_pairs)} edges")
    logger.info(f"Found {len(unique_clusters)} clusters: {unique_clusters}")
    logger.info(f"Found {len(unique_origins)} origins: {unique_origins}")
    
    return node_positions, link_pairs, link_colors, node_ids, layers, node_clusters, unique_clusters, node_colors, node_origins, unique_origins

class MultilayerNetworkViz(QWidget):
    def __init__(self, node_positions=None, link_pairs=None, link_colors=None, node_ids=None, 
                 layers=None, node_clusters=None, unique_clusters=None, data_dir=None):
        super().__init__()
        logger = logging.getLogger(__name__)
        logger.info("Initializing visualization...")
        
        # Store the data directory
        self.data_dir = data_dir
        
        # Create layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)  # Minimal margins
        main_layout.setSpacing(5)  # Minimal spacing
        self.setLayout(main_layout)
        
        # Create dropdown for disease selection
        if self.data_dir:
            disease_layout = QHBoxLayout()
            disease_layout.setContentsMargins(0, 0, 0, 0)  # No margins
            disease_layout.setSpacing(5)
            disease_layout.addWidget(QLabel("Select Disease:"))
            self.disease_combo = self.create_disease_dropdown()
            disease_layout.addWidget(self.disease_combo)
            disease_layout.addStretch(1)  # Push controls to the left
            disease_widget = QWidget()
            disease_widget.setLayout(disease_layout)
            main_layout.addWidget(disease_widget)
        
        # Create the main content area (horizontal layout for controls and canvas)
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        content_layout.setSpacing(5)
        content_widget = QWidget()
        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget, 1)  # Give it stretch factor
        
        # Create left panel for controls
        control_panel = QVBoxLayout()
        control_panel.setContentsMargins(0, 0, 0, 0)  # No margins
        control_panel.setSpacing(5)
        control_widget = QWidget()
        control_widget.setLayout(control_panel)
        control_widget.setFixedWidth(180)  # Make control panel narrower
        content_layout.addWidget(control_widget)
        
        # Create canvas
        logger.info("Creating canvas...")
        self.canvas = scene.SceneCanvas(keys='interactive', size=(800, 600))
        content_layout.addWidget(self.canvas.native, 1)  # Give it stretch factor
        
        # Create view
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.fov = 45
        self.view.camera.distance = 8
        
        # Initialize visualization elements
        self.scatter = Markers()
        self.view.add(self.scatter)
        
        self.lines = Line(pos=np.zeros((0, 3)), color=np.zeros((0, 4)),
                         connect='segments', width=2)
        self.view.add(self.lines)
        
        # Create containers for controls
        self.layer_group = QGroupBox("Layers")
        self.layer_group.setFlat(True)  # Make it less prominent
        layer_layout = QVBoxLayout()
        layer_layout.setContentsMargins(5, 5, 5, 5)  # Minimal margins
        layer_layout.setSpacing(2)  # Minimal spacing between checkboxes
        self.layer_group.setLayout(layer_layout)
        control_panel.addWidget(self.layer_group)
        
        self.cluster_group = QGroupBox("Clusters")
        self.cluster_group.setFlat(True)  # Make it less prominent
        cluster_layout = QVBoxLayout()
        cluster_layout.setContentsMargins(5, 5, 5, 5)  # Minimal margins
        cluster_layout.setSpacing(2)  # Minimal spacing between checkboxes
        self.cluster_group.setLayout(cluster_layout)
        control_panel.addWidget(self.cluster_group)
        
        # Add origin group
        self.origin_group = QGroupBox("Origins")
        self.origin_group.setFlat(True)
        origin_layout = QVBoxLayout()
        origin_layout.setContentsMargins(5, 5, 5, 5)
        origin_layout.setSpacing(2)
        self.origin_group.setLayout(origin_layout)
        control_panel.addWidget(self.origin_group)
        
        # Add stretch at the bottom to push controls to the top
        control_panel.addStretch(1)
        
        # Store references to layouts for later updates
        self.layer_layout = layer_layout
        self.cluster_layout = cluster_layout
        self.origin_layout = origin_layout
        
        # Initialize empty control lists
        self.layer_checkboxes = []
        self.cluster_checkboxes = {}
        self.origin_checkboxes = {}
        
        # If data is provided, load it
        if node_positions is not None:
            self.load_data(node_positions, link_pairs, link_colors, node_ids, layers, node_clusters, unique_clusters)
        elif self.data_dir and self.disease_combo.count() > 0:
            # Load the first disease by default
            self.load_disease(self.disease_combo.currentText())
        
        logger.info("Visualization setup complete")
        self.setWindowTitle("Multilayer Network Visualization")
        self.resize(1000, 800)
        self.show()
    
    def create_disease_dropdown(self):
        """Create dropdown menu with available disease datasets"""
        combo = QComboBox()
        
        # Get list of available diseases from the data directory
        diseases = set()
        for filename in os.listdir(self.data_dir):
            if filename.endswith("_Multiplex_Network.tsv"):
                disease_name = filename.replace("_Multiplex_Network.tsv", "")
                diseases.add(disease_name)
        
        # Add diseases to dropdown
        for disease in sorted(diseases):
            combo.addItem(disease)
        
        # Connect signal
        combo.currentTextChanged.connect(self.load_disease)
        
        return combo
    
    def load_disease(self, disease_name):
        """Load the selected disease dataset"""
        logger = logging.getLogger(__name__)
        logger.info(f"Loading disease: {disease_name}")
        
        edge_list_path = os.path.join(self.data_dir, f"{disease_name}_Multiplex_Network.tsv")
        node_metadata_path = os.path.join(self.data_dir, f"{disease_name}_Multiplex_Metadata.tsv")
        
        if not os.path.exists(edge_list_path) or not os.path.exists(node_metadata_path):
            logger.error(f"Data files for {disease_name} not found")
            return
        
        # Build network
        node_positions, link_pairs, link_colors, node_ids, layers, node_clusters, unique_clusters, node_colors, node_origins, unique_origins = build_multilayer_network(
            edge_list_path, node_metadata_path, add_interlayer_edges=True
        )
        
        # Load the data into the visualization
        self.load_data(node_positions, link_pairs, link_colors, node_ids, layers, node_clusters, unique_clusters, node_colors, node_origins, unique_origins)
    
    def load_data(self, node_positions, link_pairs, link_colors, node_ids, layers, node_clusters, unique_clusters, 
                 node_colors=None, node_origins=None, unique_origins=None):
        """Load network data into the visualization"""
        logger = logging.getLogger(__name__)
        
        # Store the data
        self.node_positions = node_positions
        self.link_pairs = link_pairs
        self.link_colors = link_colors
        self.node_ids = node_ids
        self.layers = layers
        self.node_clusters = node_clusters
        self.unique_clusters = unique_clusters
        self.node_origins = node_origins or {}
        self.unique_origins = unique_origins or []
        
        # Convert hex colors to RGBA for vispy
        self.node_colors_rgba = np.ones((len(node_ids), 4))  # Default white
        
        if node_colors:
            # Use provided node colors from metadata
            for i, color_hex in enumerate(node_colors):
                if color_hex.startswith('#'):
                    color = color_hex[1:]
                    r = int(color[0:2], 16) / 255.0
                    g = int(color[2:4], 16) / 255.0
                    b = int(color[4:6], 16) / 255.0
                    a = 1.0
                    if len(color) >= 8:  # If alpha is included
                        a = int(color[6:8], 16) / 255.0
                    self.node_colors_rgba[i] = [r, g, b, a]
                else:
                    # Default color if hex format is invalid
                    self.node_colors_rgba[i] = [0.7, 0.7, 0.7, 1.0]
        else:
            # Use cluster-based coloring as fallback
            for i, node_id in enumerate(node_ids):
                cluster = node_clusters[node_id]
                # Use cluster to determine color
                hue = hash(cluster) % 360 / 360.0
                self.node_colors_rgba[i, 0] = hue
                self.node_colors_rgba[i, 1] = 0.8
                self.node_colors_rgba[i, 2] = 0.8
        
        # Convert link colors from hex to RGBA
        self.link_colors_rgba = []
        for color in link_colors:
            if color.startswith('#'):
                color = color[1:]
                r = int(color[0:2], 16) / 255.0
                g = int(color[2:4], 16) / 255.0
                b = int(color[4:6], 16) / 255.0
                self.link_colors_rgba.append([r, g, b, 0.8])
            else:
                self.link_colors_rgba.append([0.5, 0.5, 0.5, 0.8])  # Default gray
        
        # Update the visualization
        self.update_controls()
        self.update_visibility()
    
    def update_controls(self):
        """Update the layer, cluster, and origin controls based on loaded data"""
        logger = logging.getLogger(__name__)
        logger.info("Updating controls...")
        
        # Clear existing controls
        self.clear_layout(self.layer_layout)
        self.clear_layout(self.cluster_layout)
        self.clear_layout(self.origin_layout)
        self.layer_checkboxes = []
        self.cluster_checkboxes = {}
        self.origin_checkboxes = {}
        
        # Create layer controls
        for i, layer in enumerate(self.layers):
            cb = QCheckBox(f"{layer}")
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_visibility)
            self.layer_layout.addWidget(cb)
            self.layer_checkboxes.append(cb)
        
        # Create cluster controls
        for cluster in self.unique_clusters:
            cb = QCheckBox(f"{cluster}")
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_visibility)
            self.cluster_layout.addWidget(cb)
            self.cluster_checkboxes[cluster] = cb
        
        # Create origin controls
        for origin in self.unique_origins:
            cb = QCheckBox(f"{origin}")
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_visibility)
            self.origin_layout.addWidget(cb)
            self.origin_checkboxes[origin] = cb
    
    def clear_layout(self, layout):
        """Clear all widgets from a layout"""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
    
    def update_lines(self):
        logger = logging.getLogger(__name__)
        logger.debug(f"Updating {len(self.link_pairs)} edge positions...")
        pos = np.zeros((len(self.link_pairs)*2, 3))
        colors = np.zeros((len(self.link_pairs)*2, 4))
        
        for i, (start_idx, end_idx) in enumerate(self.link_pairs):
            pos[i*2] = self.node_positions[start_idx]
            pos[i*2 + 1] = self.node_positions[end_idx]
            colors[i*2] = self.link_colors_rgba[i]
            colors[i*2 + 1] = self.link_colors_rgba[i]
        
        self.lines.set_data(pos=pos, color=colors)
    
    def update_visibility(self):
        logger = logging.getLogger(__name__)
        # Get visible layers
        visible_layers = [i for i, cb in enumerate(self.layer_checkboxes) if cb.isChecked()]
        
        # Get visible clusters
        visible_clusters = [cluster for cluster, cb in self.cluster_checkboxes.items() if cb.isChecked()]
        
        # Get visible origins
        visible_origins = [origin for origin, cb in self.origin_checkboxes.items() if cb.isChecked()]
        
        # Calculate nodes per layer
        nodes_per_layer = len(self.node_positions) // len(self.layers)
        
        # Create node mask based on layers, clusters, and origins
        node_mask = np.zeros(len(self.node_positions), dtype=bool)
        
        for i, node_id in enumerate(self.node_ids):
            # Check layer visibility
            layer_idx = i // nodes_per_layer
            if layer_idx not in visible_layers:
                continue
                
            # Check cluster visibility
            cluster = self.node_clusters[node_id]
            if cluster not in visible_clusters:
                continue
            
            # Check origin visibility
            origin = self.node_origins.get(node_id, 'Unknown')
            if origin not in visible_origins:
                continue
                
            # Node passes all filters
            node_mask[i] = True
        
        # Update node visibility
        visible_nodes = self.node_positions[node_mask]
        visible_colors = self.node_colors_rgba[node_mask]
        
        # Handle case when no nodes are visible
        if len(visible_nodes) == 0:
            self.scatter.set_data(np.zeros((1, 3)), edge_color='black',
                                face_color=np.array([[0, 0, 0, 0]]), size=0)
            self.lines.set_data(pos=np.zeros((0, 3)),
                              color=np.zeros((0, 4)),
                              width=2)
            return
        
        # Update scatter plot
        self.scatter.set_data(visible_nodes, edge_color='black', 
                            face_color=visible_colors, size=3)
        
        # Update edge visibility
        edge_mask = np.zeros(len(self.link_pairs), dtype=bool)
        for i, (start_idx, end_idx) in enumerate(self.link_pairs):
            if node_mask[start_idx] and node_mask[end_idx]:
                edge_mask[i] = True
        
        # Get visible edges
        visible_edges = self.link_pairs[edge_mask]
        visible_colors = [self.link_colors_rgba[i] for i, mask in enumerate(edge_mask) if mask]
        
        # Create line positions for visible edges
        if len(visible_edges) > 0:
            pos = np.zeros((len(visible_edges)*2, 3))
            colors = np.zeros((len(visible_edges)*2, 4))
            
            for i, (start_idx, end_idx) in enumerate(visible_edges):
                pos[i*2] = self.node_positions[start_idx]
                pos[i*2 + 1] = self.node_positions[end_idx]
                
                # Check if this is a horizontal (intra-layer) edge
                # If nodes are in the same layer, their z-coordinates will be the same
                is_horizontal = abs(self.node_positions[start_idx][2] - self.node_positions[end_idx][2]) < 0.001
                
                # Copy the color but adjust opacity for horizontal edges
                edge_color = visible_colors[i].copy()
                if is_horizontal:
                    edge_color[3] = 0.2  # Set opacity to 20% for horizontal edges
                
                colors[i*2] = edge_color
                colors[i*2 + 1] = edge_color
                
            self.lines.set_data(pos=pos, color=colors, width=2)
        else:
            self.lines.set_data(pos=np.zeros((0, 3)), color=np.zeros((0, 4)), width=2)

def main():
    app_instance = QApplication([])
    logger = setup_logging()
    start_time = datetime.now()
    
    # Set data directory
    data_dir = "Multiplex_DataDiVR/Multiplex_Net_Files"
    
    # Create visualization with just the data directory
    main_widget = MultilayerNetworkViz(data_dir=data_dir)
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    logger.info(f"Total setup time: {total_duration:.2f} seconds")
    
    logger.info("Starting viz loop")
    app_instance.exec_()

if __name__ == "__main__":
    main() 