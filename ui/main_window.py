from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
import logging
import numpy as np
import os

from ui.network_canvas import NetworkCanvas
from ui.stats_panel import NetworkStatsPanel
from ui.control_panel import ControlPanel
from data.data_loader import get_available_diseases, load_disease_data
from utils.color_utils import hex_to_rgba

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
        main_layout.setContentsMargins(5, 5, 5, 5) 
        main_layout.setSpacing(5)  # Minimal spacing
        self.setLayout(main_layout)

        # Create dropdown for disease selection
        if self.data_dir:
            disease_layout = QHBoxLayout()
            disease_layout.setContentsMargins(0, 0, 0, 0)  # No margins
            disease_layout.setSpacing(5)
            disease_layout.addWidget(QLabel("dataset:"))
            self.disease_combo = self.create_disease_dropdown()
            disease_layout.addWidget(self.disease_combo)
            disease_layout.addStretch(1)  # Push controls to the left
            disease_widget = QWidget()
            disease_widget.setLayout(disease_layout)
            main_layout.addWidget(disease_widget)

        # Create the main content area (horizontal layout for controls, canvas, and stats)
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(5)
        content_widget = QWidget()
        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget, 1)

        # Create left panel for controls
        self.control_panel = ControlPanel()
        self.control_panel.setFixedWidth(180)
        content_layout.addWidget(self.control_panel)

        # Create canvas
        logger.info("Creating canvas...")
        self.network_canvas = NetworkCanvas()
        content_layout.addWidget(self.network_canvas.canvas.native, 1)  # Give it stretch factor

        # Create stats panel
        self.stats_panel = NetworkStatsPanel()
        self.stats_panel.setFixedWidth(600)
        content_layout.addWidget(self.stats_panel)

        # Initialize data attributes
        self.node_positions = None
        self.link_pairs = None
        self.link_colors = None
        self.node_ids = None
        self.layers = None
        self.node_clusters = None
        self.unique_clusters = None
        self.node_origins = None
        self.unique_origins = None

        # If data is provided, load it
        if node_positions is not None:
            self.load_data(node_positions, link_pairs, link_colors, node_ids, layers, node_clusters, unique_clusters)
        elif self.data_dir and self.disease_combo.count() > 0:
            # Load the first dataset by default
            self.load_disease(self.disease_combo.currentText())

        logger.info("Visualization setup complete")
        self.setWindowTitle("Multilayer Network Visualization")
        self.resize(1200, 768) 
        self.show()

    def create_disease_dropdown(self):
        combo = QComboBox()

        diseases = get_available_diseases(self.data_dir)

        for disease in diseases:
            combo.addItem(disease)

        combo.currentTextChanged.connect(self.load_disease)

        return combo

    def load_disease(self, disease_name):
        logger = logging.getLogger(__name__)
        logger.info(f"Loading dataset: {disease_name}")

        data = load_disease_data(self.data_dir, disease_name)

        if data:
            # Unpack the data including layer_colors
            node_positions, link_pairs, link_colors, node_ids, layers, node_clusters, unique_clusters, node_colors, node_origins, unique_origins, layer_colors = data

            # Load the data into the visualization
            self.load_data(node_positions, link_pairs, link_colors, node_ids, layers, node_clusters, unique_clusters, 
                          node_colors, node_origins, unique_origins, layer_colors)

    def load_data(self, node_positions, link_pairs, link_colors, node_ids, layers, node_clusters, unique_clusters, 
                 node_colors=None, node_origins=None, unique_origins=None, layer_colors=None):
        """Load network data into the visualization"""

        self.node_positions = node_positions
        self.link_pairs = link_pairs
        self.link_colors = link_colors
        self.node_ids = node_ids
        self.layers = layers
        self.node_clusters = node_clusters
        self.unique_clusters = unique_clusters
        self.node_origins = node_origins or {}
        self.unique_origins = unique_origins or []
        self.layer_colors = layer_colors or {}

        self.network_canvas.load_data(node_positions, link_pairs, link_colors, node_colors)

        # Update the controls with layer colors
        self.control_panel.update_controls(
            self.layers, self.unique_clusters, self.unique_origins, 
            self.update_visibility, self.layer_colors
        )

        self.update_visibility()

    def update_visibility(self):
        """Update the visibility of nodes and edges based on control panel settings"""
        logger = logging.getLogger(__name__)

        # Get visible layers, clusters, and origins from control panel
        visible_layers = self.control_panel.get_visible_layers()
        visible_clusters = self.control_panel.get_visible_clusters()
        visible_origins = self.control_panel.get_visible_origins()
        show_intralayer = self.control_panel.show_intralayer_edges()
        show_nodes = self.control_panel.show_nodes()  # Get the show_nodes setting

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

        # Create edge mask based on node visibility
        edge_mask = np.zeros(len(self.link_pairs), dtype=bool)
        for i, (start_idx, end_idx) in enumerate(self.link_pairs):
            if node_mask[start_idx] and node_mask[end_idx]:
                edge_mask[i] = True

        # Update network canvas with layer information
        self.network_canvas.visible_layers = visible_layers
        self.network_canvas.layer_names = {i: layer for i, layer in enumerate(self.layers)}
        self.network_canvas.nodes_per_layer = nodes_per_layer
        self.network_canvas.node_mask = node_mask

        # Update network canvas with visibility settings
        self.network_canvas.update_visibility(node_mask, edge_mask, show_intralayer, show_nodes)

        # Update statistics panel with visible layer indices and layer colors
        self.stats_panel.update_stats(
            self.node_positions, self.link_pairs, self.node_ids,
            self.layers, self.node_clusters, node_mask, edge_mask,
            visible_layers, self.layer_colors
        )