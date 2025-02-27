# Implementation for Sankey panel 
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QCheckBox
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from charts.cluster_sankey import create_cluster_sankey_chart
from .base_panel import BaseStatsPanel

class SankeyPanel(BaseStatsPanel):
    """Panel for Sankey diagram visualization"""
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Add checkbox to enable/disable Sankey diagram
        controls_layout = QHBoxLayout()
        self.enable_checkbox = QCheckBox("Enable")
        self.enable_checkbox.setChecked(False)  # Disabled by default
        self.enable_checkbox.stateChanged.connect(self.on_state_changed)
        controls_layout.addWidget(self.enable_checkbox)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Create figure for Sankey diagram
        self.figure = Figure(figsize=(8, 10), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Initialize subplot
        self.ax = self.figure.add_subplot(111)
        
        # Store current data
        self._current_data = None
    
    def on_state_changed(self, state):
        """Handle enable/disable state change"""
        if self._current_data and state:
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            
            # Unpack stored data
            visible_links, node_ids, node_clusters, nodes_per_layer, layers, medium_font, large_font, visible_layer_indices = self._current_data
            
            # Redraw Sankey diagram
            create_cluster_sankey_chart(
                self.ax, visible_links, node_ids, node_clusters, 
                nodes_per_layer, layers, medium_font, large_font, visible_layer_indices
            )
            
            self.figure.tight_layout(pad=1.0)
            self.canvas.draw()
        elif not state:
            # Clear the diagram when disabled
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            self.ax.text(0.5, 0.5, "Sankey diagram disabled", 
                        ha='center', va='center', fontsize=12)
            self.canvas.draw()
    
    def update_stats(self, data_manager):
        """Update the Sankey diagram with current data"""
        # Clear figure
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        
        # Get data from manager
        node_positions = data_manager.node_positions
        link_pairs = data_manager.link_pairs
        node_ids = data_manager.node_ids
        layers = data_manager.layers
        node_clusters = data_manager.node_clusters
        node_mask = data_manager.current_node_mask
        edge_mask = data_manager.current_edge_mask
        visible_layer_indices = data_manager.visible_layers
        
        # Get visible edges
        visible_edges = [i for i, mask in enumerate(edge_mask) if mask]
        visible_links = [link_pairs[i] for i in visible_edges]
        
        # Calculate nodes per layer
        nodes_per_layer = len(node_positions) // len(layers)
        
        # Define font sizes
        medium_font = {'fontsize': 7}
        large_font = {'fontsize': 9}
        
        # Store current data for later use
        self._current_data = (visible_links, node_ids, node_clusters, 
                             nodes_per_layer, layers, medium_font, large_font, visible_layer_indices)
        
        # Only create Sankey diagram if enabled
        if self.enable_checkbox.isChecked():
            create_cluster_sankey_chart(
                self.ax, visible_links, node_ids, node_clusters, 
                nodes_per_layer, layers, medium_font, large_font, visible_layer_indices
            )
        else:
            self.ax.text(0.5, 0.5, "Sankey diagram disabled", 
                        ha='center', va='center', fontsize=12)
        
        # Apply tight layout
        self.figure.tight_layout(pad=1.0)
        
        # Draw canvas
        self.canvas.draw() 