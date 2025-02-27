# Implementation for Layer Graph panel (test2) 
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QCheckBox, QComboBox, QLabel
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from charts.interlayer_graph import create_interlayer_graph
from .base_panel import BaseStatsPanel

class LayerGraphPanel(BaseStatsPanel):
    """Panel for Layer Graph visualization"""
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Add controls
        controls_layout = QHBoxLayout()
        
        # Add checkbox to enable/disable visualization
        self.enable_checkbox = QCheckBox("Enable")
        self.enable_checkbox.setChecked(False)  # Disabled by default
        self.enable_checkbox.stateChanged.connect(self.on_state_changed)
        controls_layout.addWidget(self.enable_checkbox)
        
        # Add layout algorithm dropdown
        controls_layout.addWidget(QLabel("Layout Algorithm:"))
        self.layout_algorithm_dropdown = QComboBox()
        self.layout_algorithm_dropdown.addItems([
            "hierarchical_betweeness_centrality", "connection_centric", "weighted_spring", 
            "spring", "circular", "kamada_kawai", "planar", "spiral", "force_atlas2", 
            "radial", "weighted_spectral", "pagerank_centric"
        ])
        self.layout_algorithm_dropdown.currentTextChanged.connect(self.on_layout_algorithm_changed)
        controls_layout.addWidget(self.layout_algorithm_dropdown)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Create figure for Layer Graph
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
            layer_connections, layers, medium_font, large_font, visible_layer_indices, layer_colors = self._current_data
            
            # Redraw with current layout algorithm
            create_interlayer_graph(
                self.ax, layer_connections, layers,
                medium_font, large_font, visible_layer_indices, layer_colors,
                layout_algorithm=self.layout_algorithm_dropdown.currentText()
            )
            
            self.figure.tight_layout(pad=1.0)
            self.canvas.draw()
        elif not state:
            # Clear the visualization when disabled
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            self.ax.text(0.5, 0.5, "Layer graph disabled", 
                        ha='center', va='center', fontsize=12)
            self.canvas.draw()
    
    def on_layout_algorithm_changed(self, algorithm):
        """Handle layout algorithm change"""
        if self._current_data and self.enable_checkbox.isChecked():
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            
            # Unpack stored data
            layer_connections, layers, medium_font, large_font, visible_layer_indices, layer_colors = self._current_data
            
            # Redraw with new layout algorithm
            create_interlayer_graph(
                self.ax, layer_connections, layers,
                medium_font, large_font, visible_layer_indices, layer_colors,
                layout_algorithm=algorithm
            )
            
            self.figure.tight_layout(pad=1.0)
            self.canvas.draw()
    
    def update_stats(self, data_manager):
        """Update the Layer Graph with current data"""
        # Clear figure
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        
        # Get data from manager
        layers = data_manager.layers
        visible_layer_indices = data_manager.visible_layers
        layer_colors = data_manager.layer_colors
        
        # Get layer connections from data manager
        layer_connections = data_manager.get_layer_connections()
        
        # Define font sizes
        medium_font = {'fontsize': 7}
        large_font = {'fontsize': 9}
        
        # Store current data for later use
        self._current_data = (layer_connections, layers, medium_font, large_font, visible_layer_indices, layer_colors)
        
        # Only create visualization if enabled
        if self.enable_checkbox.isChecked():
            create_interlayer_graph(
                self.ax, layer_connections, layers,
                medium_font, large_font, visible_layer_indices, layer_colors,
                layout_algorithm=self.layout_algorithm_dropdown.currentText()
            )
        else:
            self.ax.text(0.5, 0.5, "Layer graph disabled", 
                        ha='center', va='center', fontsize=12)
        
        # Apply tight layout
        self.figure.tight_layout(pad=1.0)
        
        # Draw canvas
        self.canvas.draw() 