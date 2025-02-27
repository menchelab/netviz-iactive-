# Implementation for Layer Influence panel 
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QCheckBox, QComboBox, QLabel
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.gridspec as gridspec

from charts.layer_influence import create_layer_influence_chart
from .base_panel import BaseStatsPanel

class LayerInfluencePanel(BaseStatsPanel):
    """Panel for Layer Influence visualization"""
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Add controls
        controls_layout = QHBoxLayout()
        
        # Add checkbox to enable/disable visualization
        self.enable_checkbox = QCheckBox("Enable Layer Influence")
        self.enable_checkbox.setChecked(False)  # Disabled by default
        self.enable_checkbox.stateChanged.connect(self.on_state_changed)
        controls_layout.addWidget(self.enable_checkbox)
        
        # Add influence metric dropdown
        controls_layout.addWidget(QLabel("Influence Metric:"))
        self.influence_metric_dropdown = QComboBox()
        self.influence_metric_dropdown.addItems([
            "PageRank", "Eigenvector Centrality", "Combined Influence Index"
        ])
        self.influence_metric_dropdown.currentTextChanged.connect(self.on_metric_changed)
        controls_layout.addWidget(self.influence_metric_dropdown)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Create figure for Layer Influence
        self.figure = Figure(figsize=(8, 10), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Initialize subplots with GridSpec for custom layout
        # First row takes 30% of height, second row takes 70%
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 7])
        self.bar_ax = self.figure.add_subplot(gs[0])  # Top row (30%)
        self.network_ax = self.figure.add_subplot(gs[1])  # Bottom row (70%)
        
        # Store current data
        self._current_data = None
    
    def on_state_changed(self, state):
        """Handle enable/disable state change"""
        if self._current_data and state:
            self.figure.clear()
            
            # Recreate the GridSpec layout
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 7])
            self.bar_ax = self.figure.add_subplot(gs[0])
            self.network_ax = self.figure.add_subplot(gs[1])
            
            # Unpack stored data
            layer_connections, layers, medium_font, large_font, visible_layer_indices, layer_colors = self._current_data
            
            # Redraw with current metric
            create_layer_influence_chart(
                self.bar_ax, self.network_ax,
                layer_connections, layers, medium_font, large_font,
                visible_layer_indices, layer_colors,
                metric=self.influence_metric_dropdown.currentText()
            )
            
            self.figure.tight_layout(pad=1.0)
            self.canvas.draw()
        elif not state:
            # Clear the visualization when disabled
            self.figure.clear()
            
            # Recreate the GridSpec layout
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 7])
            self.bar_ax = self.figure.add_subplot(gs[0])
            self.network_ax = self.figure.add_subplot(gs[1])
            
            self.bar_ax.text(0.5, 0.5, "Layer influence disabled", 
                           ha='center', va='center', fontsize=12)
            self.bar_ax.axis('off')
            
            self.network_ax.text(0.5, 0.5, "Layer influence disabled", 
                               ha='center', va='center', fontsize=12)
            self.network_ax.axis('off')
            
            self.canvas.draw()
    
    def on_metric_changed(self, metric):
        """Handle influence metric change"""
        if self._current_data and self.enable_checkbox.isChecked():
            self.figure.clear()
            
            # Recreate the GridSpec layout
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 7])
            self.bar_ax = self.figure.add_subplot(gs[0])
            self.network_ax = self.figure.add_subplot(gs[1])
            
            # Unpack stored data
            layer_connections, layers, medium_font, large_font, visible_layer_indices, layer_colors = self._current_data
            
            # Redraw with new metric
            create_layer_influence_chart(
                self.bar_ax, self.network_ax,
                layer_connections, layers, medium_font, large_font,
                visible_layer_indices, layer_colors,
                metric=metric
            )
            
            self.figure.tight_layout(pad=1.0)
            self.canvas.draw()
    
    def update_stats(self, data_manager):
        """Update the Layer Influence visualization with current data"""
        # Clear figure
        self.figure.clear()
        
        # Recreate the GridSpec layout
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 7])
        self.bar_ax = self.figure.add_subplot(gs[0])
        self.network_ax = self.figure.add_subplot(gs[1])
        
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
            create_layer_influence_chart(
                self.bar_ax, self.network_ax,
                layer_connections, layers, medium_font, large_font,
                visible_layer_indices, layer_colors,
                metric=self.influence_metric_dropdown.currentText()
            )
        else:
            self.bar_ax.text(0.5, 0.5, "Layer influence disabled", 
                           ha='center', va='center', fontsize=12)
            self.bar_ax.axis('off')
            
            self.network_ax.text(0.5, 0.5, "Layer influence disabled", 
                               ha='center', va='center', fontsize=12)
            self.network_ax.axis('off')
        
        # Apply tight layout
        self.figure.tight_layout(pad=1.0)
        
        # Draw canvas
        self.canvas.draw() 