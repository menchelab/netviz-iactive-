from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import logging

from charts.layer_connectivity import create_layer_connectivity_chart
from charts.cluster_distribution import create_cluster_distribution_chart
from charts.betweenness_centrality import create_betweenness_centrality_chart
from charts.interlayer_graph import create_interlayer_graph
from charts.layer_activity import create_layer_activity_chart
from charts.layer_similarity import create_layer_similarity_chart

class NetworkStatsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create a horizontal layout for two columns of charts
        charts_layout = QHBoxLayout()
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(2)
        layout.addLayout(charts_layout)
        
        # Create two matplotlib figures for left and right columns
        self.left_figure = Figure(figsize=(4, 10), dpi=100)
        self.left_canvas = FigureCanvas(self.left_figure)
        charts_layout.addWidget(self.left_canvas)
        
        self.right_figure = Figure(figsize=(4, 10), dpi=100)
        self.right_canvas = FigureCanvas(self.right_figure)
        charts_layout.addWidget(self.right_canvas)
        
        # Initialize left column subplots - now with 3 charts instead of 4
        self.layer_connectivity_ax = self.left_figure.add_subplot(311)
        self.cluster_distribution_ax = self.left_figure.add_subplot(312)
        self.layer_activity_ax = self.left_figure.add_subplot(313)
        
        # Initialize right column subplots - now with 3 charts instead of 4
        self.betweenness_centrality_ax = self.right_figure.add_subplot(311)
        self.interlayer_graph_ax = self.right_figure.add_subplot(312)
        self.layer_similarity_ax = self.right_figure.add_subplot(313)
        
        # Set tight layout with minimal padding for both figures
        self.left_figure.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
        self.right_figure.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    
    def update_stats(self, node_positions, link_pairs, node_ids, layers, node_clusters, node_mask, edge_mask, visible_layer_indices, layer_colors=None):
        """Update statistics based on currently visible network elements"""
        logger = logging.getLogger(__name__)
        
        # Clear both figures
        self.left_figure.clear()
        self.right_figure.clear()
        
        # Re-create left column subplots - now with 3 charts instead of 4
        self.layer_connectivity_ax = self.left_figure.add_subplot(311)
        self.cluster_distribution_ax = self.left_figure.add_subplot(312)
        self.layer_activity_ax = self.left_figure.add_subplot(313)
        
        # Re-create right column subplots - now with 3 charts instead of 4
        self.betweenness_centrality_ax = self.right_figure.add_subplot(311)
        self.interlayer_graph_ax = self.right_figure.add_subplot(312)
        self.layer_similarity_ax = self.right_figure.add_subplot(313)
        
        # Set smaller font sizes for all text elements
        small_font = {'fontsize': 6}
        medium_font = {'fontsize': 7}
        
        # Get visible nodes and edges
        visible_nodes = [i for i, mask in enumerate(node_mask) if mask]
        visible_edges = [i for i, mask in enumerate(edge_mask) if mask]
        visible_links = [link_pairs[i] for i in visible_edges]
        
        # Calculate nodes per layer
        nodes_per_layer = len(node_positions) // len(layers)
        
        # --- LEFT COLUMN CHARTS ---
        
        # 1. Layer connectivity matrix
        im, layer_connections = create_layer_connectivity_chart(
            self.layer_connectivity_ax, visible_links, nodes_per_layer, layers, small_font, medium_font
        )
        cbar = self.left_figure.colorbar(im, ax=self.layer_connectivity_ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=6)  # Smaller colorbar ticks
        
        # 2. Cluster distribution
        visible_node_ids = [node_ids[i] for i in visible_nodes]
        create_cluster_distribution_chart(
            self.cluster_distribution_ax, visible_node_ids, node_clusters, small_font, medium_font
        )
        
        # 3. Layer activity chart
        create_layer_activity_chart(
            self.layer_activity_ax, visible_links, nodes_per_layer, layers, small_font, medium_font
        )
        
        # --- RIGHT COLUMN CHARTS ---
        
        # 1. Betweenness centrality analysis
        create_betweenness_centrality_chart(
            self.betweenness_centrality_ax, layer_connections, layers, 
            visible_layer_indices, small_font, medium_font
        )
        
        # 2. Interlayer graph visualization
        create_interlayer_graph(
            self.interlayer_graph_ax, layer_connections, layers, 
            small_font, medium_font, visible_layer_indices, layer_colors
        )
        
        # 3. Layer similarity dendrogram
        create_layer_similarity_chart(
            self.layer_similarity_ax, layer_connections, layers, small_font, medium_font
        )
        
        # Apply tight layout with minimal padding for both figures
        self.left_figure.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
        self.right_figure.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
        
        # Draw both canvases
        self.left_canvas.draw()
        self.right_canvas.draw() 