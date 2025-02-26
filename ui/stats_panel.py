import logging
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTabWidget

from charts.layer_connectivity import create_layer_connectivity_chart
from charts.cluster_distribution import create_cluster_distribution_chart
from charts.betweenness_centrality import create_betweenness_centrality_chart
from charts.interlayer_graph import create_interlayer_graph
from charts.layer_activity import create_layer_activity_chart
from charts.layer_similarity import create_layer_similarity_chart
from charts.cluster_sankey import create_cluster_sankey_chart
from charts.cluster_chord import create_cluster_chord_diagram
from charts.cluster_heatmap import create_cluster_heatmap
from charts.cluster_alluvial import create_cluster_alluvial

class NetworkStatsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)  # Cleaner look
        layout.addWidget(self.tab_widget)

        # Create first tab for main charts
        self.main_charts_tab = QWidget()
        self.tab_widget.addTab(self.main_charts_tab, "Network Statistics")

        # Create tabs for each cluster visualization
        self.sankey_tab = QWidget()
        self.tab_widget.addTab(self.sankey_tab, "test")

        #self.chord_tab = QWidget()
        #self.tab_widget.addTab(self.chord_tab, "Cluster Chord")

        #self.heatmap_tab = QWidget()
        #self.tab_widget.addTab(self.heatmap_tab, "Cluster Heatmap")

        #self.alluvial_tab = QWidget()
        #self.tab_widget.addTab(self.alluvial_tab, "Cluster Alluvial")

        # Setup main charts tab
        main_charts_layout = QVBoxLayout(self.main_charts_tab)
        main_charts_layout.setContentsMargins(0, 0, 0, 0)
        main_charts_layout.setSpacing(2)

        # Create a horizontal layout for two columns of charts
        charts_layout = QHBoxLayout()
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(2)
        main_charts_layout.addLayout(charts_layout)

        # Create two matplotlib figures for left and right columns
        self.left_figure = Figure(figsize=(4, 10), dpi=100)
        self.left_canvas = FigureCanvas(self.left_figure)
        charts_layout.addWidget(self.left_canvas)

        self.right_figure = Figure(figsize=(4, 10), dpi=100)
        self.right_canvas = FigureCanvas(self.right_figure)
        charts_layout.addWidget(self.right_canvas)

        # Initialize left column subplots
        self.layer_connectivity_ax = self.left_figure.add_subplot(311)
        self.cluster_distribution_ax = self.left_figure.add_subplot(312)
        self.layer_activity_ax = self.left_figure.add_subplot(313)

        # Initialize right column subplots
        self.betweenness_centrality_ax = self.right_figure.add_subplot(311)
        self.interlayer_graph_ax = self.right_figure.add_subplot(312)
        self.layer_similarity_ax = self.right_figure.add_subplot(313)

        # Setup Sankey tab
        sankey_layout = QVBoxLayout(self.sankey_tab)
        sankey_layout.setContentsMargins(5, 5, 5, 5)

        # Create figure for Sankey diagram
        self.sankey_figure = Figure(figsize=(8, 10), dpi=100)
        self.sankey_canvas = FigureCanvas(self.sankey_figure)
        sankey_layout.addWidget(self.sankey_canvas)

        # Initialize Sankey subplot
        self.cluster_sankey_ax = self.sankey_figure.add_subplot(111)
        
        # Setup Chord tab
        #chord_layout = QVBoxLayout(self.chord_tab)
        #chord_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create figure for Chord diagram
        #elf.chord_figure = Figure(figsize=(8, 10), dpi=100)
        #self.chord_canvas = FigureCanvas(self.chord_figure)
        #chord_layout.addWidget(self.chord_canvas)
        
        # Initialize Chord subplot
        #self.cluster_chord_ax = self.chord_figure.add_subplot(111)
        
        # Setup Heatmap tab
        #heatmap_layout = QVBoxLayout(self.heatmap_tab)
        #heatmap_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create figure for Heatmap
        # self.heatmap_figure = Figure(figsize=(8, 10), dpi=100)
        #self.heatmap_canvas = FigureCanvas(self.heatmap_figure)
        #heatmap_layout.addWidget(self.heatmap_canvas)
        
        # Initialize Heatmap subplot
        # self.cluster_heatmap_ax = self.heatmap_figure.add_subplot(111)
        
        # Setup Alluvial tab
        #alluvial_layout = QVBoxLayout(self.alluvial_tab)
        #alluvial_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create figure for Alluvial diagram
        #self.alluvial_figure = Figure(figsize=(8, 10), dpi=100)
        #self.alluvial_canvas = FigureCanvas(self.alluvial_figure)
        #alluvial_layout.addWidget(self.alluvial_canvas)
        
        # Initialize Alluvial subplot
        #self.cluster_alluvial_ax = self.alluvial_figure.add_subplot(111)

        # Apply tight layout to all figures
        #self.left_figure.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
        #self.right_figure.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
        #self.sankey_figure.tight_layout(pad=0.5)
        #self.chord_figure.tight_layout(pad=0.5)
        #self.heatmap_figure.tight_layout(pad=0.5)
        #self.alluvial_figure.tight_layout(pad=0.5)

    def update_stats(self, node_positions, link_pairs, node_ids, layers, node_clusters, node_mask, edge_mask, visible_layer_indices, layer_colors=None):
        """Update statistics based on currently visible network elements"""
        logger = logging.getLogger(__name__)

        # Clear all figures
        self.left_figure.clear()
        self.right_figure.clear()
        self.sankey_figure.clear()
        #self.chord_figure.clear()
        #self.heatmap_figure.clear()
        #self.alluvial_figure.clear()

        # Re-create left column subplots
        self.layer_connectivity_ax = self.left_figure.add_subplot(311)
        self.cluster_distribution_ax = self.left_figure.add_subplot(312)
        self.layer_activity_ax = self.left_figure.add_subplot(313)

        # Re-create right column subplots 
        self.betweenness_centrality_ax = self.right_figure.add_subplot(311)
        self.interlayer_graph_ax = self.right_figure.add_subplot(312)
        self.layer_similarity_ax = self.right_figure.add_subplot(313)
        
        # Re-create visualization subplots
        self.cluster_sankey_ax = self.sankey_figure.add_subplot(111)
        #self.cluster_chord_ax = self.chord_figure.add_subplot(111)
        #self.cluster_heatmap_ax = self.heatmap_figure.add_subplot(111)
        #self.cluster_alluvial_ax = self.alluvial_figure.add_subplot(111)

        small_font = {'fontsize': 6}
        medium_font = {'fontsize': 7}
        large_font = {'fontsize': 9}  # For larger visualizations

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
        
        # --- CLUSTER VISUALIZATION TABS ---
        
        # Cluster Sankey chart
        create_cluster_sankey_chart(
            self.cluster_sankey_ax, visible_links, node_ids, node_clusters, 
            nodes_per_layer, layers, medium_font, large_font, visible_layer_indices
        )
        
        # Cluster Chord diagram
        # create_cluster_chord_diagram(
        #     self.cluster_chord_ax, visible_links, node_ids, node_clusters,
        #     nodes_per_layer, layers, medium_font, large_font, visible_layer_indices
        # )
        
        # Cluster Heatmap
        # create_cluster_heatmap(
        #     self.cluster_heatmap_ax, visible_links, node_ids, node_clusters,
        #     nodes_per_layer, layers, medium_font, large_font, visible_layer_indices
        # )
        
        # Cluster Alluvial diagram
        # create_cluster_alluvial(
        #     self.cluster_alluvial_ax, visible_links, node_ids, node_clusters,
        #     nodes_per_layer, layers, medium_font, large_font, visible_layer_indices
        # )

        # Apply tight layout to all figures
        self.left_figure.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
        self.right_figure.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
        self.sankey_figure.tight_layout(pad=1.0)
        #self.chord_figure.tight_layout(pad=1.0)
        #self.heatmap_figure.tight_layout(pad=1.0)
        #self.alluvial_figure.tight_layout(pad=1.0)

        # Draw all canvases
        self.left_canvas.draw()
        self.right_canvas.draw()
        self.sankey_canvas.draw()
        #self.chord_canvas.draw()
        #self.heatmap_canvas.draw()
        #self.alluvial_canvas.draw() 