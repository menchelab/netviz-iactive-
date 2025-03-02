import logging
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QTabWidget, QComboBox, QLabel
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Import from the new location
from charts.layer_cluster import (
    create_layer_cluster_overlap_heatmap,
    create_cluster_layer_distribution,
    create_layer_cluster_distribution,
)

# Import the newly migrated functions
from charts.layer_cluster.lc4_network_diagram import create_layer_cluster_network_diagram
from charts.layer_cluster.lc5_cluster_network import create_cluster_network
from charts.layer_cluster.lc6_sankey_diagram import create_layer_cluster_sankey
from charts.layer_cluster.lc7_connectivity_matrix import create_cluster_connectivity_matrix, create_layer_connectivity_matrix
from charts.layer_cluster.lc8_chord_diagram import create_layer_cluster_chord
from charts.layer_cluster.lc9_density_heatmap import create_layer_cluster_density_heatmap
from charts.layer_cluster.lc10_cooccurrence_network import create_cluster_cooccurrence_network
from charts.layer_cluster.lc11_normalized_heatmap import create_layer_cluster_normalized_heatmap
from charts.layer_cluster.lc12_similarity_matrix import create_cluster_similarity_matrix
from charts.layer_cluster.lc13_bubble_chart import create_layer_cluster_bubble_chart
from charts.layer_cluster.lc14_treemap import create_layer_cluster_treemap
from charts.layer_cluster.lc16_interlayer_paths import create_interlayer_path_analysis
from charts.layer_cluster.lc17_cluster_bridging_analysis import create_cluster_bridging_analysis

from ui.stats_panel.base_panel import BaseStatsPanel


class LayerClusterOverlapPanel(BaseStatsPanel):
    """Panel for visualizing layer and cluster overlaps"""

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Add checkbox to enable/disable charts
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(5, 5, 5, 0)
        self.enable_checkbox = QCheckBox("Enable Charts")
        self.enable_checkbox.setChecked(True)
        self.enable_checkbox.stateChanged.connect(self.on_state_changed)
        controls_layout.addWidget(self.enable_checkbox)
        
        # Add layout algorithm dropdown for LC4
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QLabel("LC4 Layout:"))
        self.layout_algorithm_combo = QComboBox()
        self.layout_algorithm_combo.addItems([
            "Community", "Bipartite", "Circular", "Spectral", "Spring"
        ])
        self.layout_algorithm_combo.setCurrentText("Community")
        self.layout_algorithm_combo.currentTextChanged.connect(self.on_layout_changed)
        controls_layout.addWidget(self.layout_algorithm_combo)
        
        # Add aspect ratio dropdown for LC4
        controls_layout.addSpacing(10)
        controls_layout.addWidget(QLabel("Aspect:"))
        self.aspect_ratio_combo = QComboBox()
        self.aspect_ratio_combo.addItems(["0.75", "1.0", "1.25", "1.5", "2.0"])
        self.aspect_ratio_combo.setCurrentText("1.0")
        self.aspect_ratio_combo.currentTextChanged.connect(self.on_layout_changed)
        controls_layout.addWidget(self.aspect_ratio_combo)
        
        # Add analysis type dropdown for LC16
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QLabel("LC16 Analysis:"))
        self.path_analysis_combo = QComboBox()
        self.path_analysis_combo.addItems([
            "Path Length", "Betweenness", "Bottleneck"
        ])
        self.path_analysis_combo.currentIndexChanged.connect(self.on_layout_changed)
        controls_layout.addWidget(self.path_analysis_combo)
        
        # Add analysis type dropdown for LC17
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QLabel("LC17 Analysis:"))
        self.bridge_analysis_combo = QComboBox()
        self.bridge_analysis_combo.addItems([
            "Bridge Score", "Flow Efficiency", "Layer Span"
        ])
        self.bridge_analysis_combo.currentIndexChanged.connect(self.on_layout_changed)
        controls_layout.addWidget(self.bridge_analysis_combo)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Create tab widget for the charts
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)  # Cleaner look
        layout.addWidget(self.tab_widget)

        # Create figures and canvases for all charts
        # LC1: Heatmap
        self.heatmap_figure = Figure(figsize=(10, 8), dpi=100)
        self.heatmap_canvas = FigureCanvas(self.heatmap_figure)
        self.heatmap_ax = self.heatmap_figure.add_subplot(111)
        
        # LC2: Distribution (clusters on x-axis)
        self.distribution_figure = Figure(figsize=(10, 6), dpi=100)
        self.distribution_canvas = FigureCanvas(self.distribution_figure)
        self.distribution_ax = self.distribution_figure.add_subplot(111)
        
        # LC3: Distribution (layers on x-axis)
        self.layer_distribution_figure = Figure(figsize=(10, 6), dpi=100)
        self.layer_distribution_canvas = FigureCanvas(self.layer_distribution_figure)
        self.layer_distribution_ax = self.layer_distribution_figure.add_subplot(111)
        
        # LC4: Network diagram with similarity matrix
        self.network_figure = Figure(figsize=(12, 10), dpi=100)
        self.network_canvas = FigureCanvas(self.network_figure)
        self.network_ax1 = self.network_figure.add_subplot(111)  # Full plot instead of 121
        
        # LC5: Cluster network with co-occurrence network
        self.cluster_network_figure = Figure(figsize=(10, 8), dpi=100)
        self.cluster_network_canvas = FigureCanvas(self.cluster_network_figure)
        self.cluster_network_ax1 = self.cluster_network_figure.add_subplot(121)  # Left subplot
        self.cluster_network_ax2 = self.cluster_network_figure.add_subplot(122)  # Right subplot
        
        # LC6: Sankey diagram
        self.sankey_figure = Figure(figsize=(10, 6), dpi=100)
        self.sankey_canvas = FigureCanvas(self.sankey_figure)
        self.sankey_ax = self.sankey_figure.add_subplot(111)
        
        # LC7: Cluster connectivity matrix
        self.connectivity_figure = Figure(figsize=(15, 12), dpi=100)  # Taller figure for 6 subplots (2 rows, 3 columns)
        self.connectivity_canvas = FigureCanvas(self.connectivity_figure)
        # Cluster connectivity matrices (top row)
        self.connectivity_ax1 = self.connectivity_figure.add_subplot(231)  # Top-left
        self.connectivity_ax2 = self.connectivity_figure.add_subplot(232)  # Top-middle
        self.connectivity_ax3 = self.connectivity_figure.add_subplot(233)  # Top-right
        # Layer connectivity matrices (bottom row)
        self.connectivity_ax4 = self.connectivity_figure.add_subplot(234)  # Bottom-left
        self.connectivity_ax5 = self.connectivity_figure.add_subplot(235)  # Bottom-middle
        self.connectivity_ax6 = self.connectivity_figure.add_subplot(236)  # Bottom-right
        
        # LC8: Chord diagram
        self.chord_figure = Figure(figsize=(8, 8), dpi=100)
        self.chord_canvas = FigureCanvas(self.chord_figure)
        self.chord_ax = self.chord_figure.add_subplot(111)
        
        # LC9: Density heatmap
        self.density_figure = Figure(figsize=(10, 8), dpi=100)
        self.density_canvas = FigureCanvas(self.density_figure)
        self.density_ax = self.density_figure.add_subplot(111)
        
        # LC10: Co-occurrence network
        self.cooccurrence_figure = Figure(figsize=(8, 8), dpi=100)
        self.cooccurrence_canvas = FigureCanvas(self.cooccurrence_figure)
        self.cooccurrence_ax = self.cooccurrence_figure.add_subplot(111)
        
        # LC11: Normalized heatmap
        self.normalized_figure = Figure(figsize=(10, 8), dpi=100)
        self.normalized_canvas = FigureCanvas(self.normalized_figure)
        self.normalized_ax = self.normalized_figure.add_subplot(111)
        
        # LC12: Similarity matrix
        self.similarity_figure = Figure(figsize=(10, 8), dpi=100)
        self.similarity_canvas = FigureCanvas(self.similarity_figure)
        self.similarity_ax = self.similarity_figure.add_subplot(111)
        
        # LC13: Bubble chart
        self.bubble_figure = Figure(figsize=(10, 8), dpi=100)
        self.bubble_canvas = FigureCanvas(self.bubble_figure)
        self.bubble_ax = self.bubble_figure.add_subplot(111)
        
        # LC14: Treemap
        self.treemap_figure = Figure(figsize=(10, 8), dpi=100)
        self.treemap_canvas = FigureCanvas(self.treemap_figure)
        self.treemap_ax = self.treemap_figure.add_subplot(111)
        
        # LC15: Flow Visualization Ideas
        self.flow_ideas_figure = Figure(figsize=(12, 10), dpi=100)
        self.flow_ideas_canvas = FigureCanvas(self.flow_ideas_figure)
        self.flow_ideas_ax1 = self.flow_ideas_figure.add_subplot(221)  # Top-left
        self.flow_ideas_ax2 = self.flow_ideas_figure.add_subplot(222)  # Top-right
        self.flow_ideas_ax3 = self.flow_ideas_figure.add_subplot(223)  # Bottom-left
        self.flow_ideas_ax4 = self.flow_ideas_figure.add_subplot(224)  # Bottom-right

        # Create figures for the new LC16 and LC17 tabs
        self.path_analysis_figure = Figure(figsize=(12, 10), dpi=100)
        self.path_analysis_canvas = FigureCanvas(self.path_analysis_figure)
        
        self.bridge_analysis_figure = Figure(figsize=(12, 10), dpi=100)
        self.bridge_analysis_canvas = FigureCanvas(self.bridge_analysis_figure)

        # Add each canvas to a tab
        self.tab_widget.addTab(self.heatmap_canvas, "LC1: Heatmap")
        self.tab_widget.addTab(self.distribution_canvas, "LC2: Distribution")
        self.tab_widget.addTab(self.layer_distribution_canvas, "LC3: Layer Distribution")
        self.tab_widget.addTab(self.network_canvas, "LC4: Network & Similarity")
        self.tab_widget.addTab(self.cluster_network_canvas, "LC5: Cluster Networks")
        self.tab_widget.addTab(self.sankey_canvas, "LC6: Sankey")
        self.tab_widget.addTab(self.connectivity_canvas, "LC7: Connectivity Matrix")
        self.tab_widget.addTab(self.chord_canvas, "LC8: Chord Diagram")
        self.tab_widget.addTab(self.density_canvas, "LC9: Density Heatmap")
        self.tab_widget.addTab(self.cooccurrence_canvas, "LC10: Co-occurrence Network")
        self.tab_widget.addTab(self.normalized_canvas, "LC11: Normalized Heatmap")
        self.tab_widget.addTab(self.similarity_canvas, "LC12: Similarity Matrix")
        self.tab_widget.addTab(self.bubble_canvas, "LC13: Bubble Chart")
        self.tab_widget.addTab(self.treemap_canvas, "LC14: Treemap")
        self.tab_widget.addTab(self.path_analysis_canvas, "LC16: Interlayer Path Analysis")
        self.tab_widget.addTab(self.bridge_analysis_canvas, "LC17: Cluster Bridging Analysis")
        self.tab_widget.addTab(self.flow_ideas_canvas, "LC15: Flow Ideas")

        # Create tooltips for each tab
        self._create_tooltips()

    def on_state_changed(self, state):
        """Handle enable/disable state change"""
        if state and hasattr(self, "_current_data"):
            self.update_stats(self._current_data)
        elif not state:
            # Clear all figures when disabled
            self.heatmap_figure.clear()
            self.distribution_figure.clear()
            self.layer_distribution_figure.clear()
            self.network_figure.clear()
            self.cluster_network_figure.clear()
            self.sankey_figure.clear()
            self.connectivity_figure.clear()
            self.chord_figure.clear()
            self.density_figure.clear()
            self.cooccurrence_figure.clear()
            self.normalized_figure.clear()
            self.similarity_figure.clear()
            self.bubble_figure.clear()
            self.treemap_figure.clear()
            self.flow_ideas_figure.clear()

            # Add disabled message to all figures
            for fig in [self.heatmap_figure, self.distribution_figure, self.layer_distribution_figure,
                        self.network_figure, self.cluster_network_figure, self.sankey_figure,
                        self.connectivity_figure, self.chord_figure, self.density_figure,
                        self.cooccurrence_figure, self.normalized_figure, self.similarity_figure,
                        self.bubble_figure, self.treemap_figure, self.flow_ideas_figure]:
                fig.text(0.5, 0.5, "Charts disabled", ha="center", va="center", fontsize=14)
                
            # Draw all canvases
            self.heatmap_canvas.draw()
            self.distribution_canvas.draw()
            self.layer_distribution_canvas.draw()
            self.network_canvas.draw()
            self.cluster_network_canvas.draw()
            self.sankey_canvas.draw()
            self.connectivity_canvas.draw()
            self.chord_canvas.draw()
            self.density_canvas.draw()
            self.cooccurrence_canvas.draw()
            self.normalized_canvas.draw()
            self.similarity_canvas.draw()
            self.bubble_canvas.draw()
            self.treemap_canvas.draw()
            self.flow_ideas_canvas.draw()

    def on_layout_changed(self, _):
        """Handle layout algorithm or aspect ratio change"""
        if hasattr(self, "_current_data") and self.enable_checkbox.isChecked():
            # Only update the LC4 tab to avoid redrawing everything
            self.update_lc4_network_diagram(self._current_data)
            
    def update_lc4_network_diagram(self, data_manager):
        """Update only the LC4 network diagram with the current layout settings"""
        if not hasattr(self, "network_figure") or not self.enable_checkbox.isChecked():
            return
            
        # Clear the figure and recreate the axis
        self.network_figure.clear()
        self.network_ax1 = self.network_figure.add_subplot(111)
        
        # Get the selected layout algorithm and aspect ratio
        layout_algorithm = self.layout_algorithm_combo.currentText().lower()
        aspect_ratio = float(self.aspect_ratio_combo.currentText())
        
        # Get visible links
        visible_links = []
        if data_manager.current_edge_mask is not None:
            visible_links = [data_manager.link_pairs[i] for i in range(len(data_manager.link_pairs)) 
                            if data_manager.current_edge_mask[i]]
        
        # Create the network diagram with the selected layout
        create_layer_cluster_network_diagram(
            self.network_ax1,
            visible_links,
            data_manager.node_ids,
            data_manager.node_clusters,
            data_manager.nodes_per_layer,
            data_manager.layers,
            10,  # small_font
            12,  # medium_font
            data_manager.visible_layers,
            data_manager.cluster_colors,
            layout_algorithm=layout_algorithm,
            aspect_ratio=aspect_ratio
        )
        
        # Apply tight layout and draw
        self.network_figure.tight_layout(pad=1.5)
        self.network_canvas.draw()

    def update_stats(self, data_manager):
        """Update all visualizations with the current data"""
        if not self.enable_checkbox.isChecked():
            return
            
        self._current_data = data_manager
        
        # Clear all figures
        self.heatmap_figure.clear()
        self.distribution_figure.clear()
        self.layer_distribution_figure.clear()
        self.network_figure.clear()
        self.cluster_network_figure.clear()
        self.sankey_figure.clear()
        self.connectivity_figure.clear()
        self.chord_figure.clear()
        self.density_figure.clear()
        self.cooccurrence_figure.clear()
        self.normalized_figure.clear()
        self.similarity_figure.clear()
        self.bubble_figure.clear()
        self.treemap_figure.clear()
        self.flow_ideas_figure.clear()

        # Re-create all axes
        self.heatmap_ax = self.heatmap_figure.add_subplot(111)
        self.distribution_ax = self.distribution_figure.add_subplot(111)
        self.layer_distribution_ax = self.layer_distribution_figure.add_subplot(111)
        self.network_ax1 = self.network_figure.add_subplot(111)  # Full plot instead of 121
        self.cluster_network_ax1 = self.cluster_network_figure.add_subplot(121)
        self.cluster_network_ax2 = self.cluster_network_figure.add_subplot(122)
        self.sankey_ax = self.sankey_figure.add_subplot(111)
        # Cluster connectivity matrices (top row)
        self.connectivity_ax1 = self.connectivity_figure.add_subplot(231)  # Top-left
        self.connectivity_ax2 = self.connectivity_figure.add_subplot(232)  # Top-middle
        self.connectivity_ax3 = self.connectivity_figure.add_subplot(233)  # Top-right
        # Layer connectivity matrices (bottom row)
        self.connectivity_ax4 = self.connectivity_figure.add_subplot(234)  # Bottom-left
        self.connectivity_ax5 = self.connectivity_figure.add_subplot(235)  # Bottom-middle
        self.connectivity_ax6 = self.connectivity_figure.add_subplot(236)  # Bottom-right
        self.chord_ax = self.chord_figure.add_subplot(111)
        self.density_ax = self.density_figure.add_subplot(111)
        self.cooccurrence_ax = self.cooccurrence_figure.add_subplot(111)
        self.normalized_ax = self.normalized_figure.add_subplot(111)
        self.similarity_ax = self.similarity_figure.add_subplot(111)
        self.bubble_ax = self.bubble_figure.add_subplot(111)
        self.treemap_ax = self.treemap_figure.add_subplot(111)
        self.flow_ideas_ax1 = self.flow_ideas_figure.add_subplot(221)
        self.flow_ideas_ax2 = self.flow_ideas_figure.add_subplot(222)
        self.flow_ideas_ax3 = self.flow_ideas_figure.add_subplot(223)
        self.flow_ideas_ax4 = self.flow_ideas_figure.add_subplot(224)

        # Define font sizes - increased for better readability
        small_font = {"fontsize": 9}
        medium_font = {"fontsize": 12}

        # Get filtered data from the data manager using optimized methods
        filtered_data = data_manager.get_filtered_data_for_vispy()
        
        # Extract the data we need for visualization
        node_positions = filtered_data['node_positions']
        node_colors = filtered_data['node_colors']
        edge_connections = filtered_data['edge_connections']
        
        # Get additional data needed for statistics
        layers = data_manager.layers
        visible_layer_indices = data_manager.visible_layers
        layer_colors = data_manager.layer_colors
        
        # Get node IDs and clusters
        node_ids = data_manager.node_ids
        
        # Get cluster information directly from the data manager
        # This ensures we're using the correct cluster mapping
        node_clusters = data_manager.node_clusters
        unique_clusters = data_manager.unique_clusters
        
        # Get nodes per layer
        nodes_per_layer = data_manager.nodes_per_layer
        
        # LC1: Create layer-cluster overlap heatmap
        create_layer_cluster_overlap_heatmap(
            self.heatmap_ax,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.cluster_colors
        )

        # LC2: Create cluster-layer distribution chart
        create_cluster_layer_distribution(
            self.distribution_ax,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.cluster_colors
        )
        
        # LC3: Create layer-cluster distribution chart (inverse of LC2)
        create_layer_cluster_distribution(
            self.layer_distribution_ax,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.cluster_colors
        )

        # LC4: Network diagram
        # Get the selected layout algorithm and aspect ratio
        layout_algorithm = self.layout_algorithm_combo.currentText().lower()
        aspect_ratio = float(self.aspect_ratio_combo.currentText())
        
        create_layer_cluster_network_diagram(
            self.network_ax1,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.cluster_colors,
            layout_algorithm=layout_algorithm,
            aspect_ratio=aspect_ratio
        )
        
        # Set title for the network diagram
        self.network_ax1.set_title("Layer-Cluster Network", fontsize=medium_font["fontsize"])
        self.network_figure.tight_layout(pad=1.5)  # Increase padding for better fit
        self.network_canvas.draw()
        
        # LC5: Create cluster network
        create_cluster_network(
            self.cluster_network_ax1,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices
        )

        # Add co-occurrence network to LC5 tab
        create_cluster_cooccurrence_network(
            self.cluster_network_ax2,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.cluster_colors
        )
        
        # Set titles for the subplots
        self.cluster_network_ax1.set_title("Cluster Network", fontsize=medium_font["fontsize"])
        self.cluster_network_ax2.set_title("Cluster Co-occurrence Network", fontsize=medium_font["fontsize"])
        self.cluster_network_figure.tight_layout()
        self.cluster_network_canvas.draw()
        
        # LC6: Create layer-cluster Sankey diagram
        create_layer_cluster_sankey(
            self.sankey_ax,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices
        )
        
        # LC7: Create cluster connectivity matrices for different edge types
        # 1. All edges
        create_cluster_connectivity_matrix(
            self.connectivity_ax1,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.cluster_colors,
            edge_type="all"
        )
        self.connectivity_ax1.set_title("All Connections\nBetween Clusters", fontsize=medium_font["fontsize"])
        
        # 2. Same layer edges
        create_cluster_connectivity_matrix(
            self.connectivity_ax2,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.cluster_colors,
            edge_type="same_layer"
        )
        self.connectivity_ax2.set_title("Same Layer Connections\nBetween Clusters", fontsize=medium_font["fontsize"])
        
        # 3. Interlayer edges
        create_cluster_connectivity_matrix(
            self.connectivity_ax3,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.cluster_colors,
            edge_type="interlayer"
        )
        self.connectivity_ax3.set_title("Interlayer Connections\nBetween Clusters", fontsize=medium_font["fontsize"])
        
        # Create layer connectivity matrices for different edge types
        # 1. All edges
        create_layer_connectivity_matrix(
            self.connectivity_ax4,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.layer_colors,
            edge_type="all"
        )
        self.connectivity_ax4.set_title("All Connections\nBetween Layers", fontsize=medium_font["fontsize"])
        
        # 2. Same cluster edges
        create_layer_connectivity_matrix(
            self.connectivity_ax5,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.layer_colors,
            edge_type="same_cluster"
        )
        self.connectivity_ax5.set_title("Same Cluster Connections\nBetween Layers", fontsize=medium_font["fontsize"])
        
        # 3. Different cluster edges
        create_layer_connectivity_matrix(
            self.connectivity_ax6,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.layer_colors,
            edge_type="different_cluster"
        )
        self.connectivity_ax6.set_title("Different Cluster Connections\nBetween Layers", fontsize=medium_font["fontsize"])
        
        # Add a super title for the entire figure
        self.connectivity_figure.suptitle("Connectivity Matrices\nTop: Cluster-to-Cluster | Bottom: Layer-to-Layer", 
                                         fontsize=medium_font["fontsize"]+2, y=0.98)
        
        # LC8: Create layer-cluster chord diagram
        create_layer_cluster_chord(
            self.chord_ax,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.cluster_colors
        )
        
        # LC9: Create layer-cluster density heatmap
        create_layer_cluster_density_heatmap(
            self.density_ax,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.cluster_colors
        )
        
        # LC10: Create cluster co-occurrence network (now shown in LC5 tab)
        # We'll keep this for backward compatibility but hide the tab
        create_cluster_cooccurrence_network(
            self.cooccurrence_ax,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.cluster_colors
        )
        
        # LC11: Create normalized heatmap
        create_layer_cluster_normalized_heatmap(
            self.normalized_ax,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.cluster_colors
        )
        
        # LC13: Create bubble chart
        create_layer_cluster_bubble_chart(
            self.bubble_ax,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.cluster_colors
        )
        
        # LC14: Create treemap
        create_layer_cluster_treemap(
            self.treemap_ax,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.cluster_colors
        )
        
        # LC15: Create four different flow visualization ideas
        # Idea 1: Circular flow diagram
        self._create_circular_flow_diagram(
            self.flow_ideas_ax1,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.cluster_colors
        )
        self.flow_ideas_ax1.set_title("Idea 1: Circular Flow Diagram", fontsize=medium_font["fontsize"])
        
        # Idea 2: Force-directed graph with layers as regions
        self._create_force_directed_regions(
            self.flow_ideas_ax2,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.cluster_colors
        )
        self.flow_ideas_ax2.set_title("Idea 2: Force-Directed Regions", fontsize=medium_font["fontsize"])
        
        # Idea 3: Alluvial diagram
        self._create_alluvial_diagram(
            self.flow_ideas_ax3,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.cluster_colors
        )
        self.flow_ideas_ax3.set_title("Idea 3: Alluvial Diagram", fontsize=medium_font["fontsize"])
        
        # Idea 4: Radial network
        self._create_radial_network(
            self.flow_ideas_ax4,
            edge_connections,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.cluster_colors
        )
        self.flow_ideas_ax4.set_title("Idea 4: Radial Network", fontsize=medium_font["fontsize"])
        
        self.flow_ideas_figure.tight_layout()
        self.flow_ideas_canvas.draw()

        # Adjust layout and draw
        self.heatmap_figure.tight_layout()
        self.distribution_figure.tight_layout()
        self.layer_distribution_figure.tight_layout()
        self.network_figure.tight_layout()
        self.cluster_network_figure.tight_layout()
        self.sankey_figure.tight_layout()
        self.connectivity_figure.tight_layout()
        self.chord_figure.tight_layout()
        self.density_figure.tight_layout()
        self.cooccurrence_figure.tight_layout()
        self.normalized_figure.tight_layout()
        self.similarity_figure.tight_layout()
        self.bubble_figure.tight_layout()
        self.treemap_figure.tight_layout()
        self.flow_ideas_figure.tight_layout()
        
        self.heatmap_canvas.draw()
        self.distribution_canvas.draw()
        self.layer_distribution_canvas.draw()
        self.network_canvas.draw()
        self.cluster_network_canvas.draw()
        self.sankey_canvas.draw()
        self.connectivity_canvas.draw()
        self.chord_canvas.draw()
        self.density_canvas.draw()
        self.cooccurrence_canvas.draw()
        self.normalized_canvas.draw()
        self.similarity_canvas.draw()
        self.bubble_canvas.draw()
        self.treemap_canvas.draw()
        self.flow_ideas_canvas.draw() 

        # Update LC16: Interlayer Path Analysis
        self.update_lc16_path_analysis(data_manager)
        
        # Update LC17: Cluster Bridging Analysis
        self.update_lc17_bridge_analysis(data_manager)

    def update_lc16_path_analysis(self, data_manager):
        """Update the LC16 interlayer path analysis with the current analysis type"""
        if not hasattr(self, "path_analysis_figure") or not self.enable_checkbox.isChecked():
            return
            
        # Clear the figure and recreate the axis
        self.path_analysis_figure.clear()
        self.path_analysis_ax = self.path_analysis_figure.add_subplot(111)
        
        # Get the selected analysis type
        analysis_type = self.path_analysis_combo.currentText().lower().replace(" ", "_")
        
        # Get visible links
        visible_links = []
        if data_manager.current_edge_mask is not None:
            visible_links = [data_manager.link_pairs[i] for i in range(len(data_manager.link_pairs)) 
                            if data_manager.current_edge_mask[i]]
        
        # Get visible layer indices
        visible_layer_indices = []
        if hasattr(data_manager, 'visible_layer_indices'):
            visible_layer_indices = data_manager.visible_layer_indices
        else:
            visible_layer_indices = list(range(len(data_manager.layers)))
        
        # Create the interlayer path analysis
        create_interlayer_path_analysis(
            self.path_analysis_ax,
            visible_links,
            data_manager.node_ids,
            data_manager.node_clusters,
            data_manager.nodes_per_layer,
            data_manager.layers,
            visible_layer_indices,
            data_manager.cluster_colors,
            analysis_type
        )
        
        # Draw the canvas
        self.path_analysis_canvas.draw()

    def update_lc17_bridge_analysis(self, data_manager):
        """Update the LC17 cluster bridging analysis with the current analysis type"""
        if not hasattr(self, "bridge_analysis_figure") or not self.enable_checkbox.isChecked():
            return
            
        # Clear the figure and recreate the axis
        self.bridge_analysis_figure.clear()
        self.bridge_analysis_ax = self.bridge_analysis_figure.add_subplot(111)
        
        # Get the selected analysis type
        analysis_type = self.bridge_analysis_combo.currentText().lower().replace(" ", "_")
        
        # Get visible links
        visible_links = []
        if data_manager.current_edge_mask is not None:
            visible_links = [data_manager.link_pairs[i] for i in range(len(data_manager.link_pairs)) 
                            if data_manager.current_edge_mask[i]]
        
        # Get visible layer indices
        visible_layer_indices = []
        if hasattr(data_manager, 'visible_layer_indices'):
            visible_layer_indices = data_manager.visible_layer_indices
        else:
            visible_layer_indices = list(range(len(data_manager.layers)))
        
        # Create the cluster bridging analysis
        create_cluster_bridging_analysis(
            self.bridge_analysis_ax,
            visible_links,
            data_manager.node_ids,
            data_manager.node_clusters,
            data_manager.nodes_per_layer,
            data_manager.layers,
            visible_layer_indices,
            data_manager.cluster_colors,
            data_manager.layer_colors,
            analysis_type
        )
        
        # Draw the canvas
        self.bridge_analysis_canvas.draw()

    def _create_circular_flow_diagram(self, ax, visible_links, node_ids, node_clusters, nodes_per_layer, layers, small_font, medium_font, visible_layer_indices, cluster_colors):
        """
        Create a circular flow diagram showing connections between layers and clusters.
        Layers and clusters are arranged in a circle, with edges representing connections.
        """
        import networkx as nx
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
        from collections import defaultdict
        
        # Clear the axis
        ax.clear()
        
        # Filter visible layers if specified
        if visible_layer_indices is not None:
            visible_layers = set(visible_layer_indices)
        else:
            visible_layers = set(range(len(layers)))
        
        # Count nodes by cluster and layer
        cluster_layer_counts = defaultdict(lambda: defaultdict(int))
        
        for node_id, cluster in node_clusters.items():
            if node_id in node_ids:
                node_idx = node_ids.index(node_id)
                layer_idx = node_idx // nodes_per_layer
                
                if layer_idx in visible_layers:
                    cluster_layer_counts[cluster][layer_idx] += 1
        
        # Check if we have any data
        if not cluster_layer_counts:
            ax.text(0.5, 0.5, "No data to display", ha="center", va="center")
            ax.axis("off")
            return
        
        # Get unique layers and clusters
        unique_clusters = sorted(cluster_layer_counts.keys())
        unique_layer_indices = sorted(visible_layers)
        
        # Create a graph
        G = nx.Graph()
        
        # Add nodes for layers and clusters
        for layer_idx in unique_layer_indices:
            if layer_idx < len(layers):
                G.add_node(f"L_{layers[layer_idx]}", type="layer")
        
        for cluster in unique_clusters:
            G.add_node(f"C_{cluster}", type="cluster")
        
        # Add edges between layers and clusters
        for cluster, layer_dict in cluster_layer_counts.items():
            for layer_idx, count in layer_dict.items():
                if layer_idx in unique_layer_indices and layer_idx < len(layers):
                    G.add_edge(f"L_{layers[layer_idx]}", f"C_{cluster}", weight=count)
        
        # Create a circular layout
        pos = nx.circular_layout(G)
        
        # Draw nodes
        layer_nodes = [n for n in G.nodes() if n.startswith("L_")]
        cluster_nodes = [n for n in G.nodes() if n.startswith("C_")]
        
        # Draw layer nodes
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=layer_nodes,
            node_color="lightblue",
            node_size=500,
            alpha=0.8,
            edgecolors='black',
            ax=ax
        )
        
        # Draw cluster nodes with their respective colors
        for cluster in unique_clusters:
            node = f"C_{cluster}"
            color = cluster_colors.get(cluster, "gray") if cluster_colors else "gray"
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=[node],
                node_color=[color],
                node_size=500,
                alpha=0.8,
                edgecolors='black',
                ax=ax
            )
        
        # Draw edges with width based on weight
        edge_weights = [G.edges[edge]['weight'] for edge in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        
        for (u, v, data) in G.edges(data=True):
            weight = data.get('weight', 1)
            width = 0.5 + 2.5 * (weight / max_weight)
            # Use a color that reflects the weight
            edge_color = plt.cm.Blues(0.2 + 0.8 * (weight / max_weight))
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=width,
                alpha=0.7,
                edge_color=[edge_color],
                ax=ax
            )
        
        # Draw node labels
        nx.draw_networkx_labels(
            G, pos,
            labels={n: n.split("_")[1] for n in G.nodes()},
            font_size=8,
            font_weight='bold',
            ax=ax
        )
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    
    def _create_force_directed_regions(self, ax, visible_links, node_ids, node_clusters, nodes_per_layer, layers, small_font, medium_font, visible_layer_indices, cluster_colors):
        """
        Create a force-directed graph with layers as regions.
        Nodes are colored by cluster and positioned in regions based on their layer.
        """
        import networkx as nx
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from collections import defaultdict
        
        # Clear the axis
        ax.clear()
        
        # Filter visible layers if specified
        if visible_layer_indices is not None:
            visible_layers = set(visible_layer_indices)
        else:
            visible_layers = set(range(len(layers)))
        
        # Get visible node indices
        visible_node_indices = set()
        for start_idx, end_idx in visible_links:
            visible_node_indices.add(start_idx)
            visible_node_indices.add(end_idx)
        
        # Create a graph
        G = nx.Graph()
        
        # Map node indices to clusters and layers
        node_to_cluster = {}
        node_to_layer = {}
        
        for node_idx in visible_node_indices:
            layer_idx = node_idx // nodes_per_layer
            
            if layer_idx not in visible_layers:
                continue
                
            node_id = node_ids[node_idx]
            cluster = node_clusters.get(node_id, "Unknown")
            
            # Add node to graph
            G.add_node(node_idx, cluster=cluster, layer=layer_idx)
            
            node_to_cluster[node_idx] = cluster
            node_to_layer[node_idx] = layer_idx
        
        # Add edges
        for start_idx, end_idx in visible_links:
            if start_idx in node_to_cluster and end_idx in node_to_cluster:
                G.add_edge(start_idx, end_idx)
        
        # Check if we have any data
        if not G.nodes():
            ax.text(0.5, 0.5, "No data to display", ha="center", va="center")
            ax.axis("off")
            return
        
        # Create a layout that groups nodes by layer
        pos = {}
        layer_centers = {}
        
        # Calculate layer centers in a circle
        num_layers = len(visible_layers)
        for i, layer_idx in enumerate(sorted(visible_layers)):
            angle = 2 * np.pi * i / num_layers
            layer_centers[layer_idx] = (np.cos(angle), np.sin(angle))
        
        # Position nodes near their layer center with some randomness
        for node in G.nodes():
            layer_idx = G.nodes[node]['layer']
            center_x, center_y = layer_centers[layer_idx]
            
            # Add some random offset
            offset_x = np.random.normal(0, 0.1)
            offset_y = np.random.normal(0, 0.1)
            
            pos[node] = (center_x + offset_x, center_y + offset_y)
        
        # Draw layer regions
        for layer_idx, (center_x, center_y) in layer_centers.items():
            if layer_idx < len(layers):
                ellipse = Ellipse(
                    (center_x, center_y), 0.5, 0.5,
                    alpha=0.2,
                    facecolor='lightgray',
                    edgecolor='gray'
                )
                ax.add_patch(ellipse)
                
                # Add layer label
                ax.text(
                    center_x, center_y,
                    layers[layer_idx],
                    ha='center', va='center',
                    fontsize=10,
                    fontweight='bold'
                )
        
        # Draw nodes colored by cluster
        for cluster in set(node_to_cluster.values()):
            cluster_nodes = [n for n in G.nodes() if G.nodes[n]['cluster'] == cluster]
            
            if not cluster_nodes:
                continue
                
            color = cluster_colors.get(cluster, "gray") if cluster_colors else "gray"
            
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=cluster_nodes,
                node_color=color,
                node_size=50,
                alpha=0.8,
                edgecolors='black',
                ax=ax
            )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            width=0.5,
            alpha=0.5,
            ax=ax
        )
        
        # Create a legend for clusters
        legend_elements = []
        for cluster in sorted(set(node_to_cluster.values()))[:5]:  # Limit to top 5
            color = cluster_colors.get(cluster, "gray") if cluster_colors else "gray"
            legend_elements.append(
                plt.Line2D([0], [0], 
                          marker='o', 
                          color='w',
                          markerfacecolor=color,
                          markersize=8,
                          label=f"Cluster {cluster}")
            )
        
        # Add the legend
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=8
        )
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    
    def _create_alluvial_diagram(self, ax, visible_links, node_ids, node_clusters, nodes_per_layer, layers, small_font, medium_font, visible_layer_indices, cluster_colors):
        """
        Create an alluvial diagram showing flow between layers and clusters.
        Similar to a Sankey diagram but with a different layout.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch
        from collections import defaultdict
        
        # Clear the axis
        ax.clear()
        
        # Filter visible layers if specified
        if visible_layer_indices is not None:
            visible_layers = sorted(visible_layer_indices)
        else:
            visible_layers = sorted(range(len(layers)))
        
        # Count nodes by cluster and layer
        cluster_layer_counts = defaultdict(lambda: defaultdict(int))
        
        for node_id, cluster in node_clusters.items():
            if node_id in node_ids:
                node_idx = node_ids.index(node_id)
                layer_idx = node_idx // nodes_per_layer
                
                if layer_idx in visible_layers:
                    cluster_layer_counts[cluster][layer_idx] += 1
        
        # Check if we have any data
        if not cluster_layer_counts:
            ax.text(0.5, 0.5, "No data to display", ha="center", va="center")
            ax.axis("off")
            return
        
        # Get unique clusters
        unique_clusters = sorted(cluster_layer_counts.keys())
        
        # Calculate total nodes per layer
        layer_totals = defaultdict(int)
        for cluster, layer_dict in cluster_layer_counts.items():
            for layer_idx, count in layer_dict.items():
                layer_totals[layer_idx] += count
        
        # Calculate positions for each layer
        layer_positions = {}
        x_step = 1.0 / (len(visible_layers) + 1)
        
        for i, layer_idx in enumerate(visible_layers):
            x_pos = (i + 1) * x_step
            layer_positions[layer_idx] = x_pos
        
        # Draw the alluvial diagram
        for cluster in unique_clusters:
            # Get a color for this cluster
            color = cluster_colors.get(cluster, "gray") if cluster_colors else "gray"
            
            # Track y positions for this cluster in each layer
            y_positions = {}
            
            # Calculate y positions
            for layer_idx in visible_layers:
                if layer_idx in cluster_layer_counts[cluster]:
                    count = cluster_layer_counts[cluster][layer_idx]
                    total = layer_totals[layer_idx]
                    
                    # Calculate the height of this cluster in this layer
                    height = count / total if total > 0 else 0
                    
                    # Calculate the y position (centered)
                    y_pos = 0.5 - height / 2
                    
                    y_positions[layer_idx] = (y_pos, height)
            
            # Draw flows between consecutive layers
            for i in range(len(visible_layers) - 1):
                layer1 = visible_layers[i]
                layer2 = visible_layers[i + 1]
                
                if layer1 in y_positions and layer2 in y_positions:
                    x1 = layer_positions[layer1]
                    y1, h1 = y_positions[layer1]
                    
                    x2 = layer_positions[layer2]
                    y2, h2 = y_positions[layer2]
                    
                    # Create a path for the flow
                    verts = [
                        (x1, y1),  # Start at top-left
                        (x1, y1 + h1),  # Bottom-left
                        (x2, y2 + h2),  # Bottom-right
                        (x2, y2),  # Top-right
                        (x1, y1)   # Back to start
                    ]
                    
                    codes = [
                        Path.MOVETO,
                        Path.LINETO,
                        Path.LINETO,
                        Path.LINETO,
                        Path.CLOSEPOLY
                    ]
                    
                    path = Path(verts, codes)
                    patch = PathPatch(
                        path, facecolor=color, edgecolor='black',
                        alpha=0.7, linewidth=0.5
                    )
                    ax.add_patch(patch)
            
            # Draw rectangles for each layer
            for layer_idx in visible_layers:
                if layer_idx in y_positions:
                    x = layer_positions[layer_idx]
                    y, height = y_positions[layer_idx]
                    
                    rect = plt.Rectangle(
                        (x - 0.02, y), 0.04, height,
                        facecolor=color, edgecolor='black',
                        alpha=0.8, linewidth=0.5
                    )
                    ax.add_patch(rect)
        
        # Add layer labels
        for layer_idx, x_pos in layer_positions.items():
            if layer_idx < len(layers):
                ax.text(
                    x_pos, 0.05,
                    layers[layer_idx],
                    ha='center', va='center',
                    fontsize=8,
                    fontweight='bold',
                    rotation=90
                )
        
        # Create a legend for clusters
        legend_elements = []
        for cluster in unique_clusters[:5]:  # Limit to top 5
            color = cluster_colors.get(cluster, "gray") if cluster_colors else "gray"
            legend_elements.append(
                plt.Line2D([0], [0], 
                          marker='s', 
                          color='w',
                          markerfacecolor=color,
                          markersize=8,
                          label=f"Cluster {cluster}")
            )
        
        # Add the legend
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=8
        )
        
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _create_radial_network(self, ax, visible_links, node_ids, node_clusters, nodes_per_layer, layers, small_font, medium_font, visible_layer_indices, cluster_colors):
        """
        Create a radial network with layers as rings and clusters as segments.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Wedge, ConnectionPatch
        from collections import defaultdict
        
        # Clear the axis
        ax.clear()
        
        # Filter visible layers if specified
        if visible_layer_indices is not None:
            visible_layers = sorted(visible_layer_indices)
        else:
            visible_layers = sorted(range(len(layers)))
        
        # Count nodes by cluster and layer
        cluster_layer_counts = defaultdict(lambda: defaultdict(int))
        
        for node_id, cluster in node_clusters.items():
            if node_id in node_ids:
                node_idx = node_ids.index(node_id)
                layer_idx = node_idx // nodes_per_layer
                
                if layer_idx in visible_layers:
                    cluster_layer_counts[cluster][layer_idx] += 1
        
        # Check if we have any data
        if not cluster_layer_counts:
            ax.text(0.5, 0.5, "No data to display", ha="center", va="center")
            ax.axis("off")
            return
        
        # Get unique clusters
        unique_clusters = sorted(cluster_layer_counts.keys())
        
        # Calculate total nodes per layer
        layer_totals = defaultdict(int)
        for cluster, layer_dict in cluster_layer_counts.items():
            for layer_idx, count in layer_dict.items():
                layer_totals[layer_idx] += count
        
        # Calculate radii for each layer
        layer_radii = {}
        max_radius = 0.8
        min_radius = 0.2
        radius_step = (max_radius - min_radius) / max(1, len(visible_layers) - 1)
        
        for i, layer_idx in enumerate(visible_layers):
            radius = max_radius - i * radius_step
            layer_radii[layer_idx] = radius
        
        # Draw the radial network
        for cluster_idx, cluster in enumerate(unique_clusters):
            # Get a color for this cluster
            color = cluster_colors.get(cluster, "gray") if cluster_colors else "gray"
            
            # Calculate angle range for this cluster
            angle_per_cluster = 360 / len(unique_clusters)
            start_angle = cluster_idx * angle_per_cluster
            end_angle = (cluster_idx + 1) * angle_per_cluster
            
            # Draw wedges for each layer
            for layer_idx in visible_layers:
                if layer_idx in cluster_layer_counts[cluster]:
                    count = cluster_layer_counts[cluster][layer_idx]
                    total = layer_totals[layer_idx]
                    
                    # Calculate the angle span based on the proportion
                    angle_span = angle_per_cluster * (count / total) if total > 0 else 0
                    
                    # Calculate the center angle
                    center_angle = start_angle + angle_per_cluster / 2
                    
                    # Calculate the start and end angles
                    wedge_start = center_angle - angle_span / 2
                    wedge_end = center_angle + angle_span / 2
                    
                    # Draw the wedge
                    radius = layer_radii[layer_idx]
                    width = radius_step * 0.8
                    
                    wedge = Wedge(
                        (0, 0), radius, wedge_start, wedge_end,
                        width=width,
                        facecolor=color, edgecolor='black',
                        alpha=0.8, linewidth=0.5
                    )
                    ax.add_patch(wedge)
                    
                    # Add a label if the wedge is large enough
                    if angle_span > 10:
                        # Calculate the position for the label
                        angle_rad = np.radians(center_angle)
                        label_radius = radius - width / 2
                        x = label_radius * np.cos(angle_rad)
                        y = label_radius * np.sin(angle_rad)
                        
                        ax.text(
                            x, y,
                            str(count),
                            ha='center', va='center',
                            fontsize=8,
                            fontweight='bold'
                        )
        
        # Add layer labels
        for layer_idx, radius in layer_radii.items():
            if layer_idx < len(layers):
                ax.text(
                    0, radius,
                    layers[layer_idx],
                    ha='center', va='center',
                    fontsize=8,
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                )
        
        # Create a legend for clusters
        legend_elements = []
        for cluster in unique_clusters[:5]:  # Limit to top 5
            color = cluster_colors.get(cluster, "gray") if cluster_colors else "gray"
            legend_elements.append(
                plt.Line2D([0], [0], 
                          marker='s', 
                          color='w',
                          markerfacecolor=color,
                          markersize=8,
                          label=f"Cluster {cluster}")
            )
        
        # Add the legend
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=8
        )
        
        # Set axis limits
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal') 

    def _create_tooltips(self):
        """Create detailed tooltips for each tab with technical details"""
        
        # LC1: Heatmap tooltip
        heatmap_tooltip = """
        <h3>LC1: Layer-Cluster Overlap Heatmap</h3>
        
        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.nodes_per_layer: Dictionary mapping layer indices to lists of node IDs</li>
            <li>data_manager.layers: List of layer names</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes using current_node_mask</li>
            <li>Count nodes by cluster and layer using defaultdict(int)</li>
            <li>Create count matrix with shape (num_clusters, num_layers)</li>
            <li>Visualize matrix using matplotlib.pyplot.imshow with 'viridis' colormap</li>
            <li>Add colorbar and annotations showing exact counts</li>
        </ol>
        
        <p><b>Interpretation:</b> Each cell (i,j) shows count of nodes in cluster i and layer j. Darker colors = more nodes.</p>
        """
        self.tab_widget.setTabToolTip(0, heatmap_tooltip)
        
        # LC2: Distribution tooltip
        distribution_tooltip = """
        <h3>LC2: Cluster-Layer Distribution</h3>
        
        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.nodes_per_layer: Dictionary mapping layer indices to lists of node IDs</li>
            <li>data_manager.layers: List of layer names</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
            <li>data_manager.cluster_colors: Dictionary mapping cluster IDs to colors</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes using current_node_mask</li>
            <li>Count nodes by cluster and layer using defaultdict(int)</li>
            <li>For each cluster, calculate percentage of nodes in each layer</li>
            <li>Create stacked bar chart using matplotlib.pyplot.bar with bottom parameter for stacking</li>
            <li>Apply cluster colors from data_manager.cluster_colors</li>
        </ol>
        
        <p><b>Interpretation:</b> Each bar = one cluster. Colored segments = percentage of cluster's nodes in each layer. Full bar height = 100%.</p>
        """
        self.tab_widget.setTabToolTip(1, distribution_tooltip)
        
        # LC3: Layer Distribution tooltip
        layer_distribution_tooltip = """
        <h3>LC3: Layer-Cluster Distribution</h3>
        
        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.nodes_per_layer: Dictionary mapping layer indices to lists of node IDs</li>
            <li>data_manager.layers: List of layer names</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
            <li>data_manager.cluster_colors: Dictionary mapping cluster IDs to colors</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes using current_node_mask</li>
            <li>Count nodes by cluster and layer using defaultdict(int)</li>
            <li>For each layer, calculate percentage of nodes in each cluster</li>
            <li>Create stacked bar chart using matplotlib.pyplot.bar with bottom parameter for stacking</li>
            <li>Apply cluster colors from data_manager.cluster_colors</li>
        </ol>
        
        <p><b>Interpretation:</b> Each bar = one layer. Colored segments = percentage of layer's nodes in each cluster. Full bar height = 100%.</p>
        """
        self.tab_widget.setTabToolTip(2, layer_distribution_tooltip)
        
        # LC4: Network & Similarity tooltip
        network_tooltip = """
        <h3>LC4: Layer-Cluster Network Diagram</h3>
        
        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.nodes_per_layer: Dictionary mapping layer indices to lists of node IDs</li>
            <li>data_manager.layers: List of layer names</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
            <li>data_manager.cluster_colors: Dictionary mapping cluster IDs to colors</li>
            <li>data_manager.layer_colors: Dictionary mapping layer indices to colors</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes using current_node_mask</li>
            <li>Count nodes by cluster and layer using defaultdict(int)</li>
            <li>Create networkx.Graph with nodes for each layer and cluster</li>
            <li>Add edges between layers and clusters with weight = node count</li>
            <li>Apply selected layout algorithm:
                <ul>
                    <li>Community: Force-directed with community detection (Louvain algorithm)</li>
                    <li>Bipartite: Optimized layout with layers on left, clusters on right</li>
                    <li>Circular: Circular arrangement with edge bundling</li>
                    <li>Spectral: Based on graph Laplacian eigendecomposition</li>
                    <li>Spring: Force-directed with weight-based attraction</li>
        </ul>
            </li>
            <li>Draw nodes with size proportional to node count</li>
            <li>Draw edges with width proportional to weight, curved based on weight</li>
            <li>Apply aspect ratio adjustment (0.75-2.0) to layout</li>
        </ol>
        
        <p><b>Interpretation:</b> Nodes = layers/clusters. Node size = node count. Edge width = connection strength. Layout shows structural relationships.</p>
        """
        self.tab_widget.setTabToolTip(3, network_tooltip)
        
        # LC5: Cluster Networks tooltip
        cluster_network_tooltip = """
        <h3>LC5: Cluster Networks</h3>
        
        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.link_pairs: List of tuples (source_id, target_id) for all edges</li>
            <li>data_manager.current_edge_mask: Boolean mask for visible edges</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
            <li>data_manager.cluster_colors: Dictionary mapping cluster IDs to colors</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes and edges using masks</li>
            <li>Create networkx.Graph with nodes for each cluster</li>
            <li>Count connections between clusters using filtered link_pairs</li>
            <li>Add edges between clusters with weight = connection count</li>
            <li>Apply networkx.spring_layout with weight consideration</li>
            <li>Draw nodes with size proportional to node count in cluster</li>
            <li>Draw edges with width proportional to connection count</li>
            <li>Apply cluster colors from data_manager.cluster_colors</li>
        </ol>
        
        <p><b>Interpretation:</b> Nodes = clusters. Node size = node count. Edge width = connection count between clusters. Position shows connectivity structure.</p>
        """
        self.tab_widget.setTabToolTip(4, cluster_network_tooltip)
        
        # LC6: Sankey tooltip
        sankey_tooltip = """
        <h3>LC6: Layer-Cluster Sankey Diagram</h3>
        
        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.nodes_per_layer: Dictionary mapping layer indices to lists of node IDs</li>
            <li>data_manager.layers: List of layer names</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
            <li>data_manager.cluster_colors: Dictionary mapping cluster IDs to colors</li>
            <li>data_manager.layer_colors: Dictionary mapping layer indices to colors</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes using current_node_mask</li>
            <li>Count nodes by cluster and layer using defaultdict(int)</li>
            <li>Create lists for Sankey diagram:
                <ul>
                    <li>labels: Layer names and cluster names</li>
                    <li>colors: Layer colors and cluster colors</li>
                    <li>sources: Index of source node for each flow</li>
                    <li>targets: Index of target node for each flow</li>
                    <li>values: Count of nodes for each flow</li>
                    <li>flow_counts: List of flow values for width calculation</li>
        </ul>
            </li>
            <li>Create Sankey diagram using plotly.graph_objects.Sankey</li>
            <li>Configure node and link properties (colors, hover info)</li>
            <li>Convert to matplotlib figure using plotly.io.to_image</li>
        </ol>
        
        <p><b>Interpretation:</b> Nodes = layers/clusters. Link width = node count. Flow direction shows layer-cluster relationships. Colors match layer/cluster.</p>
        """
        self.tab_widget.setTabToolTip(5, sankey_tooltip)
        
        # LC7: Connectivity Matrix tooltip
        connectivity_tooltip = """
        <h3>LC7: Connectivity Matrices</h3>
        
        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.nodes_per_layer: Dictionary mapping layer indices to lists of node IDs</li>
            <li>data_manager.layers: List of layer names</li>
            <li>data_manager.link_pairs: List of tuples (source_id, target_id) for all edges</li>
            <li>data_manager.current_edge_mask: Boolean mask for visible edges</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
            <li>data_manager.cluster_colors: Dictionary mapping cluster IDs to colors</li>
            <li>data_manager.layer_colors: Dictionary mapping layer indices to colors</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes and edges using masks</li>
            <li>Create six connectivity matrices:
                <ul>
                    <li>Cluster-Cluster (All Edges): Count all connections between clusters</li>
                    <li>Cluster-Cluster (Same Cluster): Count connections between nodes in same cluster</li>
                    <li>Cluster-Cluster (Different Cluster): Count connections between nodes in different clusters</li>
                    <li>Layer-Layer (All Edges): Count all connections between layers</li>
                    <li>Layer-Layer (Same Cluster): Count connections between nodes in same cluster across layers</li>
                    <li>Layer-Layer (Different Cluster): Count connections between nodes in different clusters across layers</li>
        </ul>
            </li>
            <li>For each matrix:
                <ul>
                    <li>Create matrix with shape (num_entities, num_entities)</li>
                    <li>Iterate through visible links and increment matrix cells based on edge type</li>
                    <li>Visualize matrix using matplotlib.pyplot.imshow with 'viridis' colormap</li>
                    <li>Add colorbar and annotations showing exact counts</li>
        </ul>
            </li>
        </ol>
        
        <p><b>Interpretation:</b> Each cell (i,j) shows connection count between entities i and j. Darker colors = more connections. Diagonal = internal connections.</p>
        """
        self.tab_widget.setTabToolTip(6, connectivity_tooltip)
        
        # LC8: Chord Diagram tooltip
        chord_tooltip = """
        <h3>LC8: Layer-Cluster Chord Diagram</h3>
        
        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.nodes_per_layer: Dictionary mapping layer indices to lists of node IDs</li>
            <li>data_manager.layers: List of layer names</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
            <li>data_manager.cluster_colors: Dictionary mapping cluster IDs to colors</li>
            <li>data_manager.layer_colors: Dictionary mapping layer indices to colors</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes using current_node_mask</li>
            <li>Count nodes by cluster and layer using defaultdict(int)</li>
            <li>Create connection matrix of size (num_layers + num_clusters) × (num_layers + num_clusters)</li>
            <li>Fill matrix with connection counts between layers and clusters</li>
            <li>Create entity colors by combining layer_colors and cluster_colors</li>
            <li>Draw circular arcs for each entity using matplotlib.patches.Wedge</li>
            <li>Draw connections between entities using matplotlib.patches.Bezier</li>
            <li>Connection width proportional to connection count</li>
            <li>Add labels for layers and clusters</li>
        </ol>
        
        <p><b>Interpretation:</b> Arcs = layers/clusters. Arc length = node count. Connections = relationships. Width = connection strength. Colors match layer/cluster.</p>
        """
        self.tab_widget.setTabToolTip(7, chord_tooltip)
        
        # LC9: Density Heatmap tooltip
        density_tooltip = """
        <h3>LC9: Layer-Cluster Density Heatmap</h3>
        
        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.nodes_per_layer: Dictionary mapping layer indices to lists of node IDs</li>
            <li>data_manager.layers: List of layer names</li>
            <li>data_manager.link_pairs: List of tuples (source_id, target_id) for all edges</li>
            <li>data_manager.current_edge_mask: Boolean mask for visible edges</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes and edges using masks</li>
            <li>Count nodes by cluster and layer using defaultdict(int)</li>
            <li>Create density matrix with shape (num_clusters, num_layers)</li>
            <li>For each cluster-layer pair:
                <ul>
                    <li>Count actual connections between nodes in the cluster and layer</li>
                    <li>Calculate maximum possible connections = n × (n-1) / 2 where n = node count</li>
                    <li>Compute density = actual / maximum (or 0 if maximum = 0)</li>
        </ul>
            </li>
            <li>Visualize density matrix using matplotlib.pyplot.imshow with 'viridis' colormap</li>
            <li>Add colorbar and annotations showing density values</li>
        </ol>
        
        <p><b>Interpretation:</b> Each cell (i,j) shows connection density in cluster i, layer j. Values range 0-1. Darker colors = higher density (more connected).</p>
        """
        self.tab_widget.setTabToolTip(8, density_tooltip)
        
        # LC10: Co-occurrence Network tooltip
        cooccurrence_tooltip = """
        <h3>LC10: Cluster Co-occurrence Network</h3>
        
        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.nodes_per_layer: Dictionary mapping layer indices to lists of node IDs</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
            <li>data_manager.cluster_colors: Dictionary mapping cluster IDs to colors</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes using current_node_mask</li>
            <li>Identify clusters present in each layer</li>
            <li>Create networkx.Graph with nodes for each cluster</li>
            <li>For each pair of clusters:
                <ul>
                    <li>Count layers where both clusters appear</li>
                    <li>Add edge with weight = co-occurrence count if > 0</li>
        </ul>
            </li>
            <li>Apply networkx.spring_layout with weight consideration</li>
            <li>Draw nodes with size proportional to total node count in cluster</li>
            <li>Draw edges with width proportional to co-occurrence count</li>
            <li>Apply cluster colors from data_manager.cluster_colors</li>
        </ol>
        
        <p><b>Interpretation:</b> Nodes = clusters. Node size = node count. Edge width = number of layers where both clusters appear. Position shows co-occurrence patterns.</p>
        """
        self.tab_widget.setTabToolTip(9, cooccurrence_tooltip)
        
        # LC11: Normalized Heatmap tooltip
        normalized_tooltip = """
        <h3>LC11: Normalized Layer-Cluster Heatmap</h3>
        
        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.nodes_per_layer: Dictionary mapping layer indices to lists of node IDs</li>
            <li>data_manager.layers: List of layer names</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes using current_node_mask</li>
            <li>Count nodes by cluster and layer using defaultdict(int)</li>
            <li>Create count matrix with shape (num_clusters, num_layers)</li>
            <li>Create normalized matrix with same shape</li>
            <li>For each layer:
                <ul>
                    <li>Calculate total nodes in layer</li>
                    <li>Normalize each cluster's count by dividing by total (percentage)</li>
        </ul>
            </li>
            <li>Visualize normalized matrix using matplotlib.pyplot.imshow with 'viridis' colormap</li>
            <li>Add colorbar and annotations showing percentage values</li>
        </ol>
        
        <p><b>Interpretation:</b> Each cell (i,j) shows percentage of layer j's nodes in cluster i. Values sum to 1.0 per column. Darker colors = higher percentage.</p>
        """
        self.tab_widget.setTabToolTip(10, normalized_tooltip)
        
        # LC12: Similarity Matrix tooltip
        similarity_tooltip = """
        <h3>LC12: Cluster Similarity Matrix</h3>
        
        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.nodes_per_layer: Dictionary mapping layer indices to lists of node IDs</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes using current_node_mask</li>
            <li>For each cluster, create binary vector indicating presence in each layer</li>
            <li>Create similarity matrix with shape (num_clusters, num_clusters)</li>
            <li>For each pair of clusters (i,j):
                <ul>
                    <li>Calculate Jaccard similarity: |intersection| / |union|</li>
                    <li>Store similarity value in matrix[i,j]</li>
        </ul>
            </li>
            <li>Visualize similarity matrix using matplotlib.pyplot.imshow with 'viridis' colormap</li>
            <li>Add colorbar and annotations showing similarity values</li>
        </ol>
        
        <p><b>Interpretation:</b> Each cell (i,j) shows Jaccard similarity between clusters i and j. Values range 0-1. Diagonal = 1.0 (self-similarity). Darker colors = more similar.</p>
        """
        self.tab_widget.setTabToolTip(11, similarity_tooltip)
        
        # LC13: Bubble Chart tooltip
        bubble_tooltip = """
        <h3>LC13: Layer-Cluster Bubble Chart</h3>
        
        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.nodes_per_layer: Dictionary mapping layer indices to lists of node IDs</li>
            <li>data_manager.layers: List of layer names</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
            <li>data_manager.cluster_colors: Dictionary mapping cluster IDs to colors</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes using current_node_mask</li>
            <li>Count nodes by cluster and layer using defaultdict(int)</li>
            <li>Create data arrays for bubble chart:
                <ul>
                    <li>x: Layer indices for each bubble</li>
                    <li>y: Cluster indices for each bubble</li>
                    <li>sizes: Node counts for each bubble (scaled for visualization)</li>
                    <li>colors: Cluster colors for each bubble</li>
        </ul>
            </li>
            <li>Draw bubbles using matplotlib.pyplot.scatter with size parameter</li>
            <li>Add labels for layers and clusters</li>
            <li>Add annotations showing exact counts</li>
        </ol>
        
        <p><b>Interpretation:</b> Each bubble represents cluster-layer combination. Position = (layer, cluster). Bubble size = node count. Color = cluster color.</p>
        """
        self.tab_widget.setTabToolTip(12, bubble_tooltip)
        
        # LC14: Treemap tooltip
        treemap_tooltip = """
        <h3>LC14: Layer-Cluster Treemap</h3>
        
        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.nodes_per_layer: Dictionary mapping layer indices to lists of node IDs</li>
            <li>data_manager.layers: List of layer names</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
            <li>data_manager.cluster_colors: Dictionary mapping cluster IDs to colors</li>
            <li>data_manager.layer_colors: Dictionary mapping layer indices to colors</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes using current_node_mask</li>
            <li>Count nodes by cluster and layer using defaultdict(int)</li>
            <li>Create lists for treemap:
                <ul>
                    <li>labels: Cluster-layer combination names</li>
                    <li>parents: Cluster names for each combination</li>
                    <li>values: Node counts for each combination</li>
                    <li>colors: Colors for each combination (derived from cluster colors)</li>
        </ul>
            </li>
            <li>Create treemap using matplotlib_squarify:
                <ul>
                    <li>Normalize sizes to fit available space</li>
                    <li>Calculate rectangle positions using squarify algorithm</li>
                    <li>Draw rectangles using matplotlib.patches.Rectangle</li>
                    <li>Add labels with size proportional to rectangle size</li>
                </ul>
            </li>
        </ol>
        
        <p><b>Interpretation:</b> Each rectangle = cluster-layer combination. Rectangle size = node count. Color = cluster color. Hierarchy: clusters contain layers.</p>
        """
        self.tab_widget.setTabToolTip(13, treemap_tooltip)
        
        # LC15: Flow Ideas tooltip
        flow_ideas_tooltip = """
        <h3>LC15: Flow Visualization Ideas</h3>
        
        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.nodes_per_layer: Dictionary mapping layer indices to lists of node IDs</li>
            <li>data_manager.layers: List of layer names</li>
            <li>data_manager.link_pairs: List of tuples (source_id, target_id) for all edges</li>
            <li>data_manager.current_edge_mask: Boolean mask for visible edges</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
            <li>data_manager.cluster_colors: Dictionary mapping cluster IDs to colors</li>
            <li>data_manager.layer_colors: Dictionary mapping layer indices to colors</li>
        </ul>
        
        <p><b>Visualizations:</b></p>
        <ol>
            <li><b>Circular Flow Diagram:</b>
                <ul>
                    <li>Data: Node counts and connections between clusters and layers</li>
                    <li>Method: Arrange entities in circle, draw curved connections with width proportional to connection strength</li>
                    <li>Result: Shows layer-cluster relationships</li>
        </ul>
            </li>
            <li><b>Force-Directed Regions:</b>
                <ul>
                    <li>Data: Layer-cluster relationships</li>
                    <li>Method: Arrange clusters in regions based on layer</li>
                    <li>Result: Shows layer-cluster relationships</li>
        </ul>
            </li>
            <li><b>Alluvial Diagram:</b>
                <ul>
                    <li>Data: Layer-cluster relationships</li>
                    <li>Method: Show flow between layers and clusters</li>
                    <li>Result: Shows layer-cluster relationships</li>
        </ul>
            </li>
            <li><b>Radial Network:</b>
                <ul>
                    <li>Data: Layer-cluster relationships</li>
                    <li>Method: Show layers as rings and clusters as segments</li>
                    <li>Result: Shows layer-cluster relationships</li>
        </ul>
            </li>
        </ol>
        """
        self.tab_widget.setTabToolTip(14, flow_ideas_tooltip) 