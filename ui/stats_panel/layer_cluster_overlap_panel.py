import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTabWidget,
    QLabel,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QGridLayout,
    QDoubleSpinBox,
    QPushButton,
)
from PyQt5.QtCore import Qt

from ui.stats_panel.base_panel import BaseStatsPanel
from charts.layer_cluster.lc1_overlap_heatmap import (
    create_layer_cluster_overlap_heatmap,
)
from charts.layer_cluster.lc2_cluster_layer_distribution import (
    create_cluster_layer_distribution,
)
from charts.layer_cluster.lc3_layer_cluster_distribution import (
    create_layer_cluster_distribution,
)
from charts.layer_cluster.lc4_network_diagram import (
    create_layer_cluster_network_diagram,
)
from charts.layer_cluster.lc4a_network_diagram import (
    create_enhanced_layer_cluster_network_diagram,
    create_enhanced_network_ui,
    update_enhanced_network,
)
from charts.layer_cluster.lc5_cluster_network import create_cluster_network
from charts.layer_cluster.lc10_cooccurrence_network import (
    create_cluster_cooccurrence_network,
)
from charts.layer_cluster.lc6_sankey_diagram import create_layer_cluster_sankey
from charts.layer_cluster.lc7_connectivity_matrix import (
    create_cluster_connectivity_matrix,
    create_layer_connectivity_matrix,
)
from charts.layer_cluster.lc8_chord_diagram import create_layer_cluster_chord
from charts.layer_cluster.lc9_density_heatmap import (
    create_layer_cluster_density_heatmap,
)
from charts.layer_cluster.lc11_normalized_heatmap import (
    create_layer_cluster_normalized_heatmap,
)
from charts.layer_cluster.lc12_similarity_matrix import create_cluster_similarity_matrix
from charts.layer_cluster.lc12_enhanced_similarity import (
    create_enhanced_cluster_similarity,
)
from charts.layer_cluster.lc13_bubble_chart import create_layer_cluster_bubble_chart
from charts.layer_cluster.lc14_treemap import create_layer_cluster_treemap
from charts.layer_cluster.lc16_interlayer_paths import (
    create_interlayer_path_analysis,
    create_lc16_ui_elements,
    integrate_lc16_ui_with_panel,
    get_selected_cluster,
    get_lc16_visualization_settings,
)
from charts.layer_cluster.lc17_cluster_bridging_analysis import (
    create_cluster_bridging_analysis,
)
from charts.layer_cluster.lc20_interlayer_path_similarity import (
    create_interlayer_path_similarity,
)
from utils.calc_community import AVAILABLE_COMMUNITY_ALGORITHMS


class LayerClusterOverlapPanel(BaseStatsPanel):
    """Panel for visualizing layer and cluster overlaps"""

    def setup_ui(self):
        """Set up the UI components"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

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
        self.layout_algorithm_combo.addItems(
            ["Community", "Bipartite", "Circular", "Spectral", "Spring"]
        )
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

        # Remove LC16 Analysis dropdown from top controls
        # It will be moved to the LC16 tab
        self.path_analysis_combo = QComboBox()
        self.path_analysis_combo.addItems(["Path Length", "Betweenness", "Bottleneck"])
        self.path_analysis_combo.currentIndexChanged.connect(self.on_layout_changed)

        # Add analysis type dropdown for LC17
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QLabel("LC17 Analysis:"))
        self.bridge_analysis_combo = QComboBox()
        self.bridge_analysis_combo.addItems(
            ["Bridge Score", "Flow Efficiency", "Layer Span"]
        )
        self.bridge_analysis_combo.currentIndexChanged.connect(self.on_layout_changed)
        controls_layout.addWidget(self.bridge_analysis_combo)

        # Add similarity metric dropdown for LC12
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QLabel("LC12 Metric:"))
        self.similarity_metric_combo = QComboBox()
        self.similarity_metric_combo.addItems(
            [
                "All Metrics",
                "Jaccard",
                "Cosine",
                "Overlap",
                "Connection",
                "Layer Distribution",
                "Hierarchical",
                "Node Sharing",
                "Path-based",
                "Mutual Information",
            ]
        )
        self.similarity_metric_combo.currentIndexChanged.connect(self.on_layout_changed)
        controls_layout.addWidget(self.similarity_metric_combo)

        # Add edge type dropdown for LC12
        controls_layout.addSpacing(10)
        controls_layout.addWidget(QLabel("Edge Type:"))
        self.edge_type_combo = QComboBox()
        self.edge_type_combo.addItems(["All Edges", "Interlayer", "Intralayer"])
        self.edge_type_combo.currentIndexChanged.connect(self.on_layout_changed)
        controls_layout.addWidget(self.edge_type_combo)

        # Add cluster selector for LC20
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QLabel("LC20/16 Cluster:"))
        self.path_similarity_cluster_combo = QComboBox()
        self.path_similarity_cluster_combo.addItem("All Clusters")
        self.path_similarity_cluster_combo.currentIndexChanged.connect(
            self.on_layout_changed
        )
        controls_layout.addWidget(self.path_similarity_cluster_combo)

        # LC4A: Enhanced Network Diagram controls - moved to the LC4A tab itself
        # Remove LC16 UI elements integration from top controls
        # It will be moved to the LC16 tab
        self.enhanced_network_ui = create_enhanced_network_ui()
        self.edge_counting_combo = self.enhanced_network_ui["edge_counting_combo"]
        self.community_algorithm_combo = self.enhanced_network_ui[
            "community_algorithm_combo"
        ]
        self.community_resolution_spin = self.enhanced_network_ui[
            "community_resolution_spin"
        ]
        self.show_edge_weights_check = self.enhanced_network_ui[
            "show_edge_weights_check"
        ]
        self.show_node_labels_check = self.enhanced_network_ui["show_node_labels_check"]
        self.show_legend_check = self.enhanced_network_ui["show_legend_check"]
        self.update_enhanced_network_btn = self.enhanced_network_ui["update_btn"]
        self.update_enhanced_network_btn.clicked.connect(self.update_enhanced_network)

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
        self.network_ax1 = self.network_figure.add_subplot(
            111
        )  # Full plot instead of 121

        # LC5: Cluster network with co-occurrence network
        self.cluster_network_figure = Figure(figsize=(10, 8), dpi=100)
        self.cluster_network_canvas = FigureCanvas(self.cluster_network_figure)
        self.cluster_network_ax1 = self.cluster_network_figure.add_subplot(
            121
        )  # Left subplot
        self.cluster_network_ax2 = self.cluster_network_figure.add_subplot(
            122
        )  # Right subplot

        # LC6: Sankey diagram
        self.sankey_figure = Figure(figsize=(10, 6), dpi=100)
        self.sankey_canvas = FigureCanvas(self.sankey_figure)
        self.sankey_ax = self.sankey_figure.add_subplot(111)

        # LC7: Cluster connectivity matrix
        self.connectivity_figure = Figure(
            figsize=(15, 12), dpi=100
        )  # Taller figure for 6 subplots (2 rows, 3 columns)
        self.connectivity_canvas = FigureCanvas(self.connectivity_figure)
        # Cluster connectivity matrices (top row)
        self.connectivity_ax1 = self.connectivity_figure.add_subplot(231)  # Top-left
        self.connectivity_ax2 = self.connectivity_figure.add_subplot(232)  # Top-middle
        self.connectivity_ax3 = self.connectivity_figure.add_subplot(233)  # Top-right
        # Layer connectivity matrices (bottom row)
        self.connectivity_ax4 = self.connectivity_figure.add_subplot(234)  # Bottom-left
        self.connectivity_ax5 = self.connectivity_figure.add_subplot(
            235
        )  # Bottom-middle
        self.connectivity_ax6 = self.connectivity_figure.add_subplot(
            236
        )  # Bottom-right

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

        # LC12: Similarity Matrix
        self.similarity_figure = Figure(figsize=(8, 6), dpi=100)
        self.similarity_canvas = FigureCanvas(self.similarity_figure)
        self.similarity_ax = self.similarity_figure.add_subplot(111)

        # LC13: Bubble Chart
        self.bubble_figure = Figure(figsize=(10, 8), dpi=100)
        self.bubble_canvas = FigureCanvas(self.bubble_figure)
        self.bubble_ax = self.bubble_figure.add_subplot(111)

        # LC14: Treemap
        self.treemap_figure = Figure(figsize=(10, 8), dpi=100)
        self.treemap_canvas = FigureCanvas(self.treemap_figure)
        self.treemap_ax = self.treemap_figure.add_subplot(111)

        # Create a container widget for the LC14 tab
        self.lc14_container = QWidget()
        lc14_layout = QVBoxLayout(self.lc14_container)

        # Add controls for LC14
        lc14_controls = QHBoxLayout()

        # Add dropdown for count type
        lc14_count_type_label = QLabel("Count Type:")
        self.lc14_count_type_combo = QComboBox()
        self.lc14_count_type_combo.addItem("Nodes")
        self.lc14_count_type_combo.addItem("Intralayer Edges")
        self.lc14_count_type_combo.currentIndexChanged.connect(
            self.on_lc14_count_type_changed
        )

        lc14_controls.addWidget(lc14_count_type_label)
        lc14_controls.addWidget(self.lc14_count_type_combo)
        lc14_controls.addStretch()

        # Add controls and canvas to the layout
        lc14_layout.addLayout(lc14_controls)
        lc14_layout.addWidget(self.treemap_canvas)

        # LC15: Flow Visualization Ideas
        self.flow_ideas_figure = Figure(figsize=(12, 10), dpi=100)
        self.flow_ideas_canvas = FigureCanvas(self.flow_ideas_figure)

        # Create a container widget for the LC15 tab
        lc15_container = QWidget()
        lc15_layout = QVBoxLayout(lc15_container)

        # Add enable/disable checkbox for LC15
        lc15_controls_layout = QHBoxLayout()
        self.flow_ideas_enable_checkbox = QCheckBox("Enable")
        self.flow_ideas_enable_checkbox.setChecked(False)  # Disabled by default
        self.flow_ideas_enable_checkbox.stateChanged.connect(
            self.on_flow_ideas_state_changed
        )
        lc15_controls_layout.addWidget(self.flow_ideas_enable_checkbox)
        lc15_controls_layout.addStretch()
        lc15_layout.addLayout(lc15_controls_layout)

        # Add the canvas to the LC15 tab
        lc15_layout.addWidget(self.flow_ideas_canvas)

        self.flow_ideas_ax1 = self.flow_ideas_figure.add_subplot(221)  # Top-left
        self.flow_ideas_ax2 = self.flow_ideas_figure.add_subplot(222)  # Top-right
        self.flow_ideas_ax3 = self.flow_ideas_figure.add_subplot(223)  # Bottom-left
        self.flow_ideas_ax4 = self.flow_ideas_figure.add_subplot(224)  # Bottom-right

        # Show disabled message initially
        for ax in [
            self.flow_ideas_ax1,
            self.flow_ideas_ax2,
            self.flow_ideas_ax3,
            self.flow_ideas_ax4,
        ]:
            ax.clear()
            ax.text(
                0.5,
                0.5,
                "Flow ideas visualization disabled",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        self.flow_ideas_canvas.draw()

        # Create figures for the new LC16 and LC17 tabs
        self.path_analysis_figure = Figure(figsize=(12, 10), dpi=100)
        self.path_analysis_canvas = FigureCanvas(self.path_analysis_figure)

        self.bridge_analysis_figure = Figure(figsize=(12, 10), dpi=100)
        self.bridge_analysis_canvas = FigureCanvas(self.bridge_analysis_figure)

        # Create figure for the new LC20 tab
        self.interlayer_path_similarity_figure = Figure(figsize=(12, 10), dpi=100)
        self.interlayer_path_similarity_canvas = FigureCanvas(
            self.interlayer_path_similarity_figure
        )

        # Add each canvas to a tab
        self.tab_widget.addTab(self.heatmap_canvas, "LC1")
        self.tab_widget.addTab(self.distribution_canvas, "LC2")
        self.tab_widget.addTab(self.layer_distribution_canvas, "LC3")
        self.tab_widget.addTab(self.network_canvas, "LC4")
        self.tab_widget.addTab(self.cluster_network_canvas, "LC5")
        self.tab_widget.addTab(self.sankey_canvas, "LC6")
        self.tab_widget.addTab(self.connectivity_canvas, "LC7")
        self.tab_widget.addTab(self.chord_canvas, "LC8")
        self.tab_widget.addTab(self.density_canvas, "LC9")
        self.tab_widget.addTab(self.cooccurrence_canvas, "LC10")
        self.tab_widget.addTab(self.normalized_canvas, "LC11")
        self.tab_widget.addTab(self.similarity_canvas, "LC12:")
        self.tab_widget.addTab(self.bubble_canvas, "LC13")
        self.tab_widget.addTab(self.lc14_container, "LC14")
        self.tab_widget.addTab(lc15_container, "LC15")
        
        # Create a container widget for the LC16 tab
        lc16_container = QWidget()
        lc16_layout = QVBoxLayout(lc16_container)
        
        # Add controls layout for LC16
        lc16_controls_layout = QHBoxLayout()
        lc16_controls_layout.addWidget(QLabel("LC16 Analysis:"))
        lc16_controls_layout.addWidget(self.path_analysis_combo)
        lc16_controls_layout.addStretch()
        lc16_layout.addLayout(lc16_controls_layout)
        
        # Add LC16 visualization controls using the integration function
        lc16_ui_elements = integrate_lc16_ui_with_panel(
            self, 
            lc16_layout,  # Add to the LC16 tab layout instead of the top controls
            analysis_combo=self.path_analysis_combo,
            cluster_combo=self.path_similarity_cluster_combo
        )
        
        # Store references to the UI elements for later use
        self.lc16_viz_style_combo = lc16_ui_elements["viz_style_combo"]
        self.lc16_show_labels_cb = lc16_ui_elements["show_labels_checkbox"]
        self.lc16_show_nodes_cb = lc16_ui_elements["show_nodes_checkbox"]
        self.lc16_color_by_centrality_cb = lc16_ui_elements["color_by_centrality_checkbox"]
        
        # Add the canvas to the LC16 tab
        lc16_layout.addWidget(self.path_analysis_canvas)
        
        # Add the LC16 tab to the tab widget
        self.tab_widget.addTab(lc16_container, "LC16")
        
        self.tab_widget.addTab(self.bridge_analysis_canvas, "LC17")
        self.tab_widget.addTab(self.interlayer_path_similarity_canvas, "LC20")

        # LC4A: Enhanced Network Diagram
        self.enhanced_network_figure = Figure(figsize=(8, 6), dpi=100)
        self.enhanced_network_canvas = FigureCanvas(self.enhanced_network_figure)
        self.enhanced_network_ax = self.enhanced_network_figure.add_subplot(111)

        # Show disabled message initially
        self.enhanced_network_ax.text(
            0.5,
            0.5,
            "Enhanced network diagram disabled",
            ha="center",
            va="center",
            fontsize=12,
        )
        self.enhanced_network_canvas.draw()

        # Create a container widget for the LC4A tab
        lc4a_container = QWidget()
        lc4a_layout = QVBoxLayout(lc4a_container)

        # Add enable/disable checkbox for LC4A
        lc4a_controls_layout = QHBoxLayout()
        self.enhanced_network_enable_checkbox = QCheckBox("Enable")
        self.enhanced_network_enable_checkbox.setChecked(False)  # Disabled by default
        self.enhanced_network_enable_checkbox.stateChanged.connect(
            self.on_enhanced_network_state_changed
        )
        lc4a_controls_layout.addWidget(self.enhanced_network_enable_checkbox)
        lc4a_controls_layout.addStretch()
        lc4a_layout.addLayout(lc4a_controls_layout)

        # Add the enhanced network controls to the LC4A tab
        lc4a_layout.addWidget(self.enhanced_network_ui["group"])

        # Initially disable the controls since the checkbox is unchecked by default
        self.enhanced_network_ui["group"].setEnabled(False)

        # Add the canvas to the LC4A tab
        lc4a_layout.addWidget(self.enhanced_network_canvas)

        # Add the LC4A tab to the tab widget
        self.tab_widget.addTab(lc4a_container, "LC4A: Enhanced Network Diagram")

        # Create tooltips for each tab
        self._create_tooltips()

        # Set the layout for the widget
        self.setLayout(layout)

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
            for fig in [
                self.heatmap_figure,
                self.distribution_figure,
                self.layer_distribution_figure,
                self.network_figure,
                self.cluster_network_figure,
                self.sankey_figure,
                self.connectivity_figure,
                self.chord_figure,
                self.density_figure,
                self.cooccurrence_figure,
                self.normalized_figure,
                self.similarity_figure,
                self.bubble_figure,
                self.treemap_figure,
                self.flow_ideas_figure,
            ]:
                fig.text(
                    0.5, 0.5, "Charts disabled", ha="center", va="center", fontsize=14
                )

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
            try:
                # Determine which control was changed and update the corresponding visualization
                sender = self.sender()

                # Update based on which control was changed
                if (
                    sender == self.layout_algorithm_combo
                    or sender == self.aspect_ratio_combo
                ):
                    self.update_lc4_network_diagram(self._current_data)
                elif sender == self.path_analysis_combo:
                    self.update_lc16_path_analysis(self._current_data)
                elif sender == self.bridge_analysis_combo:
                    self.update_lc17_bridge_analysis(self._current_data)
                elif (
                    sender == self.similarity_metric_combo
                    or sender == self.edge_type_combo
                ):
                    self.update_lc12_similarity_matrix(self._current_data)
                elif sender == self.path_similarity_cluster_combo:
                    # Update both LC20 and LC16 when the cluster dropdown changes
                    self.update_lc20_interlayer_path_similarity(self._current_data)
                    self.update_lc16_path_analysis(self._current_data)
                else:
                    # If we can't determine the sender, update based on current tab
                    current_tab = self.tab_widget.currentIndex()

                    # Map tab indices to update methods
                    tab_to_update = {
                        0: self.update_lc1_heatmap,  # LC1: Overlap Heatmap
                        1: self.update_lc2_distribution,  # LC2: Cluster-Layer Distribution
                        2: self.update_lc3_layer_distribution,  # LC3: Layer-Cluster Distribution
                        3: self.update_lc4_network_diagram,  # LC4: Network Diagram
                        4: self.update_lc5_cluster_network,  # LC5: Cluster Network
                        5: self.update_lc6_sankey,  # LC6: Sankey Diagram
                        6: self.update_lc7_connectivity,  # LC7: Connectivity Matrix
                        7: self.update_lc8_chord,  # LC8: Chord Diagram
                        8: self.update_lc9_density,  # LC9: Density Heatmap
                        9: self.update_lc10_cooccurrence,  # LC10: Co-occurrence Network
                        10: self.update_lc11_normalized,  # LC11: Normalized Heatmap
                        11: self.update_lc12_similarity_matrix,  # LC12: Similarity Matrix
                        12: self.update_lc13_bubble,  # LC13: Bubble Chart
                        13: self.update_lc14_treemap,  # LC14: Treemap
                        14: self.update_flow_ideas,  # LC15: Flow Ideas
                        15: self.update_lc16_path_analysis,  # LC16: Interlayer Path Analysis
                        16: self.update_lc17_bridge_analysis,  # LC17: Cluster Bridging Analysis
                        17: self.update_lc20_interlayer_path_similarity,  # LC20: Interlayer Path Similarity
                        18: self.update_enhanced_network,  # LC4A: Enhanced Network Diagram
                    }

                    # Call the appropriate update method if the current tab is in our map
                    if current_tab in tab_to_update:
                        tab_to_update[current_tab](self._current_data)
            except Exception as e:
                logging.error(f"Error in on_layout_changed: {str(e)}")
                import traceback

                logging.error(traceback.format_exc())

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
            visible_links = [
                data_manager.link_pairs[i]
                for i in range(len(data_manager.link_pairs))
                if data_manager.current_edge_mask[i]
            ]

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
            aspect_ratio=aspect_ratio,
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
        self.network_ax1 = self.network_figure.add_subplot(
            111
        )  # Full plot instead of 121
        self.cluster_network_ax1 = self.cluster_network_figure.add_subplot(121)
        self.cluster_network_ax2 = self.cluster_network_figure.add_subplot(122)
        self.sankey_ax = self.sankey_figure.add_subplot(111)
        # Cluster connectivity matrices (top row)
        self.connectivity_ax1 = self.connectivity_figure.add_subplot(231)  # Top-left
        self.connectivity_ax2 = self.connectivity_figure.add_subplot(232)  # Top-middle
        self.connectivity_ax3 = self.connectivity_figure.add_subplot(233)  # Top-right
        # Layer connectivity matrices (bottom row)
        self.connectivity_ax4 = self.connectivity_figure.add_subplot(234)  # Bottom-left
        self.connectivity_ax5 = self.connectivity_figure.add_subplot(
            235
        )  # Bottom-middle
        self.connectivity_ax6 = self.connectivity_figure.add_subplot(
            236
        )  # Bottom-right
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
        node_positions = filtered_data["node_positions"]
        node_colors = filtered_data["node_colors"]
        edge_connections = filtered_data["edge_connections"]

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
            data_manager.cluster_colors,
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
            data_manager.cluster_colors,
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
            data_manager.cluster_colors,
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
            aspect_ratio=aspect_ratio,
        )

        # Set title for the network diagram
        self.network_ax1.set_title(
            "Layer-Cluster Network", fontsize=medium_font["fontsize"]
        )
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
            visible_layer_indices,
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
            data_manager.cluster_colors,
        )

        # Set titles for the subplots
        self.cluster_network_ax1.set_title(
            "Cluster Network", fontsize=medium_font["fontsize"]
        )
        self.cluster_network_ax2.set_title(
            "Cluster Co-occurrence Network", fontsize=medium_font["fontsize"]
        )
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
            visible_layer_indices,
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
            edge_type="all",
        )
        self.connectivity_ax1.set_title(
            "All Connections\nBetween Clusters", fontsize=medium_font["fontsize"]
        )

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
            edge_type="same_layer",
        )
        self.connectivity_ax2.set_title(
            "Same Layer Connections\nBetween Clusters", fontsize=medium_font["fontsize"]
        )

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
            edge_type="interlayer",
        )
        self.connectivity_ax3.set_title(
            "Interlayer Connections\nBetween Clusters", fontsize=medium_font["fontsize"]
        )

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
            edge_type="all",
        )
        self.connectivity_ax4.set_title(
            "All Connections\nBetween Layers", fontsize=medium_font["fontsize"]
        )

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
            edge_type="same_cluster",
        )
        self.connectivity_ax5.set_title(
            "Same Cluster Connections\nBetween Layers", fontsize=medium_font["fontsize"]
        )

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
            edge_type="different_cluster",
        )
        self.connectivity_ax6.set_title(
            "Different Cluster Connections\nBetween Layers",
            fontsize=medium_font["fontsize"],
        )

        # Add a super title for the entire figure
        self.connectivity_figure.suptitle(
            "Connectivity Matrices\nTop: Cluster-to-Cluster | Bottom: Layer-to-Layer",
            fontsize=medium_font["fontsize"] + 2,
            y=0.98,
        )

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
            data_manager.cluster_colors,
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
            data_manager.cluster_colors,
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
            data_manager.cluster_colors,
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
            data_manager.cluster_colors,
        )

        # LC12: Create cluster similarity matrix
        self.update_lc12_similarity_matrix(data_manager)

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
            data_manager.cluster_colors,
        )

        # LC14: Create treemap
        self.update_lc14_treemap(data_manager)

        # LC15: Update flow ideas visualization
        self.update_flow_ideas(data_manager)

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

        # Update LC16: Interlayer Path Analysis
        self.update_lc16_path_analysis(data_manager)

        # Update LC17: Cluster Bridging Analysis
        self.update_lc17_bridge_analysis(data_manager)

        # LC4A: Update Enhanced Network Diagram
        self.update_enhanced_network()

        # Update LC20: Interlayer Path Similarity
        self.update_lc20_interlayer_path_similarity(data_manager)

        # Update the cluster selector for LC20 with available clusters
        self.update_cluster_selector(data_manager)

    def update_lc16_path_analysis(self, data_manager):
        """Update the LC16 interlayer path analysis with the current analysis type"""
        if (
            not hasattr(self, "path_analysis_figure")
            or not self.enable_checkbox.isChecked()
        ):
            return

        # Clear the figure and recreate the axis
        self.path_analysis_figure.clear()
        self.path_analysis_ax = self.path_analysis_figure.add_subplot(111)

        # Get the selected analysis type
        analysis_type = self.path_analysis_combo.currentText().lower().replace(" ", "_")

        # Get the selected cluster and visualization settings from the LC16 module
        selected_cluster = get_selected_cluster(self)
        viz_settings = get_lc16_visualization_settings(self)

        # Get visible links
        visible_links = []
        if data_manager.current_edge_mask is not None:
            visible_links = [
                data_manager.link_pairs[i]
                for i in range(len(data_manager.link_pairs))
                if data_manager.current_edge_mask[i]
            ]

        # Get visible layer indices
        visible_layer_indices = []
        if hasattr(data_manager, "visible_layer_indices"):
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
            analysis_type,
            selected_cluster=selected_cluster,
            viz_style=viz_settings["viz_style"],
            show_labels=viz_settings["show_labels"],
            show_nodes=viz_settings["show_nodes"],
            color_by_centrality=viz_settings["color_by_centrality"],
        )

        # Apply tight layout and draw
        self.path_analysis_figure.tight_layout()
        self.path_analysis_canvas.draw()

    def update_lc17_bridge_analysis(self, data_manager):
        """Update the LC17 cluster bridging analysis with the current analysis type"""
        if (
            not hasattr(self, "bridge_analysis_figure")
            or not self.enable_checkbox.isChecked()
        ):
            return

        # Clear the figure and recreate the axis
        self.bridge_analysis_figure.clear()
        self.bridge_analysis_ax = self.bridge_analysis_figure.add_subplot(111)

        # Get the selected analysis type
        analysis_type = (
            self.bridge_analysis_combo.currentText().lower().replace(" ", "_")
        )

        # Get visible links
        visible_links = []
        if data_manager.current_edge_mask is not None:
            visible_links = [
                data_manager.link_pairs[i]
                for i in range(len(data_manager.link_pairs))
                if data_manager.current_edge_mask[i]
            ]

        # Get visible layer indices
        visible_layer_indices = []
        if hasattr(data_manager, "visible_layer_indices"):
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
            analysis_type,
        )

        # Draw the canvas
        self.bridge_analysis_canvas.draw()

    def _create_circular_flow_diagram(
        self,
        ax,
        edge_connections,
        node_ids,
        node_clusters,
        nodes_per_layer,
        layers,
        small_font,
        medium_font,
        visible_layer_indices,
        cluster_colors,
    ):
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
            G,
            pos,
            nodelist=layer_nodes,
            node_color="lightblue",
            node_size=500,
            alpha=0.8,
            edgecolors="black",
            ax=ax,
        )

        # Draw cluster nodes with their respective colors
        for cluster in unique_clusters:
            node = f"C_{cluster}"
            color = cluster_colors.get(cluster, "gray") if cluster_colors else "gray"
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[node],
                node_color=[color],
                node_size=500,
                alpha=0.8,
                edgecolors="black",
                ax=ax,
            )

        # Draw edges with width based on weight
        edge_weights = [G.edges[edge]["weight"] for edge in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1

        for u, v, data in G.edges(data=True):
            weight = data.get("weight", 1)
            width = 0.5 + 2.5 * (weight / max_weight)
            # Use a color that reflects the weight
            edge_color = plt.cm.Blues(0.2 + 0.8 * (weight / max_weight))
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                width=width,
                alpha=0.7,
                edge_color=[edge_color],
                ax=ax,
            )

        # Draw node labels
        nx.draw_networkx_labels(
            G,
            pos,
            labels={n: n.split("_")[1] for n in G.nodes()},
            font_size=8,
            font_weight="bold",
            ax=ax,
        )

        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

    def _create_force_directed_regions(
        self,
        ax,
        edge_connections,
        node_ids,
        node_clusters,
        nodes_per_layer,
        layers,
        small_font,
        medium_font,
        visible_layer_indices,
        cluster_colors,
    ):
        """
        Create a force-directed graph with layers as regions.
        Nodes are arranged in regions based on their layer, with edges representing connections.
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
        for start_idx, end_idx in edge_connections:
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
        for start_idx, end_idx in edge_connections:
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
            layer_idx = G.nodes[node]["layer"]
            center_x, center_y = layer_centers[layer_idx]

            # Add some random offset
            offset_x = np.random.normal(0, 0.1)
            offset_y = np.random.normal(0, 0.1)

            pos[node] = (center_x + offset_x, center_y + offset_y)

        # Draw layer regions
        for layer_idx, (center_x, center_y) in layer_centers.items():
            if layer_idx < len(layers):
                ellipse = Ellipse(
                    (center_x, center_y),
                    0.5,
                    0.5,
                    alpha=0.2,
                    facecolor="lightgray",
                    edgecolor="gray",
                )
                ax.add_patch(ellipse)

                # Add layer label
                ax.text(
                    center_x,
                    center_y,
                    layers[layer_idx],
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

        # Draw nodes colored by cluster
        for cluster in set(node_to_cluster.values()):
            cluster_nodes = [n for n in G.nodes() if G.nodes[n]["cluster"] == cluster]

            if not cluster_nodes:
                continue

            color = cluster_colors.get(cluster, "gray") if cluster_colors else "gray"

            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=cluster_nodes,
                node_color=color,
                node_size=50,
                alpha=0.8,
                edgecolors="black",
                ax=ax,
            )

        # Draw edges
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, ax=ax)

        # Create a legend for clusters
        legend_elements = []
        for cluster in sorted(set(node_to_cluster.values()))[:5]:  # Limit to top 5
            color = cluster_colors.get(cluster, "gray") if cluster_colors else "gray"
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=8,
                    label=f"Cluster {cluster}",
                )
            )

        # Add the legend
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

    def _create_alluvial_diagram(
        self,
        ax,
        edge_connections,
        node_ids,
        node_clusters,
        nodes_per_layer,
        layers,
        small_font,
        medium_font,
        visible_layer_indices,
        cluster_colors,
    ):
        """
        Create an alluvial diagram showing flow between layers and clusters.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch
        from matplotlib.colors import to_rgba
        import matplotlib.cm as cm
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

        # Calculate y positions for each cluster in each layer
        cluster_y_positions = {}
        for layer_idx in visible_layers:
            y_pos = 0.1  # Start from the bottom with some margin
            cluster_y_positions[layer_idx] = {}

            # Sort clusters by count for better visualization
            layer_clusters = [
                (cluster, cluster_layer_counts[cluster][layer_idx])
                for cluster in unique_clusters
                if layer_idx in cluster_layer_counts[cluster]
            ]
            layer_clusters.sort(key=lambda x: x[1], reverse=True)

            for cluster, count in layer_clusters:
                height = (
                    0.8 * (count / layer_totals[layer_idx])
                    if layer_totals[layer_idx] > 0
                    else 0
                )
                cluster_y_positions[layer_idx][cluster] = (y_pos, height)
                y_pos += height

        # Draw the alluvial diagram with enhanced visuals
        for cluster in unique_clusters:
            # Get a color for this cluster
            base_color = (
                cluster_colors.get(cluster, "gray") if cluster_colors else "gray"
            )

            # Draw flows between consecutive layers
            for i in range(len(visible_layers) - 1):
                layer1 = visible_layers[i]
                layer2 = visible_layers[i + 1]

                if (
                    layer1 in cluster_y_positions
                    and cluster in cluster_y_positions[layer1]
                    and layer2 in cluster_y_positions
                    and cluster in cluster_y_positions[layer2]
                ):
                    x1 = layer_positions[layer1]
                    y1, h1 = cluster_y_positions[layer1][cluster]

                    x2 = layer_positions[layer2]
                    y2, h2 = cluster_y_positions[layer2][cluster]

                    # Create a path for the flow with curved edges
                    control_x = (x1 + x2) / 2

                    # Create a gradient effect
                    rgba_color = to_rgba(base_color)

                    # Create a path with Bezier curves for smoother flow
                    verts = [
                        (x1, y1),  # Start at top-left
                        (control_x, y1),  # Control point
                        (x2, y2),  # Top-right
                        (x2, y2 + h2),  # Bottom-right
                        (control_x, y1 + h1),  # Control point
                        (x1, y1 + h1),  # Bottom-left
                        (x1, y1),  # Back to start
                    ]

                    codes = [
                        Path.MOVETO,
                        Path.CURVE3,
                        Path.LINETO,
                        Path.LINETO,
                        Path.CURVE3,
                        Path.LINETO,
                        Path.CLOSEPOLY,
                    ]

                    path = Path(verts, codes)
                    patch = PathPatch(
                        path,
                        facecolor=base_color,
                        edgecolor="white",
                        alpha=0.8,
                        linewidth=0.5,
                    )
                    ax.add_patch(patch)

            # Draw rectangles for each layer with enhanced styling
            for layer_idx in visible_layers:
                if (
                    layer_idx in cluster_y_positions
                    and cluster in cluster_y_positions[layer_idx]
                ):
                    x = layer_positions[layer_idx]
                    y, height = cluster_y_positions[layer_idx][cluster]

                    # Add a slight 3D effect with a gradient
                    rect = plt.Rectangle(
                        (x - 0.03, y),
                        0.06,
                        height,
                        facecolor=base_color,
                        edgecolor="white",
                        alpha=0.9,
                        linewidth=1.0,
                    )
                    ax.add_patch(rect)

                    # Add count labels if the rectangle is large enough
                    if height > 0.05:
                        count = cluster_layer_counts[cluster][layer_idx]
                        ax.text(
                            x,
                            y + height / 2,
                            str(count),
                            ha="center",
                            va="center",
                            fontsize=9,
                            fontweight="bold",
                            color="white",
                        )

        # Add layer labels with better styling
        for layer_idx, x_pos in layer_positions.items():
            if layer_idx < len(layers):
                # Create a background for the label
                rect = plt.Rectangle(
                    (x_pos - 0.05, 0.02),
                    0.1,
                    0.06,
                    facecolor="lightgray",
                    edgecolor="gray",
                    alpha=0.8,
                    linewidth=1.0,
                    zorder=10,
                )
                ax.add_patch(rect)

                ax.text(
                    x_pos,
                    0.05,
                    layers[layer_idx],
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    zorder=11,
                )

        # Create a more attractive legend for clusters
        legend_elements = []
        for i, cluster in enumerate(unique_clusters[:5]):  # Limit to top 5
            color = cluster_colors.get(cluster, "gray") if cluster_colors else "gray"
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                    label=f"Cluster {cluster}",
                )
            )

        # Add the legend with better styling
        legend = ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=9,
            framealpha=0.9,
            edgecolor="gray",
        )
        legend.set_zorder(20)  # Make sure legend is on top

        # Add a title explaining the visualization
        ax.text(
            0.5,
            0.95,
            "Node Distribution Across Layers by Cluster",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        )

        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])

        # Add a subtle grid for better orientation
        ax.grid(True, linestyle="--", alpha=0.2)

    def _create_radial_network(
        self,
        ax,
        edge_connections,
        node_ids,
        node_clusters,
        nodes_per_layer,
        layers,
        small_font,
        medium_font,
        visible_layer_indices,
        cluster_colors,
    ):
        """
        Create a radial network diagram with layers as concentric circles and clusters as segments.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Wedge, ConnectionPatch, FancyBboxPatch
        from matplotlib.colors import to_rgba, LinearSegmentedColormap
        import matplotlib.cm as cm
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

        # Add a subtle background gradient for visual appeal
        background = np.linspace(0, 1, 100)
        background = np.vstack((background, background))
        ax.imshow(
            background,
            cmap=plt.cm.Blues,
            alpha=0.1,
            aspect="auto",
            extent=[-1, 1, -1, 1],
            origin="lower",
        )

        # Draw connecting lines between layers first (as background)
        for cluster_idx, cluster in enumerate(unique_clusters):
            # Calculate angle range for this cluster
            angle_per_cluster = 360 / len(unique_clusters)
            center_angle = cluster_idx * angle_per_cluster + angle_per_cluster / 2
            center_angle_rad = np.radians(center_angle)

            # Draw lines connecting layers for this cluster
            prev_layer = None
            prev_radius = None
            prev_count = None

            for layer_idx in visible_layers:
                if layer_idx in cluster_layer_counts[cluster]:
                    count = cluster_layer_counts[cluster][layer_idx]
                    radius = layer_radii[layer_idx]

                    if prev_layer is not None:
                        # Draw a connecting line
                        x1 = prev_radius * np.cos(center_angle_rad)
                        y1 = prev_radius * np.sin(center_angle_rad)
                        x2 = radius * np.cos(center_angle_rad)
                        y2 = radius * np.sin(center_angle_rad)

                        # Line width based on count
                        max_count = max(prev_count, count)
                        line_width = 1 + 5 * (max_count / max(layer_totals.values()))

                        # Draw the connecting line with gradient
                        line = ConnectionPatch(
                            (x1, y1),
                            (x2, y2),
                            coordsA="data",
                            coordsB="data",
                            axesA=ax,
                            axesB=ax,
                            color="gray",
                            alpha=0.3,
                            linewidth=line_width,
                        )
                        ax.add_patch(line)

                    prev_layer = layer_idx
                    prev_radius = radius
                    prev_count = count

        # Draw the radial network with enhanced visuals
        for cluster_idx, cluster in enumerate(unique_clusters):
            # Get a color for this cluster
            base_color = (
                cluster_colors.get(cluster, "gray") if cluster_colors else "gray"
            )

            # Calculate angle range for this cluster
            angle_per_cluster = 360 / len(unique_clusters)
            start_angle = cluster_idx * angle_per_cluster
            end_angle = (cluster_idx + 1) * angle_per_cluster

            # Draw wedges for each layer with enhanced styling
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

                    # Draw the wedge with enhanced styling
                    radius = layer_radii[layer_idx]
                    width = radius_step * 0.8

                    # Create a slightly darker edge color for 3D effect
                    edge_color = "white"

                    wedge = Wedge(
                        (0, 0),
                        radius,
                        wedge_start,
                        wedge_end,
                        width=width,
                        facecolor=base_color,
                        edgecolor=edge_color,
                        alpha=0.9,
                        linewidth=1.0,
                    )
                    ax.add_patch(wedge)

                    # Add a label if the wedge is large enough
                    if angle_span > 15:
                        # Calculate the position for the label
                        angle_rad = np.radians(center_angle)
                        label_radius = radius - width / 2
                        x = label_radius * np.cos(angle_rad)
                        y = label_radius * np.sin(angle_rad)

                        # Add a text label with count
                        ax.text(
                            x,
                            y,
                            str(count),
                            ha="center",
                            va="center",
                            fontsize=9,
                            fontweight="bold",
                            color="white",
                            bbox=dict(
                                facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"
                            ),
                        )

        # Add layer labels with better styling
        for layer_idx, radius in layer_radii.items():
            if layer_idx < len(layers):
                # Create a fancy box for the layer label
                box = FancyBboxPatch(
                    (-0.08, radius - 0.02),
                    0.16,
                    0.04,
                    boxstyle=f"round,pad=0.3",
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.9,
                    zorder=10,
                )
                ax.add_patch(box)

                # Add the layer text
                ax.text(
                    0,
                    radius,
                    layers[layer_idx],
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    zorder=11,
                )

        # Create a more attractive legend for clusters
        legend_elements = []
        for cluster in unique_clusters[:5]:  # Limit to top 5
            color = cluster_colors.get(cluster, "gray") if cluster_colors else "gray"
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                    label=f"Cluster {cluster}",
                )
            )

        # Add the legend with better styling
        legend = ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=9,
            framealpha=0.9,
            edgecolor="gray",
        )
        legend.set_zorder(20)  # Make sure legend is on top

        # Add a title explaining the visualization
        ax.text(
            0,
            0.9,
            "Radial Distribution of Clusters Across Layers",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.3"
            ),
        )

        # Set axis limits
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        # Add a subtle circular grid
        for r in np.linspace(0.2, 0.8, 4):
            circle = plt.Circle(
                (0, 0), r, fill=False, color="gray", linestyle="--", alpha=0.2
            )
            ax.add_patch(circle)

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
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes using current_node_mask</li>
            <li>Create matrix with shape (num_clusters, num_layers)</li>
            <li>For each visible node, increment matrix[cluster_idx, layer_idx]</li>
            <li>Visualize matrix using matplotlib.pyplot.imshow with 'viridis' colormap</li>
            <li>Add colorbar and annotations showing node counts</li>
        </ol>
        
        <p><b>Interpretation:</b> Each cell (i,j) shows count of nodes in cluster i and layer j. Darker colors = higher count.</p>
        """
        self.tab_widget.setTabToolTip(0, heatmap_tooltip)

        # LC2: Distribution tooltip
        distribution_tooltip = """
        <h3>LC2: Cluster Distribution by Layer</h3>
        
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
            <li>For each layer, count nodes in each cluster</li>
            <li>Create stacked bar chart showing distribution of nodes across clusters for each layer</li>
            <li>Add legend showing cluster colors</li>
        </ol>
        
        <p><b>Interpretation:</b> Each bar shows distribution of nodes across clusters for a specific layer. Colors correspond to clusters.</p>
        """
        self.tab_widget.setTabToolTip(1, distribution_tooltip)

        # LC3: Layer Distribution tooltip
        layer_distribution_tooltip = """
        <h3>LC3: Layer Distribution by Cluster</h3>
        
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
            <li>For each cluster, count nodes in each layer</li>
            <li>Create stacked bar chart showing distribution of nodes across layers for each cluster</li>
            <li>Add legend showing layer colors</li>
        </ol>
        
        <p><b>Interpretation:</b> Each bar shows distribution of nodes across layers for a specific cluster. Colors correspond to layers.</p>
        """
        self.tab_widget.setTabToolTip(2, layer_distribution_tooltip)

        # LC4: Network Diagram tooltip
        network_tooltip = """
        <h3>LC4: Layer-Cluster Network Diagram</h3>
        
        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.nodes_per_layer: Dictionary mapping layer indices to lists of node IDs</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
            <li>data_manager.edge_connections: List of (start_idx, end_idx) tuples representing links</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes and edges using current_node_mask</li>
            <li>Create networkx graph with nodes representing layers and clusters</li>
            <li>Add edges between nodes based on connections in the network</li>
            <li>Set node sizes based on number of nodes in each layer/cluster</li>
            <li>Set edge widths based on number of connections between layers/clusters</li>
            <li>Use networkx spring layout to position nodes</li>
            <li>Draw network using networkx drawing functions</li>
        </ol>
        
        <p><b>Interpretation:</b> Nodes represent layers (circles) and clusters (squares). Node size indicates number of nodes. Edge width indicates number of connections. Position shows relationship strength.</p>
        """
        self.tab_widget.setTabToolTip(3, network_tooltip)

        # LC5: Cluster Network tooltip
        cluster_network_tooltip = """
        <h3>LC5: Cluster Networks</h3>
        
        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
            <li>data_manager.edge_connections: List of (start_idx, end_idx) tuples representing links</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes and edges using current_node_mask</li>
            <li>Create networkx graph with nodes representing clusters</li>
            <li>Add edges between nodes based on connections in the network</li>
            <li>Set node sizes based on number of nodes in each cluster</li>
            <li>Set edge widths based on number of connections between clusters</li>
            <li>Use networkx spring layout to position nodes</li>
            <li>Draw network using networkx drawing functions</li>
        </ol>
        
        <p><b>Interpretation:</b> Nodes represent clusters. Node size indicates number of nodes. Edge width indicates number of connections. Position shows relationship strength.</p>
        """
        self.tab_widget.setTabToolTip(4, cluster_network_tooltip)

        # LC6: Sankey Diagram tooltip
        sankey_tooltip = """
        <h3>LC6: Layer-Cluster Sankey Diagram</h3>
        
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
            <li>Create list of layers and clusters as Sankey diagram nodes</li>
            <li>Create list of links between layers and clusters based on node distribution</li>
            <li>Set link colors based on source node colors</li>
            <li>Create Sankey diagram using matplotlib-sankey</li>
        </ol>
        
        <p><b>Interpretation:</b> Shows flow of nodes from layers (left) to clusters (right). Width of flow indicates number of nodes. Colors correspond to layers.</p>
        """
        self.tab_widget.setTabToolTip(5, sankey_tooltip)

        # LC7: Connectivity Matrix tooltip
        connectivity_tooltip = """
        <h3>LC7: Cluster Connectivity Matrix</h3>
        
        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
            <li>data_manager.edge_connections: List of (start_idx, end_idx) tuples representing links</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes and edges using current_node_mask</li>
            <li>Create matrix with shape (num_clusters, num_clusters)</li>
            <li>For each edge, increment matrix[source_cluster, target_cluster]</li>
                    <li>Visualize matrix using matplotlib.pyplot.imshow with 'viridis' colormap</li>
            <li>Add colorbar and annotations showing connection counts</li>
        </ol>
        
        <p><b>Interpretation:</b> Each cell (i,j) shows count of connections between cluster i and cluster j. Darker colors = higher count.</p>
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
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes using current_node_mask</li>
            <li>Create matrix with shape (num_layers + num_clusters, num_layers + num_clusters)</li>
            <li>Fill matrix with counts of nodes shared between layers and clusters</li>
            <li>Create chord diagram using matplotlib</li>
            <li>Add labels for layers and clusters</li>
        </ol>
        
        <p><b>Interpretation:</b> Arcs represent layers and clusters. Connections between arcs show shared nodes. Width indicates number of nodes.</p>
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
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
            <li>data_manager.edge_connections: List of (start_idx, end_idx) tuples representing links</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes and edges using current_node_mask</li>
            <li>Create three matrices:
                <ul>
                    <li>count_matrix: Count of nodes in each cluster-layer combination</li>
                    <li>connection_matrix: Count of connections in each cluster-layer combination</li>
                    <li>max_possible_matrix: Maximum possible connections in each cluster-layer combination</li>
                </ul>
            </li>
            <li>For each cluster-layer combination:
                <ul>
                    <li>Count intra-layer connections (both nodes in same layer and cluster)</li>
                    <li>Count inter-layer connections (nodes in different layers but same cluster)</li>
                    <li>Calculate maximum possible connections = n × (n-1) / 2 where n = node count</li>
                    <li>Compute density = actual / maximum (or normalized node count if maximum = 0)</li>
        </ul>
            </li>
            <li>Visualize as a 2x2 grid of heatmaps</li>
        </ol>
        
        <p><b>Charts Calculation & Interpretation:</b></p>
        <ul>
            <li><b>Original Density Heatmap (top-left):</b>
                <ul>
                    <li><i>Calculation:</i> Raw count of nodes in each cluster-layer combination</li>
                    <li><i>Values:</i> Integer counts (e.g., 5, 12, 7)</li>
                    <li><i>Interpretation:</i> Darker cells indicate more nodes in that cluster-layer combination</li>
                </ul>
            </li>
            <li><b>Node Count Matrix (top-right):</b>
                <ul>
                    <li><i>Calculation:</i> Same as Original Heatmap but with different color scheme</li>
                    <li><i>Values:</i> Integer counts (e.g., 5, 12, 7)</li>
                    <li><i>Interpretation:</i> Shows distribution of nodes across clusters and layers</li>
                </ul>
            </li>
            <li><b>Connection Matrix (bottom-left):</b>
                <ul>
                    <li><i>Calculation:</i> For each cluster-layer: intra-layer connections count as 1.0, inter-layer connections count as 0.5 for each layer</li>
                    <li><i>Values:</i> Decimal numbers (e.g., 3.5, 8.0, 2.5)</li>
                    <li><i>Interpretation:</i> Higher values indicate more connections within/between layers for that cluster</li>
                </ul>
            </li>
            <li><b>Density Matrix (bottom-right):</b>
                <ul>
                    <li><i>Calculation:</i> connection_matrix[i,j] / max_possible_matrix[i,j] where max_possible = n(n-1)/2</li>
                    <li><i>Values:</i> Range from 0.0 (no connections) to 1.0 (fully connected)</li>
                    <li><i>Interpretation:</i> Measures connectivity efficiency; 1.0 means all possible connections exist</li>
                </ul>
            </li>
        </ul>
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
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes using current_node_mask</li>
            <li>For each layer, identify clusters that have nodes in that layer</li>
            <li>Create co-occurrence matrix counting how many layers each pair of clusters appears in together</li>
            <li>Create networkx graph with nodes representing clusters</li>
            <li>Add edges between clusters that co-occur in at least one layer</li>
            <li>Set edge weights based on co-occurrence count</li>
            <li>Use networkx spring layout to position nodes</li>
            <li>Draw network using networkx drawing functions</li>
        </ol>
        
        <p><b>Interpretation:</b> Nodes represent clusters. Edge width indicates how many layers the clusters co-occur in. Position shows relationship strength.</p>
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
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes using current_node_mask</li>
            <li>Create matrix with shape (num_clusters, num_layers)</li>
            <li>For each visible node, increment matrix[cluster_idx, layer_idx]</li>
            <li>Normalize matrix columns to sum to 1.0 (i.e., divide by column sum)</li>
            <li>Visualize matrix using matplotlib.pyplot.imshow with 'viridis' colormap</li>
            <li>Add colorbar and annotations showing normalized values</li>
        </ol>
        
        <p><b>Interpretation:</b> Each cell (i,j) shows percentage of layer j's nodes in cluster i. Values sum to 1.0 per column. Darker colors = higher percentage.</p>
        """
        self.tab_widget.setTabToolTip(10, normalized_tooltip)

        # LC12: Similarity Matrix tooltip
        similarity_tooltip = """
        <h3>LC12: Enhanced Cluster Similarity Analysis</h3>
        
        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.nodes_per_layer: Dictionary mapping layer indices to lists of node IDs</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
            <li>data_manager.edge_connections: List of (start_idx, end_idx) tuples representing links</li>
        </ul>
        
        <p><b>Controls:</b></p>
        <ul>
            <li><b>LC12 Metric:</b> Select a specific similarity metric or view all metrics in a grid</li>
            <li><b>Edge Type:</b> Filter edges used in calculations (All Edges, Interlayer only, or Intralayer only)</li>
        </ul>
        
        <p><b>Visualization:</b></p>
        <p>This enhanced visualization presents 9 different similarity metrics in a 3x3 grid:</p>
        
        <ol>
            <li><b>Jaccard Similarity:</b>
                <ul>
                    <li><i>Calculation:</i> |A ∩ B| / |A ∪ B| where A and B are sets of nodes in clusters i and j</li>
                    <li><i>Values:</i> Range from 0.0 (no overlap) to 1.0 (identical sets)</li>
                    <li><i>Interpretation:</i> Measures proportion of shared nodes relative to total nodes</li>
        </ul>
            </li>
            <li><b>Cosine Similarity:</b>
                <ul>
                    <li><i>Calculation:</i> (A·B)/(||A||·||B||) where A and B are layer distribution vectors for clusters i and j</li>
                    <li><i>Values:</i> Range from 0.0 (orthogonal) to 1.0 (parallel vectors)</li>
                    <li><i>Interpretation:</i> Measures similarity of layer distribution patterns regardless of magnitude</li>
                </ul>
            </li>
            <li><b>Overlap Coefficient:</b>
                <ul>
                    <li><i>Calculation:</i> |A ∩ B| / min(|A|, |B|) where A and B are sets of nodes in clusters i and j</li>
                    <li><i>Values:</i> Range from 0.0 (no overlap) to 1.0 (one set is subset of other)</li>
                    <li><i>Interpretation:</i> Measures whether smaller cluster is subset of larger one</li>
                </ul>
            </li>
            <li><b>Connection Similarity:</b>
                <ul>
                    <li><i>Calculation:</i> Number of direct connections between clusters i and j divided by sqrt(|Ci|·|Cj|) where |Ci| is node count in cluster i</li>
                    <li><i>Values:</i> Range from 0.0 (no connections) to values typically < 1.0</li>
                    <li><i>Interpretation:</i> Measures strength of direct connectivity between clusters</li>
                </ul>
            </li>
            <li><b>Layer Distribution:</b>
                <ul>
                    <li><i>Calculation:</i> 1 - Jensen-Shannon divergence between normalized layer distribution vectors</li>
                    <li><i>Values:</i> Range from 0.0 (completely different) to 1.0 (identical distributions)</li>
                    <li><i>Interpretation:</i> Measures similarity of proportional layer distributions</li>
                </ul>
            </li>
            <li><b>Hierarchical Clustering:</b>
                <ul>
                    <li><i>Calculation:</i> Hierarchical clustering using average linkage on Jaccard distance matrix</li>
                    <li><i>Values:</i> Dendrogram with branch lengths representing distances</li>
                    <li><i>Interpretation:</i> Clusters grouped by similarity; shorter branches = more similar</li>
                </ul>
            </li>
            <li><b>Node Sharing Ratio:</b>
                <ul>
                    <li><i>Calculation:</i> For each layer, calculate |A ∩ B| / |A ∪ B| where A and B are sets of nodes in that layer for clusters i and j, then average across layers</li>
                    <li><i>Values:</i> Range from 0.0 (no sharing) to 1.0 (identical node sets in each layer)</li>
                    <li><i>Interpretation:</i> Measures layer-specific node sharing patterns</li>
                </ul>
            </li>
            <li><b>Path-based Similarity:</b>
                <ul>
                    <li><i>Calculation:</i> 1 / (1 + average shortest path length between nodes in clusters i and j)</li>
                    <li><i>Values:</i> Range from near 0.0 (distant) to 1.0 (directly connected)</li>
                    <li><i>Interpretation:</i> Measures topological proximity between clusters</li>
                </ul>
            </li>
            <li><b>Mutual Information:</b>
                <ul>
                    <li><i>Calculation:</i> Normalized mutual information between layer distributions: I(X;Y)/sqrt(H(X)·H(Y)) where H is entropy</li>
                    <li><i>Values:</i> Range from 0.0 (independent) to 1.0 (perfectly correlated)</li>
                    <li><i>Interpretation:</i> Measures information shared between layer distributions</li>
                </ul>
            </li>
        </ol>
        
        <p><b>Interpretation:</b></p>
        <ul>
            <li>For matrix visualizations, each cell (i,j) shows similarity between clusters i and j</li>
            <li>Values range from 0 (no similarity) to 1 (identical)</li>
            <li>Diagonal values are 1.0 (self-similarity)</li>
            <li>Darker blue colors indicate higher similarity</li>
            <li>The dendrogram groups clusters by similarity (shorter branches = more similar)</li>
        </ul>
        
        <p><b>Applications:</b></p>
        <ul>
            <li>Identify clusters that share similar layer distributions</li>
            <li>Discover functional relationships between clusters</li>
            <li>Compare different similarity metrics to gain deeper insights</li>
            <li>Understand hierarchical relationships between clusters</li>
        </ul>
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
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes using current_node_mask</li>
            <li>Create matrix with shape (num_clusters, num_layers)</li>
            <li>For each visible node, increment matrix[cluster_idx, layer_idx]</li>
            <li>Create scatter plot with x-axis representing layers and y-axis representing clusters</li>
            <li>Set bubble size based on node count</li>
            <li>Set bubble color based on cluster color</li>
        </ol>
        
        <p><b>Interpretation:</b> Each bubble represents a cluster-layer combination. Bubble size indicates number of nodes. Bubble color corresponds to cluster.</p>
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
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
            <li>data_manager.visible_links or data_manager.edge_connections: List of visible links in the network</li>
        </ul>
        
        <p><b>Controls:</b></p>
        <ul>
            <li><b>Count Type:</b> Select whether to count nodes or intralayer edges within each cluster-layer combination</li>
        </ul>
        
        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible nodes using current_node_mask</li>
            <li>For each cluster, calculate the total number of nodes or intralayer edges in each layer</li>
            <li>Create a treemap where each rectangle represents a cluster-layer combination</li>
            <li>Size of each rectangle is proportional to the count (nodes or edges) in that combination</li>
            <li>Color of each rectangle is based on the cluster color</li>
        </ol>
        
        <p><b>Interpretation:</b> The treemap provides a hierarchical visualization of the distribution of nodes or intralayer edges across layers and clusters. Larger rectangles indicate higher density in that combination.</p>
        """
        self.tab_widget.setTabToolTip(13, treemap_tooltip)

        # LC15: Flow Ideas tooltip
        flow_ideas_tooltip = """
        <h3>LC15: Flow Ideas</h3>
        
        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.nodes_per_layer: Dictionary mapping layer indices to lists of node IDs</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
        </ul>
        
        <p><b>Visualization Ideas:</b></p>
        <ul>
            <li>1. Circular Flow Diagram: Show connections between layers and clusters in a circular layout</li>
            <li>2. Force-Directed Regions: Create a force-directed graph with layers as regions</li>
            <li>3. Alluvial Diagram: Show flow between layers and clusters</li>
            <li>4. Radial Network: Create a radial network diagram with layers as concentric circles and clusters as segments</li>
                </ul>
        
        <p><b>Interpretation:</b> These visualizations provide different perspectives on the flow of information and connections between layers and clusters.</p>
        """
        self.tab_widget.setTabToolTip(14, flow_ideas_tooltip)

        # LC16: Interlayer Path Analysis tooltip
        path_analysis_tooltip = """
        <h3>LC16: Interlayer Path Analysis with Duplicated Nodes</h3>

        <p><b>Data Source:</b></p>
        <ul>
            <li>data_manager.node_ids: List of all node IDs in the network</li>
            <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
            <li>data_manager.nodes_per_layer: Dictionary mapping layer indices to lists of node IDs</li>
            <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
            <li>data_manager.edge_connections: List of (start_idx, end_idx) tuples representing links</li>
        </ul>

        <p><b>Controls:</b></p>
        <ul>
            <li><b>LC16 Analysis:</b> Select an analysis type (Path Length, Betweenness, Bottleneck)</li>
            <li><b>LC20/16 Cluster:</b> Filter the analysis to show only nodes and connections from a specific cluster</li>
            <li><b>LC16 Visualization:</b> Choose a visualization style:
                <ul>
                    <li><b>Standard:</b> Balanced layout showing network structure</li>
                    <li><b>Simplified:</b> Cleaner layout with fewer elements</li>
                    <li><b>Detailed:</b> Shows more nodes and connections, including less significant ones</li>
                    <li><b>Classic Circle:</b> Arranges nodes in a circle for a different perspective</li>
                </ul>
            </li>
            <li><b>Labels:</b> Toggle node and edge labels</li>
            <li><b>Nodes:</b> Toggle node visibility</li>
            <li><b>Color by Centrality:</b> When checked, edges are colored by betweenness centrality (red = high, blue = low) instead of edge type</li>
        </ul>

        <p><b>Calculation Method:</b></p>
        <ol>
            <li>Filter visible links using current_node_mask</li>
            <li>If a cluster is selected in the LC20 dropdown, filter nodes and edges to include only those from the selected cluster</li>
            <li>Duplicate each node for each layer it appears in, using the naming convention <layer>_<node></li>
            <li>Create intralayer edges only between duplicated nodes that have existing connections in the original network</li>
            <li>Create interlayer edges between duplicated nodes in different layers when they represent the same original nodes</li>
            <li>Build a comprehensive network where duplicated nodes connect interlayer and intralayer edges</li>
            <li>For each analysis type, calculate the corresponding path metrics across this duplicated node network</li>
        </ol>

        <p><b>Visualization:</b></p>
        <ul>
            <li><b>Path Length:</b> Heatmap showing average shortest path lengths between layers in the duplicated node network</li>
            <li><b>Betweenness:</b> Bar chart showing betweenness centrality of nodes by layer in the duplicated node network</li>
            <li><b>Bottleneck:</b> Network diagram highlighting critical connections in the duplicated node network, with several visualization options:
                <ul>
                    <li>By default, interlayer edges are shown in blue and intralayer edges in red</li>
                    <li>When "Color by Centrality" is enabled, edges are colored by their betweenness centrality</li>
                    <li>Edge thickness indicates betweenness centrality (thicker = more important)</li>
                    <li>Node size indicates betweenness centrality (larger = more important)</li>
                </ul>
            </li>
        </ul>

        <p><b>Interpretation:</b></p>
        <ul>
            <li>Each cell (i,j) shows similarity between clusters i and j based on interlayer paths</li>
            <li>Values range from near 0 (distant/disconnected) to 1 (directly connected)</li>
            <li>Darker colors indicate higher similarity (shorter paths)</li>
            <li>In single cluster view, each matrix represents paths between specific layer pairs</li>
        </ul>
        
        <p><b>Applications:</b></p>
        <ul>
            <li>Identify clusters that serve as bridges between layers</li>
            <li>Discover efficient interlayer communication pathways</li>
            <li>Analyze how information might flow between layers through specific clusters</li>
            <li>Compare interlayer connectivity patterns across different clusters</li>
        </ul>
        """
        self.tab_widget.setTabToolTip(15, path_analysis_tooltip)

        # LC17: Cluster Bridging Analysis tooltip
        bridge_analysis_tooltip = """
         <h3>LC17: Cluster Bridging Analysis with Duplicated Nodes</h3>
         
         <p><b>Data Source:</b></p>
         <ul>
             <li>data_manager.node_ids: List of all node IDs in the network</li>
             <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
             <li>data_manager.nodes_per_layer: Dictionary mapping layer indices to lists of node IDs</li>
             <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
             <li>data_manager.edge_connections: List of (start_idx, end_idx) tuples representing links</li>
         </ul>
         
         <p><b>Controls:</b></p>
         <ul>
             <li><b>LC17 Analysis:</b> Select an analysis type (Bridge Score, Flow Efficiency, Layer Span)</li>
         </ul>
         
         <p><b>Calculation Method:</b></p>
         <ol>
             <li>Filter visible links using current_node_mask</li>
             <li>Duplicate each node for each layer it appears in, using the naming convention <layer>_<node></li>
             <li>Create intralayer edges only between duplicated nodes that have existing connections in the original network</li>
             <li>Create interlayer edges between duplicated nodes in different layers when they represent the same original nodes</li>
             <li>Build a comprehensive network where duplicated nodes connect interlayer and intralayer edges</li>
             <li>For each analysis type, calculate the corresponding bridging metric across this duplicated node network</li>
         </ol>
         
         <p><b>Visualization:</b></p>
         <ul>
             <li><b>Bridge Score:</b> Bar chart showing how well each cluster connects different layers through both interlayer and intralayer edges</li>
             <li><b>Flow Efficiency:</b> Heatmap showing how efficiently information can flow between layers through each cluster in the duplicated node network</li>
             <li><b>Layer Span:</b> Stacked bar chart showing the distribution of each cluster's nodes across different layers</li>
         </ul>
         
         <p><b>Interpretation:</b></p>
         <ul>
             <li><b>Bridge Score:</b> Higher values indicate clusters that more effectively bridge between layers through both interlayer and intralayer connections. The score is calculated as the ratio of interlayer to total connections within a cluster.</li>
             <li><b>Flow Efficiency:</b> Each cell (i,j) shows the best cluster for efficient information flow between layers i and j. Higher values (brighter colors) indicate more efficient pathways through the duplicated node network.</li>
             <li><b>Layer Span:</b> Shows how each cluster's nodes are distributed across layers. Clusters with higher span values bridge more layers. The stacked bars show the proportion of each cluster's nodes in each layer.</li>
         </ul>
         
         <p><b>Applications:</b></p>
         <ul>
             <li>Identify clusters that serve as effective bridges between layers</li>
             <li>Discover efficient interlayer communication pathways through the duplicated node network</li>
             <li>Analyze how information might flow between layers through specific clusters</li>
             <li>Compare interlayer connectivity patterns across different layers</li>
             <li>Find clusters that span multiple layers and may play important roles in cross-layer interactions</li>
         </ul>
         """
        self.tab_widget.setTabToolTip(16, bridge_analysis_tooltip)

        # LC20: Interlayer Path Similarity tooltip
        interlayer_path_similarity_tooltip = """
         <h3>LC20: Interlayer Path Similarity Analysis</h3>
         
         <p><b>Data Source:</b></p>
         <ul>
             <li>data_manager.node_ids: List of all node IDs in the network</li>
             <li>data_manager.node_clusters: Dictionary mapping node IDs to cluster assignments</li>
             <li>data_manager.nodes_per_layer: Dictionary mapping layer indices to lists of node IDs</li>
             <li>data_manager.current_node_mask: Boolean mask for visible nodes</li>
             <li>data_manager.edge_connections: List of (start_idx, end_idx) tuples representing links</li>
         </ul>
         
         <p><b>Controls:</b></p>
         <ul>
             <li><b>LC20 Cluster:</b> Select a specific cluster to analyze or view all clusters</li>
         </ul>
         
         <p><b>Calculation Method:</b></p>
         <ol>
             <li>Filter visible nodes and edges using current_node_mask</li>
             <li>Identify interlayer edges (connections between nodes in different layers)</li>
             <li>For each cluster pair (or selected cluster with all others):
                 <ul>
                     <li>Calculate path-based similarity using interlayer edges only</li>
                     <li>Compute average shortest path length between nodes in different clusters</li>
                     <li>Transform to similarity score: 1 / (1 + average path length)</li>
                 </ul>
             </li>
             <li>Create heatmap showing similarity scores between clusters</li>
             <li>For selected cluster view, create individual heatmaps for each layer pair</li>
         </ol>
         
         <p><b>Visualization:</b></p>
         <ul>
             <li><b>Combined View (All Clusters):</b> Matrix showing path-based similarity between all cluster pairs using only interlayer edges</li>
             <li><b>Single Cluster View:</b> Multiple matrices showing path-based similarity between the selected cluster and all others, broken down by layer pairs</li>
         </ul>
         """
        self.tab_widget.setTabToolTip(18, interlayer_path_similarity_tooltip)

    def update_enhanced_network(self):
        """Update the enhanced network diagram with the current data"""
        if (
            not hasattr(self, "enhanced_network_figure")
            or not self.enable_checkbox.isChecked()
        ):
            return

        # Check if the enhanced network is enabled
        if not self.enhanced_network_enable_checkbox.isChecked():
            # Clear the figure
            self.enhanced_network_figure.clear()
            self.enhanced_network_ax = self.enhanced_network_figure.add_subplot(111)
            self.enhanced_network_ax.text(
                0.5,
                0.5,
                "Enhanced network diagram disabled",
                ha="center",
                va="center",
                fontsize=12,
            )
            self.enhanced_network_canvas.draw()
            return

        # Clear the figure
        self.enhanced_network_figure.clear()

        # Create the axis if it doesn't exist
        if not hasattr(self, "enhanced_network_ax"):
            self.enhanced_network_ax = self.enhanced_network_figure.add_subplot(111)
        else:
            self.enhanced_network_ax = self.enhanced_network_figure.add_subplot(111)

        # Get the current data manager
        data_manager = self._current_data
        if data_manager is None:
            return

        # Get the selected layout algorithm and aspect ratio
        layout_algorithm = self.layout_algorithm_combo.currentText().lower()
        aspect_ratio = float(self.aspect_ratio_combo.currentText())

        #
        self.enhanced_network_ax = update_enhanced_network(
            self.enhanced_network_figure,
            self.enhanced_network_ax,
            self.enhanced_network_canvas,
            data_manager,
            self.enhanced_network_ui,
            layout_algorithm,
            aspect_ratio,
        )

    def on_enhanced_network_state_changed(self, state):
        """Handle enable/disable state change for enhanced network diagram"""
        if hasattr(self, "_current_data") and self._current_data:
            self.update_enhanced_network()

            # Enable or disable the controls based on the checkbox state
            self.enhanced_network_ui["group"].setEnabled(state)

    def update_lc12_similarity_matrix(self, data_manager):
        """Update the LC12 similarity matrix with the selected metric and edge type"""
        if (
            not hasattr(self, "similarity_figure")
            or not self.enable_checkbox.isChecked()
        ):
            return

        # Clear the figure
        self.similarity_figure.clear()
        self.similarity_ax = self.similarity_figure.add_subplot(111)

        # Get visible links
        visible_links = []
        if data_manager.current_edge_mask is not None:
            visible_links = [
                data_manager.link_pairs[i]
                for i in range(len(data_manager.link_pairs))
                if data_manager.current_edge_mask[i]
            ]

        # Get the selected metric and edge type
        metric_index = self.similarity_metric_combo.currentIndex()
        edge_type_index = self.edge_type_combo.currentIndex()

        # Map indices to metric and edge type
        metric = (
            "all"
            if metric_index == 0
            else self.similarity_metric_combo.currentText().lower().replace(" ", "_")
        )
        edge_type = (
            "all"
            if edge_type_index == 0
            else self.edge_type_combo.currentText().lower()
        )

        # Define font sizes
        small_font = {"fontsize": 9}
        medium_font = {"fontsize": 12}

        # Create the similarity matrix with the selected metric and edge type
        create_enhanced_cluster_similarity(
            self.similarity_ax,
            visible_links,
            data_manager.node_ids,
            data_manager.node_clusters,
            data_manager.nodes_per_layer,
            data_manager.layers,
            small_font,
            medium_font,
            data_manager.visible_layers,
            data_manager.cluster_colors,
            metric=metric,
            edge_type=edge_type,
        )

        # Apply tight layout and draw
        self.similarity_figure.tight_layout()
        self.similarity_canvas.draw()

    def update_lc20_interlayer_path_similarity(self, data_manager):
        """Update the LC20 interlayer path similarity visualization"""
        if (
            not hasattr(self, "interlayer_path_similarity_figure")
            or not self.enable_checkbox.isChecked()
        ):
            return

        # Clear the figure
        self.interlayer_path_similarity_figure.clear()

        # Get visible links
        visible_links = []
        if data_manager.current_edge_mask is not None:
            visible_links = [
                data_manager.link_pairs[i]
                for i in range(len(data_manager.link_pairs))
                if data_manager.current_edge_mask[i]
            ]

        # Get the selected cluster
        cluster_selection = self.path_similarity_cluster_combo.currentText()
        try:
            # Handle different possible formats of cluster selection text
            if cluster_selection == "All Clusters":
                selected_cluster = None
            else:
                # Try to extract the cluster number, handling different formats
                parts = cluster_selection.split()
                if len(parts) > 1 and parts[0].lower() == "cluster":
                    selected_cluster = int(parts[1])
                elif len(parts) > 1:
                    # Try to find a number in the parts
                    for part in parts:
                        if part.isdigit():
                            selected_cluster = int(part)
                            break
                    else:
                        selected_cluster = None
                else:
                    # If it's just a number
                    selected_cluster = (
                        int(cluster_selection) if cluster_selection.isdigit() else None
                    )
        except (ValueError, IndexError):
            # If parsing fails, default to All Clusters
            logging.warning(
                f"Could not parse cluster selection: {cluster_selection}, defaulting to All Clusters"
            )
            selected_cluster = None

        # Define font sizes
        small_font = {"fontsize": 9}
        medium_font = {"fontsize": 12}

        # Create the interlayer path similarity visualization
        create_interlayer_path_similarity(
            self.interlayer_path_similarity_figure,
            visible_links,
            data_manager.node_ids,
            data_manager.node_clusters,
            data_manager.nodes_per_layer,
            data_manager.layers,
            small_font,
            medium_font,
            data_manager.visible_layers,
            data_manager.cluster_colors,
            selected_cluster=selected_cluster,
        )

        # Apply tight layout and draw
        self.interlayer_path_similarity_figure.tight_layout()
        self.interlayer_path_similarity_canvas.draw()

    def update_cluster_selector(self, data_manager):
        """Update the cluster selector with available clusters"""
        if not hasattr(self, "path_similarity_cluster_combo"):
            return

        # Store current selection
        current_text = self.path_similarity_cluster_combo.currentText()

        # Clear the combo box
        self.path_similarity_cluster_combo.clear()

        # Add "All Clusters" option
        self.path_similarity_cluster_combo.addItem("All Clusters")

        # Get unique clusters
        unique_clusters = sorted(set(data_manager.node_clusters.values()))

        # Add each cluster
        for cluster in unique_clusters:
            self.path_similarity_cluster_combo.addItem(f"Cluster {cluster}")

        # Restore selection if possible
        index = self.path_similarity_cluster_combo.findText(current_text)
        if index >= 0:
            self.path_similarity_cluster_combo.setCurrentIndex(index)
        else:
            self.path_similarity_cluster_combo.setCurrentIndex(0)

    def on_lc14_count_type_changed(self, _):
        """Handle change in LC14 count type dropdown"""
        if hasattr(self, "_current_data") and self.enable_checkbox.isChecked():
            self.update_lc14_treemap(self._current_data)

    def update_lc14_treemap(self, data_manager):
        """Update the LC14 treemap visualization"""
        # Get the current count type
        count_type = (
            "nodes"
            if self.lc14_count_type_combo.currentText() == "Nodes"
            else "intralayer_edges"
        )

        # Clear the figure
        self.treemap_ax.clear()

        # Get data from data manager
        node_ids = data_manager.node_ids
        node_clusters = data_manager.node_clusters
        nodes_per_layer = data_manager.nodes_per_layer
        layers = data_manager.layers

        # Get visible links (edge connections)
        visible_links = []
        if hasattr(data_manager, "visible_links"):
            visible_links = data_manager.visible_links
        elif hasattr(data_manager, "edge_connections"):
            visible_links = data_manager.edge_connections

        # Get visible layer indices
        visible_layer_indices = []
        if hasattr(data_manager, "get_visible_layer_indices"):
            visible_layer_indices = data_manager.get_visible_layer_indices()
        elif hasattr(data_manager, "visible_layer_indices"):
            visible_layer_indices = data_manager.visible_layer_indices

        # Get font sizes
        small_font = {"fontsize": 8}
        medium_font = {"fontsize": 10}

        # Create the treemap
        from charts.layer_cluster.lc14_treemap import create_layer_cluster_treemap

        create_layer_cluster_treemap(
            self.treemap_ax,
            visible_links,
            node_ids,
            node_clusters,
            nodes_per_layer,
            layers,
            small_font,
            medium_font,
            visible_layer_indices,
            data_manager.cluster_colors,
            count_type=count_type,
        )

        # Draw the canvas
        self.treemap_figure.tight_layout()
        self.treemap_canvas.draw()

    def on_flow_ideas_state_changed(self, state):
        """Handle enable/disable state change for flow ideas visualization"""
        if hasattr(self, "_current_data") and self._current_data:
            if state:
                # If enabled, update the flow ideas visualization
                self.update_flow_ideas(self._current_data)
            else:
                # If disabled, show the disabled message
                for ax in [
                    self.flow_ideas_ax1,
                    self.flow_ideas_ax2,
                    self.flow_ideas_ax3,
                    self.flow_ideas_ax4,
                ]:
                    ax.clear()
                    ax.text(
                        0.5,
                        0.5,
                        "Flow ideas visualization disabled",
                        ha="center",
                        va="center",
                        fontsize=12,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                self.flow_ideas_canvas.draw()

    def update_flow_ideas(self, data_manager):
        """Update the flow ideas visualization with the current data"""
        if (
            not hasattr(self, "flow_ideas_figure")
            or not self.enable_checkbox.isChecked()
            or not self.flow_ideas_enable_checkbox.isChecked()
        ):
            return

        # Get visible links
        visible_links = []
        if data_manager.current_edge_mask is not None:
            visible_links = [
                data_manager.link_pairs[i]
                for i in range(len(data_manager.link_pairs))
                if data_manager.current_edge_mask[i]
            ]

        # Define font sizes
        small_font = {"fontsize": 9}
        medium_font = {"fontsize": 12}

        # Idea 1: Circular flow diagram
        self._create_circular_flow_diagram(
            self.flow_ideas_ax1,
            visible_links,
            data_manager.node_ids,
            data_manager.node_clusters,
            data_manager.nodes_per_layer,
            data_manager.layers,
            small_font,
            medium_font,
            data_manager.visible_layers,
            data_manager.cluster_colors,
        )
        self.flow_ideas_ax1.set_title(
            "Idea 1: Circular Flow Diagram", fontsize=medium_font["fontsize"]
        )

        # Idea 2: Force-directed graph with layers as regions
        self._create_force_directed_regions(
            self.flow_ideas_ax2,
            visible_links,
            data_manager.node_ids,
            data_manager.node_clusters,
            data_manager.nodes_per_layer,
            data_manager.layers,
            small_font,
            medium_font,
            data_manager.visible_layers,
            data_manager.cluster_colors,
        )
        self.flow_ideas_ax2.set_title(
            "Idea 2: Force-Directed Regions", fontsize=medium_font["fontsize"]
        )

        # Idea 3: Alluvial diagram
        self._create_alluvial_diagram(
            self.flow_ideas_ax3,
            visible_links,
            data_manager.node_ids,
            data_manager.node_clusters,
            data_manager.nodes_per_layer,
            data_manager.layers,
            small_font,
            medium_font,
            data_manager.visible_layers,
            data_manager.cluster_colors,
        )
        self.flow_ideas_ax3.set_title(
            "Idea 3: Alluvial Diagram", fontsize=medium_font["fontsize"]
        )

        # Idea 4: Radial network
        self._create_radial_network(
            self.flow_ideas_ax4,
            visible_links,
            data_manager.node_ids,
            data_manager.node_clusters,
            data_manager.nodes_per_layer,
            data_manager.layers,
            small_font,
            medium_font,
            data_manager.visible_layers,
            data_manager.cluster_colors,
        )
        self.flow_ideas_ax4.set_title(
            "Idea 4: Radial Network", fontsize=medium_font["fontsize"]
        )

        self.flow_ideas_figure.tight_layout()
        self.flow_ideas_canvas.draw()

    def update_lc1_heatmap(self, data_manager):
        """Update the LC1 heatmap with the current data"""
        # This is a stub method that can be implemented later
        pass

    def update_lc2_distribution(self, data_manager):
        """Update the LC2 distribution with the current data"""
        # This is a stub method that can be implemented later
        pass

    def update_lc3_layer_distribution(self, data_manager):
        """Update the LC3 layer distribution with the current data"""
        # This is a stub method that can be implemented later
        pass

    def update_lc5_cluster_network(self, data_manager):
        """Update the LC5 cluster network with the current data"""
        # This is a stub method that can be implemented later
        pass

    def update_lc6_sankey(self, data_manager):
        """Update the LC6 sankey diagram with the current data"""
        # This is a stub method that can be implemented later
        pass

    def update_lc7_connectivity(self, data_manager):
        """Update the LC7 connectivity matrix with the current data"""
        # This is a stub method that can be implemented later
        pass

    def update_lc8_chord(self, data_manager):
        """Update the LC8 chord diagram with the current data"""
        # This is a stub method that can be implemented later
        pass

    def update_lc9_density(self, data_manager):
        """Update the LC9 density heatmap with the current data"""
        # This is a stub method that can be implemented later
        pass

    def update_lc10_cooccurrence(self, data_manager):
        """Update the LC10 cooccurrence network with the current data"""
        # This is a stub method that can be implemented later
        pass

    def update_lc11_normalized(self, data_manager):
        """Update the LC11 normalized heatmap with the current data"""
        # This is a stub method that can be implemented later
        pass

    def update_lc13_bubble(self, data_manager):
        """Update the LC13 bubble chart with the current data"""
        # This is a stub method that can be implemented later
        pass
