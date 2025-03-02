from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from charts.layer_connectivity import create_layer_connectivity_chart
from charts.cluster_distribution import create_cluster_distribution_chart
from charts.betweenness_centrality import create_betweenness_centrality_chart
from charts.interlayer_graph import create_interlayer_graph
from charts.layer_activity import create_layer_activity_chart
from charts.layer_similarity import create_layer_similarity_chart

from .base_panel import BaseStatsPanel


class MainStatsPanel(BaseStatsPanel):
    """Panel for the main network statistics"""

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Add checkbox to enable/disable all charts
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(5, 5, 5, 0)
        self.enable_checkbox = QCheckBox("Enable Charts")
        self.enable_checkbox.setChecked(True)
        self.enable_checkbox.stateChanged.connect(self.on_state_changed)
        controls_layout.addWidget(self.enable_checkbox)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

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

        # Initialize left column subplots
        self.layer_connectivity_ax = self.left_figure.add_subplot(311)
        self.cluster_distribution_ax = self.left_figure.add_subplot(312)
        self.layer_activity_ax = self.left_figure.add_subplot(313)

        # Initialize right column subplots
        self.betweenness_centrality_ax = self.right_figure.add_subplot(311)
        self.interlayer_graph_ax = self.right_figure.add_subplot(312)
        self.layer_similarity_ax = self.right_figure.add_subplot(313)

    def on_state_changed(self, state):
        """Handle enable/disable state change"""
        if state and hasattr(self, "_current_data"):
            self.update_stats(self._current_data)
        elif not state:
            # Clear all figures when disabled
            self.left_figure.clear()
            self.right_figure.clear()

            # Add disabled message to both figures
            self.left_figure.text(
                0.5, 0.5, "Charts disabled", ha="center", va="center", fontsize=12
            )
            self.right_figure.text(
                0.5, 0.5, "Charts disabled", ha="center", va="center", fontsize=12
            )

            # Draw canvases
            self.left_canvas.draw()
            self.right_canvas.draw()

    def update_stats(self, data_manager):
        self._current_data = data_manager

        # Only update charts if enabled
        if not self.enable_checkbox.isChecked():
            self.on_state_changed(False)
            return

        # Clear all figures
        self.left_figure.clear()
        self.right_figure.clear()

        # Re-create left column subplots
        self.layer_connectivity_ax = self.left_figure.add_subplot(311)
        self.cluster_distribution_ax = self.left_figure.add_subplot(312)
        self.layer_activity_ax = self.left_figure.add_subplot(313)

        # Re-create right column subplots
        self.betweenness_centrality_ax = self.right_figure.add_subplot(311)
        self.interlayer_graph_ax = self.right_figure.add_subplot(312)
        self.layer_similarity_ax = self.right_figure.add_subplot(313)

        small_font = {"fontsize": 6}
        medium_font = {"fontsize": 7}

        # Get filtered data from the data manager using optimized methods
        # This is more efficient than accessing raw properties and filtering manually
        filtered_data = data_manager.get_filtered_data_for_vispy()
        
        # Extract the data we need for visualization
        node_positions = filtered_data['node_positions']
        node_colors = filtered_data['node_colors']
        edge_connections = filtered_data['edge_connections']
        
        # Get additional data needed for statistics
        layers = data_manager.layers
        visible_layer_indices = data_manager.visible_layers
        layer_colors = data_manager.layer_colors
        
        # Get layer connections from data manager (already filtered)
        layer_connections = data_manager.get_layer_connections(filter_to_visible=True)
        
        # Get NetworkX graph for analysis (already filtered)
        G = data_manager.get_networkx_graph(visible_only=True)
        
        # Get interlayer edge counts for statistics
        interlayer_edge_counts = data_manager.get_interlayer_edge_counts()

        # Create layer connectivity chart
        create_layer_connectivity_chart(
            self.layer_connectivity_ax,
            edge_connections,  # visible_links
            data_manager.nodes_per_layer,  # nodes_per_layer
            [layers[i] for i in visible_layer_indices],  # layers
            small_font,  # small_font
            medium_font,  # medium_font
            {i: idx for idx, i in enumerate(visible_layer_indices)}  # layer_index_map
        )

        # Create cluster distribution chart
        # Extract cluster information from the NetworkX graph
        node_ids = [data["id"] for _, data in G.nodes(data=True)]
        node_clusters_dict = {data["id"]: data["cluster"] for _, data in G.nodes(data=True)}
        create_cluster_distribution_chart(
            self.cluster_distribution_ax,
            node_ids,
            node_clusters_dict,
            small_font,
            medium_font,
            {cluster: data_manager.cluster_colors.get(cluster, "#CCCCCC") for cluster in data_manager.unique_clusters}
        )

        # Create betweenness centrality chart
        create_betweenness_centrality_chart(
            self.betweenness_centrality_ax,
            layer_connections,
            layers,
            visible_layer_indices,
            small_font,
            medium_font
        )

        # Create interlayer graph
        create_interlayer_graph(
            self.interlayer_graph_ax,
            layer_connections,
            [layers[i] for i in visible_layer_indices],
            small_font,
            medium_font,
            layer_colors={layers[i]: layer_colors.get(layers[i], "#CCCCCC") for i in visible_layer_indices}
        )

        # Create layer activity chart
        create_layer_activity_chart(
            self.layer_activity_ax,
            edge_connections,  # visible_links
            data_manager.nodes_per_layer,  # nodes_per_layer
            [layers[i] for i in visible_layer_indices],  # layers
            small_font,
            medium_font,
            {i: idx for idx, i in enumerate(visible_layer_indices)}  # layer_index_map
        )

        # Create layer similarity chart
        create_layer_similarity_chart(
            self.layer_similarity_ax,
            layer_connections,
            [layers[i] for i in visible_layer_indices],
            small_font,
            medium_font
        )

        # Adjust layout and draw
        self.left_figure.tight_layout()
        self.right_figure.tight_layout()
        self.left_canvas.draw()
        self.right_canvas.draw()
