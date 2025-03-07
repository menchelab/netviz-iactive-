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
        self.enable_checkbox.setChecked(False)
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

        # Get data from manager
        node_positions = data_manager.node_positions
        link_pairs = data_manager.link_pairs
        node_ids = data_manager.node_ids
        layers = data_manager.layers
        node_clusters = data_manager.node_clusters
        node_mask = data_manager.current_node_mask
        edge_mask = data_manager.current_edge_mask
        visible_layer_indices = data_manager.visible_layers
        layer_colors = data_manager.layer_colors

        # Get layer connections from data manager (already filtered)
        layer_connections = data_manager.get_layer_connections(filter_to_visible=True)

        # Get visible nodes and edges
        visible_nodes = [i for i, mask in enumerate(node_mask) if mask]
        visible_edges = [i for i, mask in enumerate(edge_mask) if mask]
        visible_links = [link_pairs[i] for i in visible_edges]

        # Calculate nodes per layer
        nodes_per_layer = len(node_positions) // len(layers)

        # Get visible layers list for filtered views
        visible_layers = [layers[i] for i in visible_layer_indices] if visible_layer_indices else []

        # Create mapping from original layer indices to filtered indices
        layer_index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(visible_layer_indices)} if visible_layer_indices else {}

        # --- LEFT COLUMN CHARTS ---

        # 1. Layer connectivity matrix
        im, layer_connections = create_layer_connectivity_chart(
            self.layer_connectivity_ax,
            visible_links,
            nodes_per_layer,
            visible_layers,  # Use visible layers instead of all layers
            small_font,
            medium_font,
            layer_index_map,  # Pass the mapping
        )
        cbar = self.left_figure.colorbar(
            im, ax=self.layer_connectivity_ax, fraction=0.046, pad=0.02
        )
        cbar.ax.tick_params(labelsize=6)  # Smaller colorbar ticks

        # 2. Cluster distribution
        visible_node_ids = [node_ids[i] for i in visible_nodes]
        create_cluster_distribution_chart(
            self.cluster_distribution_ax,
            visible_node_ids,
            node_clusters,
            small_font,
            medium_font,
            data_manager.cluster_colors,
        )

        # 3. Layer activity chart
        create_layer_activity_chart(
            self.layer_activity_ax,
            visible_links,
            nodes_per_layer,
            visible_layers,  # Use visible layers instead of all layers
            small_font,
            medium_font,
            layer_index_map,  # Pass the mapping
        )

        # --- RIGHT COLUMN CHARTS ---

        # 1. Betweenness centrality analysis
        create_betweenness_centrality_chart(
            self.betweenness_centrality_ax,
            layer_connections,
            visible_layers,  # Already filtered list of layers
            list(range(len(visible_layers))),  # Use sequential indices since data is already filtered
            small_font,
            medium_font,
        )

        # 2. Interlayer graph visualization
        create_interlayer_graph(
            self.interlayer_graph_ax,
            layer_connections,
            visible_layers,  # Already filtered list of layers
            small_font,
            medium_font,
            list(range(len(visible_layers))),  # Use sequential indices since data is already filtered
            layer_colors,
        )

        # 3. Layer similarity dendrogram
        create_layer_similarity_chart(
            self.layer_similarity_ax,
            layer_connections,
            visible_layers,  # Use visible layers instead of all layers
            small_font,
            medium_font,
        )

        # Apply tight layout to all figures
        self.left_figure.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
        self.right_figure.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)

        # Draw all canvases
        self.left_canvas.draw()
        self.right_canvas.draw()
