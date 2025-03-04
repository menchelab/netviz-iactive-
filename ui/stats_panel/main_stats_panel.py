from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

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

        # Set up fonts
        small_font = {"fontsize": 6}
        medium_font = {"fontsize": 7}

        # --- LEFT COLUMN CHARTS ---

        # 1. Layer connectivity matrix
        self._create_layer_connectivity_chart(
            self.layer_connectivity_ax, data_manager, small_font, medium_font
        )

        # 2. Cluster distribution
        self._create_cluster_distribution_chart(
            self.cluster_distribution_ax, data_manager, small_font, medium_font
        )

        # 3. Layer activity chart
        self._create_layer_activity_chart(
            self.layer_activity_ax, data_manager, small_font, medium_font
        )

        # --- RIGHT COLUMN CHARTS ---

        # 1. Betweenness centrality analysis
        self._create_betweenness_centrality_chart(
            self.betweenness_centrality_ax, data_manager, small_font, medium_font
        )

        # 2. Interlayer graph visualization
        self._create_interlayer_graph(
            self.interlayer_graph_ax, data_manager, small_font, medium_font
        )

        # 3. Layer similarity dendrogram
        self._create_layer_similarity_chart(
            self.layer_similarity_ax, data_manager, small_font, medium_font
        )

        # Apply tight layout to all figures
        self.left_figure.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
        self.right_figure.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)

        # Draw all canvases
        self.left_canvas.draw()
        self.right_canvas.draw()

    def _create_layer_connectivity_chart(self, ax, data_manager, small_font, medium_font):
        """Create layer connectivity matrix chart"""
        # Get layer connections from data manager
        layer_connections = data_manager.get_layer_connections(filter_to_visible=True)
        
        # Get visible layers
        visible_layer_indices = np.where(data_manager.visible_layers)[0]
        visible_layers = [data_manager.layers[i] for i in visible_layer_indices]
        
        # Create heatmap
        im = ax.imshow(layer_connections, cmap='viridis')
        
        # Set labels
        ax.set_xticks(np.arange(len(visible_layers)))
        ax.set_yticks(np.arange(len(visible_layers)))
        ax.set_xticklabels(visible_layers, **small_font)
        ax.set_yticklabels(visible_layers, **small_font)
        
        # Rotate x labels
        plt = ax.figure.canvas.manager.canvas.figure.plt
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add title
        ax.set_title("Layer Connectivity Matrix", **medium_font)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=6)
        
        return im, layer_connections

    def _create_cluster_distribution_chart(self, ax, data_manager, small_font, medium_font):
        """Create cluster distribution chart"""
        # Skip if no clusters
        if not hasattr(data_manager, 'node_clusters') or data_manager.node_clusters is None:
            ax.text(0.5, 0.5, "No cluster data available", 
                   ha='center', va='center', **medium_font)
            return
            
        # Get visible nodes
        node_mask = data_manager.current_node_mask
        visible_node_indices = np.where(node_mask)[0]
        
        # Count clusters
        cluster_counts = {}
        for idx in visible_node_indices:
            cluster = data_manager.node_clusters[idx]
            if cluster not in cluster_counts:
                cluster_counts[cluster] = 0
            cluster_counts[cluster] += 1
        
        # Sort clusters by count
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
        clusters = [c[0] for c in sorted_clusters]
        counts = [c[1] for c in sorted_clusters]
        
        # Create bar chart
        colors = [data_manager.cluster_colors.get(c, "#CCCCCC") for c in clusters]
        ax.bar(clusters, counts, color=colors)
        
        # Set labels
        ax.set_xlabel("Cluster", **small_font)
        ax.set_ylabel("Count", **small_font)
        ax.set_title("Cluster Distribution", **medium_font)
        
        # Rotate x labels if many clusters
        if len(clusters) > 5:
            plt = ax.figure.canvas.manager.canvas.figure.plt
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        ax.tick_params(axis='both', which='major', labelsize=6)

    def _create_layer_activity_chart(self, ax, data_manager, small_font, medium_font):
        """Create layer activity chart"""
        # Get layer connections
        layer_connections = data_manager.get_layer_connections(filter_to_visible=True)
        
        # Get visible layers
        visible_layer_indices = np.where(data_manager.visible_layers)[0]
        visible_layers = [data_manager.layers[i] for i in visible_layer_indices]
        
        # Calculate activity (sum of connections)
        activity = np.sum(layer_connections, axis=1)
        
        # Create bar chart
        colors = [data_manager.layer_colors.get(layer, "#CCCCCC") for layer in visible_layers]
        ax.bar(visible_layers, activity, color=colors)
        
        # Set labels
        ax.set_xlabel("Layer", **small_font)
        ax.set_ylabel("Activity", **small_font)
        ax.set_title("Layer Activity", **medium_font)
        
        # Rotate x labels if many layers
        if len(visible_layers) > 5:
            plt = ax.figure.canvas.manager.canvas.figure.plt
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        ax.tick_params(axis='both', which='major', labelsize=6)

    def _create_betweenness_centrality_chart(self, ax, data_manager, small_font, medium_font):
        """Create betweenness centrality chart"""
        # Get layer connections
        layer_connections = data_manager.get_layer_connections(filter_to_visible=True)
        
        # Get visible layers
        visible_layer_indices = np.where(data_manager.visible_layers)[0]
        visible_layers = [data_manager.layers[i] for i in visible_layer_indices]
        
        # Calculate betweenness centrality (simplified)
        n = len(visible_layers)
        if n < 3:
            ax.text(0.5, 0.5, "Need at least 3 layers for betweenness centrality", 
                   ha='center', va='center', **medium_font)
            return
            
        # Simple approximation of betweenness centrality
        centrality = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    # Check if i is on a path from j to k
                    if layer_connections[j, i] > 0 and layer_connections[i, k] > 0:
                        centrality[i] += 1
        
        # Normalize
        if np.max(centrality) > 0:
            centrality = centrality / np.max(centrality)
        
        # Create bar chart
        colors = [data_manager.layer_colors.get(layer, "#CCCCCC") for layer in visible_layers]
        ax.bar(visible_layers, centrality, color=colors)
        
        # Set labels
        ax.set_xlabel("Layer", **small_font)
        ax.set_ylabel("Centrality", **small_font)
        ax.set_title("Layer Betweenness Centrality", **medium_font)
        
        # Rotate x labels if many layers
        if len(visible_layers) > 5:
            plt = ax.figure.canvas.manager.canvas.figure.plt
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        ax.tick_params(axis='both', which='major', labelsize=6)

    def _create_interlayer_graph(self, ax, data_manager, small_font, medium_font):
        """Create interlayer graph visualization"""
        # Get layer connections
        layer_connections = data_manager.get_layer_connections(filter_to_visible=True)
        
        # Get visible layers
        visible_layer_indices = np.where(data_manager.visible_layers)[0]
        visible_layers = [data_manager.layers[i] for i in visible_layer_indices]
        
        # Simple network visualization
        n = len(visible_layers)
        if n < 2:
            ax.text(0.5, 0.5, "Need at least 2 layers for interlayer graph", 
                   ha='center', va='center', **medium_font)
            return
        
        # Create positions in a circle
        pos = {}
        for i, layer in enumerate(visible_layers):
            angle = 2 * np.pi * i / n
            pos[i] = (np.cos(angle), np.sin(angle))
        
        # Draw nodes
        colors = [data_manager.layer_colors.get(layer, "#CCCCCC") for layer in visible_layers]
        for i, layer in enumerate(visible_layers):
            ax.scatter(pos[i][0], pos[i][1], s=100, color=colors[i], edgecolors='black', zorder=2)
            ax.text(pos[i][0]*1.1, pos[i][1]*1.1, layer, ha='center', va='center', **small_font)
        
        # Draw edges
        for i in range(n):
            for j in range(i+1, n):
                if layer_connections[i, j] > 0:
                    # Scale line width by connection strength
                    width = 0.5 + 2 * layer_connections[i, j] / np.max(layer_connections)
                    ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], 
                           linewidth=width, color='gray', alpha=0.7, zorder=1)
        
        # Set limits and title
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title("Interlayer Connections", **medium_font)
        ax.set_aspect('equal')
        ax.axis('off')

    def _create_layer_similarity_chart(self, ax, data_manager, small_font, medium_font):
        """Create layer similarity chart"""
        # Get layer connections
        layer_connections = data_manager.get_layer_connections(filter_to_visible=True)
        
        # Get visible layers
        visible_layer_indices = np.where(data_manager.visible_layers)[0]
        visible_layers = [data_manager.layers[i] for i in visible_layer_indices]
        
        n = len(visible_layers)
        if n < 2:
            ax.text(0.5, 0.5, "Need at least 2 layers for similarity analysis", 
                   ha='center', va='center', **medium_font)
            return
        
        # Calculate similarity matrix (normalized connections)
        similarity = layer_connections.copy().astype(float)
        row_sums = similarity.sum(axis=1)
        for i in range(n):
            if row_sums[i] > 0:
                similarity[i, :] /= row_sums[i]
        
        # Create heatmap
        im = ax.imshow(similarity, cmap='viridis')
        
        # Set labels
        ax.set_xticks(np.arange(len(visible_layers)))
        ax.set_yticks(np.arange(len(visible_layers)))
        ax.set_xticklabels(visible_layers, **small_font)
        ax.set_yticklabels(visible_layers, **small_font)
        
        # Rotate x labels
        plt = ax.figure.canvas.manager.canvas.figure.plt
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add title
        ax.set_title("Layer Similarity Matrix", **medium_font)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=6)
        
        return im
