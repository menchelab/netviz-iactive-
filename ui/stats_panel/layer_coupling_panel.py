from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QCheckBox, QComboBox, 
                             QLabel, QGroupBox)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import networkx as nx
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

from .base_panel import BaseStatsPanel

class LayerCouplingPanel(BaseStatsPanel):
    """Panel for Layer Coupling Analysis including Structural Coupling Index and Hierarchical Layer Organization"""
    
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
        
        # Add coupling metric dropdown
        controls_layout.addWidget(QLabel("Coupling Metric:"))
        self.coupling_metric_dropdown = QComboBox()
        self.coupling_metric_dropdown.addItems([
            "Edge Density", "Information Flow", "Topological Overlap"
        ])
        self.coupling_metric_dropdown.currentTextChanged.connect(self.on_metric_changed)
        controls_layout.addWidget(self.coupling_metric_dropdown)
        
        # Add hierarchical clustering method dropdown
        controls_layout.addWidget(QLabel("Clustering Method:"))
        self.clustering_method_dropdown = QComboBox()
        self.clustering_method_dropdown.addItems([
            "Single", "Complete", "Average", "Ward"
        ])
        self.clustering_method_dropdown.currentTextChanged.connect(self.on_method_changed)
        controls_layout.addWidget(self.clustering_method_dropdown)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Create figure for Layer Coupling Analysis
        self.figure = Figure(figsize=(8, 10), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Initialize subplots with GridSpec for custom layout
        # First row takes 30% of height, second row takes 70%
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 7])
        
        # Top row has two equal columns
        gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0])
        self.coupling_heatmap_ax = self.figure.add_subplot(gs_top[0])  # Coupling heatmap
        self.coupling_bar_ax = self.figure.add_subplot(gs_top[1])      # Coupling bar chart
        
        # Bottom row has two equal columns
        gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1])
        self.dendrogram_ax = self.figure.add_subplot(gs_bottom[0])     # Hierarchical dendrogram
        self.circular_ax = self.figure.add_subplot(gs_bottom[1])       # Circular hierarchical layout
        
        # Store current data
        self._current_data = None
    
    def on_state_changed(self, state):
        """Handle enable/disable state change"""
        if self._current_data and state:
            self.update_visualization()
        elif not state:
            # Clear the visualization when disabled
            self.clear_visualization("Layer coupling analysis disabled")
    
    def on_metric_changed(self, metric):
        """Handle coupling metric change"""
        if self._current_data and self.enable_checkbox.isChecked():
            self.update_visualization()
    
    def on_method_changed(self, method):
        """Handle clustering method change"""
        if self._current_data and self.enable_checkbox.isChecked():
            self.update_visualization()
    
    def clear_visualization(self, message=""):
        """Clear all axes and display a message"""
        self.figure.clear()
        
        # Recreate the GridSpec layout
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 7])
        
        # Top row has two equal columns
        gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0])
        self.coupling_heatmap_ax = self.figure.add_subplot(gs_top[0])
        self.coupling_bar_ax = self.figure.add_subplot(gs_top[1])
        
        # Bottom row has two equal columns
        gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1])
        self.dendrogram_ax = self.figure.add_subplot(gs_bottom[0])
        self.circular_ax = self.figure.add_subplot(gs_bottom[1])
        
        if message:
            for ax in [self.coupling_heatmap_ax, self.coupling_bar_ax, 
                      self.dendrogram_ax, self.circular_ax]:
                ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12)
                ax.axis('off')
        
        self.canvas.draw()
    
    def update_visualization(self):
        """Update all visualizations based on current settings"""
        if not self._current_data:
            return
            
        self.clear_visualization()
        
        # Unpack stored data
        layer_connections, layers, medium_font, large_font, visible_layer_indices, layer_colors = self._current_data
        
        # Filter the layer_connections matrix to only include visible layers
        if visible_layer_indices and len(visible_layer_indices) > 0:
            filtered_connections = layer_connections[np.ix_(visible_layer_indices, visible_layer_indices)]
            filtered_layers = [layers[i] for i in visible_layer_indices]
            filtered_colors = {layer: layer_colors.get(layer, 'skyblue') for layer in filtered_layers}
        else:
            # No visible layers, display a message
            self.clear_visualization("No visible layers to analyze")
            return
        
        # Calculate structural coupling index
        coupling_matrix = self.calculate_structural_coupling(
            filtered_connections, 
            metric=self.coupling_metric_dropdown.currentText()
        )
        
        # Create coupling visualizations
        self.create_coupling_visualizations(
            coupling_matrix, 
            filtered_layers, 
            medium_font, 
            large_font,
            filtered_colors
        )
        
        # Create hierarchical organization visualizations
        self.create_hierarchical_visualizations(
            coupling_matrix, 
            filtered_layers, 
            medium_font, 
            large_font,
            filtered_colors,
            method=self.clustering_method_dropdown.currentText().lower()
        )
        
        self.figure.tight_layout(pad=1.0)
        self.canvas.draw()
    
    def calculate_structural_coupling(self, layer_connections, metric="Edge Density"):
        """
        Calculate structural coupling index between layers
        
        Parameters:
        -----------
        layer_connections : numpy.ndarray
            Matrix of connection counts between layers
        metric : str
            The coupling metric to use
            
        Returns:
        --------
        numpy.ndarray
            Matrix of coupling indices between layers
        """
        n_layers = layer_connections.shape[0]
        coupling_matrix = np.zeros((n_layers, n_layers))
        
        if metric == "Edge Density":
            # Calculate edge density between layers
            # Normalize by the maximum possible number of connections
            for i in range(n_layers):
                for j in range(n_layers):
                    if i == j:
                        coupling_matrix[i, j] = 1.0  # Self-coupling is 1
                    else:
                        # Number of connections between layers i and j
                        connections = layer_connections[i, j]
                        # Maximum possible connections (if all nodes were connected)
                        max_connections = max(1, layer_connections[i, i] * layer_connections[j, j])
                        coupling_matrix[i, j] = connections / max_connections
        
        elif metric == "Information Flow":
            # Create a graph from layer connections
            G = nx.Graph()
            for i in range(n_layers):
                G.add_node(i)
            
            for i in range(n_layers):
                for j in range(i+1, n_layers):
                    if layer_connections[i, j] > 0:
                        G.add_edge(i, j, weight=layer_connections[i, j])
            
            # Calculate information flow using random walk
            try:
                # Use personalized PageRank as a proxy for information flow
                for i in range(n_layers):
                    personalization = {j: 0.0 for j in range(n_layers)}
                    personalization[i] = 1.0
                    
                    pr = nx.pagerank(G, alpha=0.85, personalization=personalization)
                    for j in range(n_layers):
                        coupling_matrix[i, j] = pr.get(j, 0)
            except:
                # Fallback if PageRank fails
                print("Warning: PageRank calculation failed, using edge weights")
                for i in range(n_layers):
                    for j in range(n_layers):
                        if i == j:
                            coupling_matrix[i, j] = 1.0
                        elif G.has_edge(i, j):
                            coupling_matrix[i, j] = G[i][j]['weight'] / max(1, layer_connections.max())
        
        elif metric == "Topological Overlap":
            # Calculate topological overlap
            # For each pair of layers, calculate the overlap in their connections
            for i in range(n_layers):
                for j in range(n_layers):
                    if i == j:
                        coupling_matrix[i, j] = 1.0
                    else:
                        # Get connections for layers i and j
                        i_connections = set(np.where(layer_connections[i, :] > 0)[0])
                        j_connections = set(np.where(layer_connections[j, :] > 0)[0])
                        
                        # Calculate Jaccard similarity
                        if len(i_connections) == 0 and len(j_connections) == 0:
                            coupling_matrix[i, j] = 0.0
                        else:
                            intersection = len(i_connections.intersection(j_connections))
                            union = len(i_connections.union(j_connections))
                            coupling_matrix[i, j] = intersection / union
        
        return coupling_matrix
    
    def create_coupling_visualizations(self, coupling_matrix, layers, medium_font, large_font, layer_colors):
        """Create visualizations for structural coupling index"""
        # Create heatmap of coupling matrix
        im = self.coupling_heatmap_ax.imshow(coupling_matrix, cmap='viridis', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = self.coupling_heatmap_ax.figure.colorbar(im, ax=self.coupling_heatmap_ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('Coupling Strength', fontsize=8)
        
        # Add labels
        self.coupling_heatmap_ax.set_xticks(range(len(layers)))
        self.coupling_heatmap_ax.set_yticks(range(len(layers)))
        self.coupling_heatmap_ax.set_xticklabels(layers, rotation=90, fontsize=8)
        self.coupling_heatmap_ax.set_yticklabels(layers, fontsize=8)
        
        metric = self.coupling_metric_dropdown.currentText()
        self.coupling_heatmap_ax.set_title(f'Layer Coupling Matrix ({metric})', **large_font)
        
        # Calculate overall coupling score for each layer
        coupling_scores = coupling_matrix.sum(axis=1) - 1  # Subtract self-coupling
        
        # Sort layers by coupling score
        sorted_indices = np.argsort(coupling_scores)[::-1]
        sorted_layers = [layers[i] for i in sorted_indices]
        sorted_scores = coupling_scores[sorted_indices]
        
        # Create bar chart of coupling scores
        bar_colors = []
        if layer_colors:
            for layer in sorted_layers:
                if layer in layer_colors:
                    bar_colors.append(layer_colors[layer])
                else:
                    bar_colors.append('skyblue')
        else:
            bar_colors = ['skyblue' for _ in range(len(sorted_layers))]
        
        bars = self.coupling_bar_ax.barh(sorted_layers, sorted_scores, color=bar_colors)
        self.coupling_bar_ax.set_title(f'Layer Coupling Scores', **large_font)
        self.coupling_bar_ax.set_xlabel('Coupling Score', **medium_font)
        self.coupling_bar_ax.set_ylabel('Layer', **medium_font)
        
        # Add value labels to the bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x_pos = width * 1.01
            self.coupling_bar_ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                                    va='center', **medium_font)
    
    def create_hierarchical_visualizations(self, coupling_matrix, layers, medium_font, large_font, 
                                         layer_colors, method='average'):
        """Create visualizations for hierarchical layer organization"""
        # Convert coupling matrix to distance matrix
        # Higher coupling = lower distance
        distance_matrix = 1 - coupling_matrix
        np.fill_diagonal(distance_matrix, 0)  # Ensure diagonal is 0
        
        # Ensure the distance matrix is symmetric
        distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)
        
        # Calculate linkage matrix for hierarchical clustering
        try:
            # Convert to condensed distance matrix (required by hierarchy.linkage)
            condensed_dist = squareform(distance_matrix)
            
            # Calculate linkage matrix
            linkage_matrix = hierarchy.linkage(condensed_dist, method=method)
            
            # Create dendrogram
            dendrogram = hierarchy.dendrogram(
                linkage_matrix,
                labels=layers,
                orientation='right',
                ax=self.dendrogram_ax
            )
            
            self.dendrogram_ax.set_title(f'Layer Hierarchy ({method.capitalize()} Linkage)', **large_font)
            
            # Create circular layout based on hierarchical clustering
            # Create a graph
            G = nx.Graph()
            
            # Add nodes
            for i, layer in enumerate(layers):
                G.add_node(i, name=layer)
            
            # Add edges based on coupling
            for i in range(len(layers)):
                for j in range(i+1, len(layers)):
                    if coupling_matrix[i, j] > 0:
                        G.add_edge(i, j, weight=coupling_matrix[i, j])
            
            # Get the order of leaves from the dendrogram
            leaf_order = dendrogram['leaves']
            
            # Create a circular layout with nodes arranged according to dendrogram order
            pos = {}
            num_nodes = len(leaf_order)
            for i, leaf_idx in enumerate(leaf_order):
                angle = 2 * np.pi * i / num_nodes
                pos[leaf_idx] = (np.cos(angle), np.sin(angle))
            
            # Prepare node colors
            node_colors = []
            if layer_colors:
                for layer in layers:
                    if layer in layer_colors:
                        node_colors.append(layer_colors[layer])
                    else:
                        node_colors.append('skyblue')
            else:
                node_colors = ['skyblue' for _ in range(len(layers))]
            
            # Draw the graph
            nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, 
                                  alpha=0.8, ax=self.circular_ax)
            
            # Draw edges with width and color based on coupling strength
            edges = G.edges(data=True)
            if edges:
                max_weight = max([d['weight'] for _, _, d in edges])
                edge_widths = [d['weight'] * 5 / max_weight for _, _, d in edges]
                edge_colors = [plt.cm.plasma(d['weight'] / max_weight) for _, _, d in edges]
                
                nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, 
                                      alpha=0.5, ax=self.circular_ax)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, labels={i: layers[i] for i in range(len(layers))}, 
                                   font_size=8, ax=self.circular_ax)
            
            self.circular_ax.set_title('Hierarchical Circular Layout', **large_font)
            self.circular_ax.axis('off')
            
        except Exception as e:
            print(f"Warning: Error creating hierarchical visualizations: {e}")
            self.dendrogram_ax.text(0.5, 0.5, "Hierarchical clustering failed", 
                                  ha='center', va='center', fontsize=12)
            self.dendrogram_ax.axis('off')
            
            self.circular_ax.text(0.5, 0.5, "Circular layout failed", 
                                ha='center', va='center', fontsize=12)
            self.circular_ax.axis('off')
    
    def update_stats(self, data_manager):
        """Update the Layer Coupling Analysis with current data"""
        # Clear figure
        self.clear_visualization()
        
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
            self.update_visualization()
        else:
            self.clear_visualization("Layer coupling analysis disabled") 