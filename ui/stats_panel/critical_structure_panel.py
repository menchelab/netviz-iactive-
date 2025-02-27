from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QCheckBox, QComboBox, QLabel, QSlider
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy import stats

from .base_panel import BaseStatsPanel

class CriticalStructurePanel(BaseStatsPanel):
    """Panel for Critical Structure Analysis including Critical Layer Identification and Anomaly Detection"""
    
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
        
        # Add criticality metric dropdown
        controls_layout.addWidget(QLabel("Criticality Metric:"))
        self.criticality_metric_dropdown = QComboBox()
        self.criticality_metric_dropdown.addItems([
            "Connectivity Impact", "Centrality Impact", "Information Flow Impact"
        ])
        self.criticality_metric_dropdown.currentTextChanged.connect(self.on_metric_changed)
        controls_layout.addWidget(self.criticality_metric_dropdown)
        
        # Add anomaly threshold slider
        controls_layout.addWidget(QLabel("Anomaly Threshold:"))
        self.anomaly_threshold_slider = QSlider(Qt.Horizontal)
        self.anomaly_threshold_slider.setMinimum(1)
        self.anomaly_threshold_slider.setMaximum(30)
        self.anomaly_threshold_slider.setValue(10)  # Default value (1.0 standard deviations)
        self.anomaly_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.anomaly_threshold_slider.setTickInterval(5)
        self.anomaly_threshold_slider.valueChanged.connect(self.on_threshold_changed)
        controls_layout.addWidget(self.anomaly_threshold_slider)
        
        # Add threshold value label
        self.threshold_label = QLabel("1.0σ")
        controls_layout.addWidget(self.threshold_label)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Create figure for Critical Structure Analysis
        self.figure = Figure(figsize=(8, 10), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Initialize subplots with GridSpec for custom layout
        # First row takes 30% of height, second row takes 70%
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 7])
        
        # Top row has two equal columns
        gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0])
        self.criticality_bar_ax = self.figure.add_subplot(gs_top[0])  # Criticality bar chart
        self.impact_ax = self.figure.add_subplot(gs_top[1])           # Impact visualization
        
        # Bottom row has two equal columns
        gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1])
        self.anomaly_ax = self.figure.add_subplot(gs_bottom[0])       # Anomaly detection
        self.network_ax = self.figure.add_subplot(gs_bottom[1])       # Network visualization
        
        # Store current data
        self._current_data = None
    
    def on_state_changed(self, state):
        """Handle enable/disable state change"""
        if self._current_data and state:
            self.update_visualization()
        elif not state:
            # Clear the visualization when disabled
            self.clear_visualization("Critical structure analysis disabled")
    
    def on_metric_changed(self, metric):
        """Handle criticality metric change"""
        if self._current_data and self.enable_checkbox.isChecked():
            self.update_visualization()
    
    def on_threshold_changed(self, value):
        """Handle anomaly threshold change"""
        # Convert slider value to standard deviations (0.1 to 3.0)
        threshold = value / 10.0
        self.threshold_label.setText(f"{threshold:.1f}σ")
        
        if self._current_data and self.enable_checkbox.isChecked():
            self.update_visualization()
    
    def clear_visualization(self, message=""):
        """Clear all axes and display a message"""
        self.figure.clear()
        
        # Recreate the GridSpec layout
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 7])
        
        # Top row has two equal columns
        gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0])
        self.criticality_bar_ax = self.figure.add_subplot(gs_top[0])
        self.impact_ax = self.figure.add_subplot(gs_top[1])
        
        # Bottom row has two equal columns
        gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1])
        self.anomaly_ax = self.figure.add_subplot(gs_bottom[0])
        self.network_ax = self.figure.add_subplot(gs_bottom[1])
        
        if message:
            for ax in [self.criticality_bar_ax, self.impact_ax, 
                      self.anomaly_ax, self.network_ax]:
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
        
        # Calculate criticality scores
        criticality_scores, impact_data = self.identify_critical_layers(
            filtered_connections, 
            filtered_layers,
            metric=self.criticality_metric_dropdown.currentText()
        )
        
        # Create criticality visualizations
        self.create_criticality_visualizations(
            criticality_scores, 
            impact_data,
            filtered_layers, 
            medium_font, 
            large_font,
            filtered_colors
        )
        
        # Detect anomalies
        anomalies, connection_z_scores = self.detect_anomalies(
            filtered_connections, 
            filtered_layers,
            threshold=self.anomaly_threshold_slider.value() / 10.0  # Convert slider value to threshold
        )
        
        # Create anomaly visualizations
        self.create_anomaly_visualizations(
            anomalies, 
            connection_z_scores,
            filtered_connections,
            filtered_layers, 
            medium_font, 
            large_font,
            filtered_colors,
            threshold=self.anomaly_threshold_slider.value() / 10.0
        )
        
        self.figure.tight_layout(pad=1.0)
        self.canvas.draw()
    
    def identify_critical_layers(self, layer_connections, layers, metric="Connectivity Impact"):
        """
        Identify critical layers in the network
        
        Parameters:
        -----------
        layer_connections : numpy.ndarray
            Matrix of connection counts between layers
        layers : list
            List of layer names
        metric : str
            The criticality metric to use
            
        Returns:
        --------
        tuple
            (criticality_scores, impact_data)
            criticality_scores: dict mapping layer indices to criticality scores
            impact_data: dict containing impact simulation results
        """
        n_layers = layer_connections.shape[0]
        criticality_scores = {}
        impact_data = {"before": None, "after": {}, "metric": metric}
        
        # Create a graph from layer connections
        G = nx.Graph()
        for i in range(n_layers):
            G.add_node(i, name=layers[i])
        
        for i in range(n_layers):
            for j in range(i+1, n_layers):
                if layer_connections[i, j] > 0:
                    G.add_edge(i, j, weight=layer_connections[i, j])
        
        if metric == "Connectivity Impact":
            # Calculate baseline connectivity
            try:
                # Use average shortest path length as connectivity metric
                if nx.is_connected(G):
                    baseline_connectivity = nx.average_shortest_path_length(G, weight='weight')
                else:
                    # For disconnected graphs, use the average of connected components
                    components = list(nx.connected_components(G))
                    baseline_connectivity = 0
                    for component in components:
                        subgraph = G.subgraph(component)
                        if len(subgraph) > 1:  # Only consider components with at least 2 nodes
                            baseline_connectivity += nx.average_shortest_path_length(subgraph, weight='weight')
                    baseline_connectivity /= max(1, len(components))
                
                impact_data["before"] = baseline_connectivity
                
                # Calculate impact of removing each layer
                for i in range(n_layers):
                    # Create a copy of the graph without layer i
                    G_without_i = G.copy()
                    G_without_i.remove_node(i)
                    
                    # Calculate connectivity without layer i
                    if nx.is_connected(G_without_i):
                        connectivity_without_i = nx.average_shortest_path_length(G_without_i, weight='weight')
                    else:
                        # For disconnected graphs, use the average of connected components
                        components = list(nx.connected_components(G_without_i))
                        connectivity_without_i = 0
                        for component in components:
                            subgraph = G_without_i.subgraph(component)
                            if len(subgraph) > 1:  # Only consider components with at least 2 nodes
                                connectivity_without_i += nx.average_shortest_path_length(subgraph, weight='weight')
                        connectivity_without_i /= max(1, len(components))
                    
                    # Calculate impact (increase in path length = decrease in connectivity)
                    if baseline_connectivity > 0:
                        impact = (connectivity_without_i - baseline_connectivity) / baseline_connectivity
                    else:
                        impact = 0
                    
                    criticality_scores[i] = max(0, impact)  # Ensure non-negative
                    impact_data["after"][i] = connectivity_without_i
            
            except Exception as e:
                print(f"Warning: Error calculating connectivity impact: {e}")
                # Fallback to simple degree centrality
                for i in range(n_layers):
                    criticality_scores[i] = G.degree(i, weight='weight') / max(1, G.number_of_edges())
        
        elif metric == "Centrality Impact":
            # Calculate baseline centrality
            try:
                baseline_centrality = nx.eigenvector_centrality(G, weight='weight')
                impact_data["before"] = sum(baseline_centrality.values()) / len(baseline_centrality)
                
                # Calculate impact of removing each layer
                for i in range(n_layers):
                    # Create a copy of the graph without layer i
                    G_without_i = G.copy()
                    G_without_i.remove_node(i)
                    
                    # Calculate centrality without layer i
                    if G_without_i.number_of_edges() > 0:
                        centrality_without_i = nx.eigenvector_centrality(G_without_i, weight='weight')
                        avg_centrality_without_i = sum(centrality_without_i.values()) / len(centrality_without_i)
                    else:
                        avg_centrality_without_i = 0
                    
                    # Calculate impact (decrease in average centrality)
                    if impact_data["before"] > 0:
                        impact = (impact_data["before"] - avg_centrality_without_i) / impact_data["before"]
                    else:
                        impact = 0
                    
                    criticality_scores[i] = max(0, impact)  # Ensure non-negative
                    impact_data["after"][i] = avg_centrality_without_i
            
            except Exception as e:
                print(f"Warning: Error calculating centrality impact: {e}")
                # Fallback to betweenness centrality
                try:
                    betweenness = nx.betweenness_centrality(G, weight='weight')
                    for i in range(n_layers):
                        criticality_scores[i] = betweenness.get(i, 0)
                except:
                    # Last resort: use degree
                    for i in range(n_layers):
                        criticality_scores[i] = G.degree(i, weight='weight') / max(1, G.number_of_edges())
        
        else:  # Information Flow Impact
            # Calculate baseline information flow
            try:
                # Use random walk as a proxy for information flow
                baseline_flow = {}
                for source in range(n_layers):
                    # Initialize with all flow at source
                    flow = {i: 0.0 for i in range(n_layers)}
                    flow[source] = 1.0
                    
                    # Simulate random walk for a few steps
                    for _ in range(3):  # 3 steps of diffusion
                        new_flow = {i: 0.0 for i in range(n_layers)}
                        for i in range(n_layers):
                            if flow[i] > 0:
                                # Distribute flow to neighbors
                                neighbors = list(G.neighbors(i))
                                if neighbors:
                                    flow_per_neighbor = flow[i] / len(neighbors)
                                    for neighbor in neighbors:
                                        new_flow[neighbor] += flow_per_neighbor
                        flow = new_flow
                    
                    baseline_flow[source] = sum(flow.values())
                
                avg_baseline_flow = sum(baseline_flow.values()) / len(baseline_flow)
                impact_data["before"] = avg_baseline_flow
                
                # Calculate impact of removing each layer
                for i in range(n_layers):
                    # Create a copy of the graph without layer i
                    G_without_i = G.copy()
                    G_without_i.remove_node(i)
                    
                    # Calculate flow without layer i
                    flow_without_i = {}
                    for source in range(n_layers):
                        if source != i:  # Skip the removed layer as source
                            # Initialize with all flow at source
                            flow = {j: 0.0 for j in range(n_layers) if j != i}
                            flow[source] = 1.0
                            
                            # Simulate random walk for a few steps
                            for _ in range(3):  # 3 steps of diffusion
                                new_flow = {j: 0.0 for j in range(n_layers) if j != i}
                                for j in flow:
                                    if flow[j] > 0:
                                        # Distribute flow to neighbors
                                        neighbors = [n for n in G_without_i.neighbors(j)]
                                        if neighbors:
                                            flow_per_neighbor = flow[j] / len(neighbors)
                                            for neighbor in neighbors:
                                                new_flow[neighbor] += flow_per_neighbor
                                flow = new_flow
                            
                            flow_without_i[source] = sum(flow.values())
                    
                    if flow_without_i:
                        avg_flow_without_i = sum(flow_without_i.values()) / len(flow_without_i)
                    else:
                        avg_flow_without_i = 0
                    
                    # Calculate impact (decrease in average flow)
                    if avg_baseline_flow > 0:
                        impact = (avg_baseline_flow - avg_flow_without_i) / avg_baseline_flow
                    else:
                        impact = 0
                    
                    criticality_scores[i] = max(0, impact)  # Ensure non-negative
                    impact_data["after"][i] = avg_flow_without_i
            
            except Exception as e:
                print(f"Warning: Error calculating information flow impact: {e}")
                # Fallback to closeness centrality
                try:
                    closeness = nx.closeness_centrality(G, distance='weight')
                    for i in range(n_layers):
                        criticality_scores[i] = closeness.get(i, 0)
                except:
                    # Last resort: use degree
                    for i in range(n_layers):
                        criticality_scores[i] = G.degree(i, weight='weight') / max(1, G.number_of_edges())
        
        return criticality_scores, impact_data
    
    def detect_anomalies(self, layer_connections, layers, threshold=1.0):
        """
        Detect anomalies in layer connections
        
        Parameters:
        -----------
        layer_connections : numpy.ndarray
            Matrix of connection counts between layers
        layers : list
            List of layer names
        threshold : float
            Number of standard deviations for anomaly detection
            
        Returns:
        --------
        tuple
            (anomalies, anomaly_scores)
            anomalies: list of (i, j) tuples representing anomalous connections
            anomaly_scores: dict mapping (i, j) tuples to anomaly scores
        """
        n_layers = layer_connections.shape[0]
        anomalies = []
        anomaly_scores = {}
        
        # Flatten the connection matrix (excluding diagonal)
        connections = []
        for i in range(n_layers):
            for j in range(n_layers):
                if i != j:
                    connections.append(layer_connections[i, j])
        
        if not connections:
            return [], {}
        
        # Calculate statistics
        mean = np.mean(connections)
        std = np.std(connections)
        
        if std == 0:
            # No variation in connections, can't detect anomalies
            return [], {}
        
        # Calculate z-scores for each connection
        for i in range(n_layers):
            for j in range(n_layers):
                if i != j:
                    z_score = (layer_connections[i, j] - mean) / std
                    anomaly_scores[(i, j)] = z_score
                    
                    # Identify anomalies (connections with high z-scores)
                    if abs(z_score) > threshold:
                        anomalies.append((i, j))
        
        return anomalies, anomaly_scores
    
    def create_criticality_visualizations(self, criticality_scores, impact_data, layers, medium_font, large_font, layer_colors):
        """Create visualizations for critical layer identification"""
        # Sort layers by criticality score
        sorted_items = sorted(criticality_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_indices = [item[0] for item in sorted_items]
        sorted_scores = [item[1] for item in sorted_items]
        sorted_layers = [layers[i] for i in sorted_indices]
        
        # Create bar chart of criticality scores
        bar_colors = []
        if layer_colors:
            for layer in sorted_layers:
                if layer in layer_colors:
                    bar_colors.append(layer_colors[layer])
                else:
                    bar_colors.append('skyblue')
        else:
            bar_colors = ['skyblue' for _ in range(len(sorted_layers))]
        
        bars = self.criticality_bar_ax.barh(sorted_layers, sorted_scores, color=bar_colors)
        metric = self.criticality_metric_dropdown.currentText()
        self.criticality_bar_ax.set_title(f'Layer Criticality ({metric})', **large_font)
        self.criticality_bar_ax.set_xlabel('Criticality Score', **medium_font)
        self.criticality_bar_ax.set_ylabel('Layer', **medium_font)
        
        # Add value labels to the bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x_pos = width * 1.01
            self.criticality_bar_ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                                       va='center', **medium_font)
        
        # Create impact visualization
        if impact_data["before"] is not None and impact_data["after"]:
            # Create a bar chart comparing before and after values
            layers_to_show = min(5, len(sorted_indices))  # Show top 5 critical layers
            
            # Get the top critical layers
            top_indices = sorted_indices[:layers_to_show]
            top_layers = [layers[i] for i in top_indices]
            
            # Prepare data for visualization
            before_values = [impact_data["before"] for _ in range(layers_to_show)]
            after_values = [impact_data["after"].get(i, 0) for i in top_indices]
            
            # Set up x positions for bars
            x = np.arange(layers_to_show)
            width = 0.35
            
            # Create grouped bar chart
            self.impact_ax.bar(x - width/2, before_values, width, label='Before Removal', color='skyblue')
            self.impact_ax.bar(x + width/2, after_values, width, label='After Removal', color='salmon')
            
            # Add labels and title
            self.impact_ax.set_ylabel('Impact Metric Value', **medium_font)
            self.impact_ax.set_title(f'Impact of Layer Removal', **large_font)
            self.impact_ax.set_xticks(x)
            self.impact_ax.set_xticklabels(top_layers, rotation=45, ha='right', **medium_font)
            self.impact_ax.legend(fontsize=8)
            
            # Add a note explaining the metric
            if impact_data["metric"] == "Connectivity Impact":
                note = "Metric: Average Shortest Path Length\nHigher values after removal indicate critical layers"
            elif impact_data["metric"] == "Centrality Impact":
                note = "Metric: Average Eigenvector Centrality\nLower values after removal indicate critical layers"
            else:  # Information Flow Impact
                note = "Metric: Average Information Flow\nLower values after removal indicate critical layers"
                
            self.impact_ax.text(0.5, -0.3, note, transform=self.impact_ax.transAxes,
                              ha='center', va='center', fontsize=8, 
                              bbox=dict(facecolor='white', alpha=0.7))
        else:
            self.impact_ax.text(0.5, 0.5, "Impact simulation failed", 
                              ha='center', va='center', fontsize=12)
            self.impact_ax.axis('off')
    
    def create_anomaly_visualizations(self, anomalies, anomaly_scores, layer_connections, layers, medium_font, large_font, layer_colors, threshold=1.0):
        """Create visualizations for anomaly detection"""
        n_layers = len(layers)
        
        if anomaly_scores:
            # Create heatmap of anomaly scores
            anomaly_matrix = np.zeros((n_layers, n_layers))
            for i in range(n_layers):
                for j in range(n_layers):
                    if (i, j) in anomaly_scores:
                        anomaly_matrix[i, j] = anomaly_scores[(i, j)]
            
            # Use diverging colormap for z-scores
            cmap = plt.cm.RdBu_r
            vmax = max(3, max(abs(v) for v in anomaly_scores.values()))
            vmin = -vmax
            
            im = self.anomaly_ax.imshow(anomaly_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Add colorbar
            cbar = self.anomaly_ax.figure.colorbar(im, ax=self.anomaly_ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label('Z-Score (Standard Deviations)', fontsize=8)
            
            # Add labels
            self.anomaly_ax.set_xticks(range(n_layers))
            self.anomaly_ax.set_yticks(range(n_layers))
            self.anomaly_ax.set_xticklabels(layers, rotation=90, fontsize=8)
            self.anomaly_ax.set_yticklabels(layers, fontsize=8)
            
            self.anomaly_ax.set_title(f'Connection Anomalies (Threshold: {threshold}σ)', **large_font)
            
            # Highlight anomalies
            for i, j in anomalies:
                self.anomaly_ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                                     edgecolor='black', lw=2))
            
            # Create network visualization with anomalies highlighted
            G = nx.Graph()
            
            # Add nodes
            for i, layer in enumerate(layers):
                G.add_node(i, name=layer)
            
            # Add edges
            for i in range(n_layers):
                for j in range(i+1, n_layers):
                    if layer_connections[i, j] > 0:
                        G.add_edge(i, j, weight=layer_connections[i, j])
            
            # Prepare node colors
            node_colors = []
            if layer_colors:
                for layer in layers:
                    if layer in layer_colors:
                        node_colors.append(layer_colors[layer])
                    else:
                        node_colors.append('skyblue')
            else:
                node_colors = ['skyblue' for _ in range(n_layers)]
            
            # Use spring layout
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, 
                                  alpha=0.8, ax=self.network_ax)
            
            # Draw normal edges
            normal_edges = [(i, j) for i, j in G.edges() if (i, j) not in anomalies and (j, i) not in anomalies]
            if normal_edges:
                nx.draw_networkx_edges(G, pos, edgelist=normal_edges, width=1, alpha=0.5, 
                                      ax=self.network_ax)
            
            # Draw anomalous edges
            anomalous_edges = [(i, j) for i, j in G.edges() if (i, j) in anomalies or (j, i) in anomalies]
            if anomalous_edges:
                nx.draw_networkx_edges(G, pos, edgelist=anomalous_edges, width=3, 
                                      edge_color='red', ax=self.network_ax)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, labels={i: layers[i] for i in range(n_layers)}, 
                                   font_size=8, ax=self.network_ax)
            
            self.network_ax.set_title('Network with Anomalous Connections', **large_font)
            self.network_ax.axis('off')
            
            # Add a legend explaining the visualization
            legend_text = (f"Red edges: Anomalous connections\n"
                          f"Anomalies: {len(anomalies)} connections\n"
                          f"Threshold: {threshold} standard deviations")
            self.network_ax.text(0.5, -0.1, legend_text, transform=self.network_ax.transAxes, 
                               ha='center', va='center', fontsize=8, 
                               bbox=dict(facecolor='white', alpha=0.7))
        else:
            self.anomaly_ax.text(0.5, 0.5, "No anomalies detected", 
                               ha='center', va='center', fontsize=12)
            self.anomaly_ax.axis('off')
            
            self.network_ax.text(0.5, 0.5, "No anomalies to visualize", 
                               ha='center', va='center', fontsize=12)
            self.network_ax.axis('off')
    
    def update_stats(self, data_manager):
        """Update the Critical Structure Analysis with current data"""
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
            self.clear_visualization("Critical structure analysis disabled") 