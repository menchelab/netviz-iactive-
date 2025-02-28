from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QComboBox,
    QLabel,
    QSlider,
    QPushButton,
    QGroupBox,
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

from .base_panel import BaseStatsPanel


class HyperbolicEmbeddingPanel(BaseStatsPanel):
    """Panel for Hyperbolic Embedding visualization"""

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

        # Add embedding method dropdown
        controls_layout.addWidget(QLabel("Embedding Method:"))
        self.embedding_method_dropdown = QComboBox()
        self.embedding_method_dropdown.addItems(
            ["Hierarchical", "Centrality-based", "Community-based"]
        )
        self.embedding_method_dropdown.currentTextChanged.connect(
            self.on_method_changed
        )
        controls_layout.addWidget(self.embedding_method_dropdown)

        # Add curvature slider
        controls_layout.addWidget(QLabel("Curvature:"))
        self.curvature_slider = QSlider(Qt.Horizontal)
        self.curvature_slider.setMinimum(1)
        self.curvature_slider.setMaximum(10)
        self.curvature_slider.setValue(5)  # Default value
        self.curvature_slider.setTickPosition(QSlider.TicksBelow)
        self.curvature_slider.setTickInterval(1)
        self.curvature_slider.valueChanged.connect(self.on_curvature_changed)
        controls_layout.addWidget(self.curvature_slider)

        # Add node size slider
        controls_layout.addWidget(QLabel("Node Size:"))
        self.node_size_slider = QSlider(Qt.Horizontal)
        self.node_size_slider.setMinimum(1)
        self.node_size_slider.setMaximum(10)
        self.node_size_slider.setValue(5)  # Default value
        self.node_size_slider.setTickPosition(QSlider.TicksBelow)
        self.node_size_slider.setTickInterval(1)
        self.node_size_slider.valueChanged.connect(self.on_node_size_changed)
        controls_layout.addWidget(self.node_size_slider)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Add navigation controls
        nav_layout = QHBoxLayout()

        # Add zoom controls
        zoom_group = QGroupBox("Zoom")
        zoom_layout = QHBoxLayout()

        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.clicked.connect(self.on_zoom_in)
        zoom_layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("-")
        self.zoom_out_button.clicked.connect(self.on_zoom_out)
        zoom_layout.addWidget(self.zoom_out_button)

        zoom_group.setLayout(zoom_layout)
        nav_layout.addWidget(zoom_group)

        # Add rotation controls
        rotation_group = QGroupBox("Rotation")
        rotation_layout = QHBoxLayout()

        self.rotate_left_button = QPushButton("←")
        self.rotate_left_button.clicked.connect(self.on_rotate_left)
        rotation_layout.addWidget(self.rotate_left_button)

        self.rotate_right_button = QPushButton("→")
        self.rotate_right_button.clicked.connect(self.on_rotate_right)
        rotation_layout.addWidget(self.rotate_right_button)

        rotation_group.setLayout(rotation_layout)
        nav_layout.addWidget(rotation_group)

        # Add coloring controls
        coloring_group = QGroupBox("Coloring")
        coloring_layout = QHBoxLayout()

        self.coloring_dropdown = QComboBox()
        self.coloring_dropdown.addItems(
            ["Layer Colors", "Depth", "Centrality", "Communities"]
        )
        self.coloring_dropdown.currentTextChanged.connect(self.on_coloring_changed)
        coloring_layout.addWidget(self.coloring_dropdown)

        coloring_group.setLayout(coloring_layout)
        nav_layout.addWidget(coloring_group)

        # Add reset button
        self.reset_button = QPushButton("Reset View")
        self.reset_button.clicked.connect(self.on_reset_view)
        nav_layout.addWidget(self.reset_button)

        nav_layout.addStretch()
        layout.addLayout(nav_layout)

        # Create figure for Hyperbolic Embedding
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Initialize main plot
        self.ax = self.figure.add_subplot(111)

        # Store current data and view parameters
        self._current_data = None
        self._zoom_level = 1.0
        self._rotation_angle = 0.0

    def on_state_changed(self, state):
        """Handle enable/disable state change"""
        if self._current_data and state:
            self.update_visualization()
        elif not state:
            # Clear the visualization when disabled
            self.clear_visualization("Hyperbolic embedding disabled")

    def on_method_changed(self, method):
        """Handle embedding method change"""
        if self._current_data and self.enable_checkbox.isChecked():
            # Reset view parameters when changing method
            self._zoom_level = 1.0
            self._rotation_angle = 0.0
            self.update_visualization()

    def on_curvature_changed(self, value):
        """Handle curvature slider change"""
        if self._current_data and self.enable_checkbox.isChecked():
            self.update_visualization()

    def on_node_size_changed(self, value):
        """Handle node size slider change"""
        if self._current_data and self.enable_checkbox.isChecked():
            self.update_visualization()

    def on_coloring_changed(self, coloring):
        """Handle coloring method change"""
        if self._current_data and self.enable_checkbox.isChecked():
            self.update_visualization()

    def on_zoom_in(self):
        """Handle zoom in button click"""
        if self._current_data and self.enable_checkbox.isChecked():
            self._zoom_level *= 1.2
            self.update_visualization()

    def on_zoom_out(self):
        """Handle zoom out button click"""
        if self._current_data and self.enable_checkbox.isChecked():
            self._zoom_level /= 1.2
            self.update_visualization()

    def on_rotate_left(self):
        """Handle rotate left button click"""
        if self._current_data and self.enable_checkbox.isChecked():
            self._rotation_angle -= 15  # Rotate 15 degrees left
            self.update_visualization()

    def on_rotate_right(self):
        """Handle rotate right button click"""
        if self._current_data and self.enable_checkbox.isChecked():
            self._rotation_angle += 15  # Rotate 15 degrees right
            self.update_visualization()

    def on_reset_view(self):
        """Handle reset view button click"""
        if self._current_data and self.enable_checkbox.isChecked():
            self._zoom_level = 1.0
            self._rotation_angle = 0.0
            self.update_visualization()

    def clear_visualization(self, message=""):
        """Clear the plot and display a message"""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        if message:
            self.ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
            self.ax.axis("off")

        self.canvas.draw()

    def update_visualization(self):
        """Update the hyperbolic embedding visualization"""
        if not self._current_data:
            self.clear_visualization("No data available")
            return

        try:
            self.clear_visualization()

            # Unpack stored data
            data_manager, medium_font, large_font = self._current_data

            # Get filtered data from the data manager
            layers = data_manager.layers
            visible_layer_indices = data_manager.visible_layers
            layer_colors = data_manager.layer_colors

            # Check if we have valid data
            if (
                not layers
                or not visible_layer_indices
                or len(visible_layer_indices) < 2
            ):
                self.clear_visualization(
                    "Not enough visible layers to analyze (minimum 2 required)"
                )
                return

            # Get the filtered layer connections that respect all visibility settings
            # (layers, clusters, and origins)
            try:
                filtered_connections = data_manager.get_layer_connections()
                if filtered_connections is None or filtered_connections.size == 0:
                    self.clear_visualization("No connection data available")
                    return
            except Exception as e:
                print(f"Error getting layer connections: {e}")
                self.clear_visualization("Error retrieving connection data")
                return

            # Filter layers based on visibility
            filtered_layers = [
                layers[i] for i in visible_layer_indices if i < len(layers)
            ]
            filtered_colors = {
                layer: layer_colors.get(layer, "skyblue") for layer in filtered_layers
            }

            if len(filtered_layers) < 2:
                self.clear_visualization(
                    "Not enough visible layers to analyze (minimum 2 required)"
                )
                return

            # Create hyperbolic embedding
            try:
                embedding, hierarchy_data = self.create_hyperbolic_embedding(
                    filtered_connections,
                    filtered_layers,
                    method=self.embedding_method_dropdown.currentText(),
                    curvature=self.curvature_slider.value() / 5.0,  # Scale to 0.2-2.0
                )

                if not embedding:
                    self.clear_visualization(
                        "Could not create embedding with current data"
                    )
                    return
            except Exception as e:
                print(f"Error creating hyperbolic embedding: {e}")
                self.clear_visualization(f"Error creating embedding: {str(e)}")
                return

            # Visualize the embedding
            try:
                self.visualize_hyperbolic_embedding(
                    embedding,
                    hierarchy_data,
                    filtered_layers,
                    medium_font,
                    large_font,
                    filtered_colors,
                    coloring=self.coloring_dropdown.currentText(),
                    node_size_factor=self.node_size_slider.value()
                    / 5.0
                    * 300,  # Scale to 60-600
                )
            except Exception as e:
                print(f"Error visualizing hyperbolic embedding: {e}")
                self.clear_visualization(f"Error visualizing embedding: {str(e)}")
                return

            self.canvas.draw()

        except Exception as e:
            print(f"Error updating Hyperbolic Embedding visualization: {e}")
            self.clear_visualization(f"Error updating visualization: {str(e)}")
            return

    def create_hyperbolic_embedding(
        self, layer_connections, layers, method="Hierarchical", curvature=1.0
    ):
        """
        Create a hyperbolic embedding of the layer network

        Parameters:
        -----------
        layer_connections : numpy.ndarray
            Matrix of connection counts between layers
        layers : list
            List of layer names
        method : str
            The embedding method to use
        curvature : float
            The curvature of the hyperbolic space

        Returns:
        --------
        tuple
            (embedding, hierarchy_data)
            embedding: dict mapping layer indices to (x, y) coordinates
            hierarchy_data: dict containing hierarchical information
        """
        # Check if we have valid data
        if layer_connections is None or layers is None or len(layers) < 2:
            print("Not enough data for hyperbolic embedding")
            return {}, {"depth": {}, "parent": {}, "children": {}, "communities": {}}

        try:
            n_layers = layer_connections.shape[0]

            # Ensure dimensions match
            if n_layers != len(layers):
                print(
                    f"Warning: Layer connection matrix dimensions ({n_layers}) don't match layer count ({len(layers)})"
                )
                n_layers = min(n_layers, len(layers))

            embedding = {}
            hierarchy_data = {
                "depth": {},
                "parent": {},
                "children": {},
                "communities": {},
            }

            # Create a graph from layer connections
            G = nx.Graph()
            for i in range(n_layers):
                G.add_node(i, name=layers[i])

            for i in range(n_layers):
                for j in range(i + 1, n_layers):
                    if layer_connections[i, j] > 0:
                        G.add_edge(i, j, weight=layer_connections[i, j])

            # Check if graph has any edges
            if G.number_of_edges() == 0:
                print("No connections between layers for hyperbolic embedding")
                # Create a simple circular layout as fallback
                for i in range(n_layers):
                    angle = 2 * np.pi * i / n_layers
                    r = 0.8  # Fixed radius
                    embedding[i] = (r * np.cos(angle), r * np.sin(angle))
                    hierarchy_data["depth"][i] = 1
                return embedding, hierarchy_data

            if method == "Hierarchical":
                # Create hierarchical embedding based on clustering
                try:
                    # Convert connection matrix to distance matrix
                    # Higher connection = lower distance
                    distance_matrix = np.zeros_like(layer_connections, dtype=float)
                    max_connection = layer_connections.max()
                    if max_connection > 0:
                        for i in range(n_layers):
                            for j in range(n_layers):
                                if i == j:
                                    distance_matrix[i, j] = 0
                                else:
                                    # Invert and normalize connections to get distances
                                    # Use max of connections in both directions for symmetry
                                    connection_value = max(
                                        layer_connections[i, j], layer_connections[j, i]
                                    )
                                    distance_matrix[i, j] = 1 - (
                                        connection_value / max_connection
                                    )

                    # Ensure the distance matrix is symmetric
                    distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)

                    # Perform hierarchical clustering
                    condensed_dist = squareform(distance_matrix)
                    linkage_matrix = hierarchy.linkage(condensed_dist, method="average")

                    # Create a tree from the linkage matrix
                    tree = hierarchy.to_tree(linkage_matrix, rd=True)

                    # Assign coordinates based on the tree structure
                    # Root at the center, children arranged in a circle
                    def assign_coordinates(node, radius, angle_range, depth=0):
                        if node.is_leaf():
                            # Leaf node (original layer)
                            idx = node.id
                            angle = np.mean(angle_range)

                            # Convert to Poincaré disk coordinates
                            r = np.tanh(radius * curvature / 2)  # Scale by curvature
                            x = r * np.cos(angle)
                            y = r * np.sin(angle)

                            embedding[idx] = (x, y)
                            hierarchy_data["depth"][idx] = depth
                            return [idx]
                        else:
                            # Internal node
                            mid_angle = np.mean(angle_range)
                            left_range = (angle_range[0], mid_angle)
                            right_range = (mid_angle, angle_range[1])

                            # Process left and right subtrees
                            left_leaves = assign_coordinates(
                                node.left, radius + 1, left_range, depth + 1
                            )
                            right_leaves = assign_coordinates(
                                node.right, radius + 1, right_range, depth + 1
                            )

                            # Store parent-child relationships
                            for leaf in left_leaves + right_leaves:
                                hierarchy_data["parent"][leaf] = node.id + n_layers

                            hierarchy_data["children"][node.id + n_layers] = (
                                left_leaves + right_leaves
                            )

                            return left_leaves + right_leaves

                    # Start from the root with full angle range
                    assign_coordinates(tree, 1.0, (0, 2 * np.pi))

                except Exception as e:
                    print(f"Error in hierarchical embedding: {e}")
                    # Fallback to circular layout
                    for i in range(n_layers):
                        angle = 2 * np.pi * i / n_layers
                        r = 0.8  # Fixed radius
                        embedding[i] = (r * np.cos(angle), r * np.sin(angle))
                        hierarchy_data["depth"][i] = 1

            elif method == "Centrality-based":
                try:
                    # Calculate centrality
                    centrality = nx.eigenvector_centrality(G, weight="weight")

                    # Normalize centrality values
                    max_centrality = max(centrality.values()) if centrality else 1
                    normalized_centrality = {
                        i: centrality.get(i, 0) / max_centrality
                        for i in range(n_layers)
                    }

                    # Assign coordinates based on centrality
                    # More central nodes closer to the center
                    for i in range(n_layers):
                        # Calculate radius based on centrality (inverse relationship)
                        r = 1 - normalized_centrality[i]
                        r = np.tanh(r * curvature)  # Apply hyperbolic scaling

                        # Assign angle based on node index
                        angle = 2 * np.pi * i / n_layers

                        # Convert to Cartesian coordinates
                        x = r * np.cos(angle)
                        y = r * np.sin(angle)

                        embedding[i] = (x, y)
                        hierarchy_data["depth"][i] = (
                            int(r * 5) + 1
                        )  # Approximate depth from radius

                except Exception as e:
                    print(f"Error in centrality-based embedding: {e}")
                    # Fallback to circular layout
                    for i in range(n_layers):
                        angle = 2 * np.pi * i / n_layers
                        r = 0.8  # Fixed radius
                        embedding[i] = (r * np.cos(angle), r * np.sin(angle))
                        hierarchy_data["depth"][i] = 1

            else:  # Community-based
                try:
                    # Detect communities
                    communities = nx.community.greedy_modularity_communities(
                        G, weight="weight"
                    )

                    # Assign community IDs
                    community_mapping = {}
                    for i, community in enumerate(communities):
                        for node in community:
                            community_mapping[node] = i
                            hierarchy_data["communities"][node] = i

                    # Arrange communities in a circle
                    n_communities = len(communities)
                    community_angles = {
                        i: 2 * np.pi * i / n_communities for i in range(n_communities)
                    }

                    # Position nodes within their communities
                    for i in range(n_layers):
                        comm_id = community_mapping.get(i, 0)
                        comm_size = len(communities[comm_id])

                        # Base angle for this community
                        base_angle = community_angles[comm_id]

                        # Position within community
                        node_idx = (
                            list(communities[comm_id]).index(i)
                            if i in communities[comm_id]
                            else 0
                        )
                        angle_offset = (
                            0.2 * np.pi * (node_idx / max(1, comm_size - 1) - 0.5)
                        )
                        angle = base_angle + angle_offset

                        # Radius based on community size (larger communities further out)
                        base_r = 0.5 + 0.3 * (comm_size / n_layers)
                        r = np.tanh(base_r * curvature)

                        # Convert to Cartesian coordinates
                        x = r * np.cos(angle)
                        y = r * np.sin(angle)

                        embedding[i] = (x, y)
                        hierarchy_data["depth"][i] = (
                            comm_id + 1
                        )  # Use community ID as depth

                except Exception as e:
                    print(f"Error in community-based embedding: {e}")
                    # Fallback to circular layout
                    for i in range(n_layers):
                        angle = 2 * np.pi * i / n_layers
                        r = 0.8  # Fixed radius
                        embedding[i] = (r * np.cos(angle), r * np.sin(angle))
                        hierarchy_data["depth"][i] = 1

            return embedding, hierarchy_data

        except Exception as e:
            print(f"Error creating hyperbolic embedding: {e}")
            # Return empty embedding as fallback
            return {}, {"depth": {}, "parent": {}, "children": {}, "communities": {}}

    def visualize_hyperbolic_embedding(
        self,
        embedding,
        hierarchy_data,
        layers,
        medium_font,
        large_font,
        layer_colors,
        coloring="Layer Colors",
        node_size_factor=300,
    ):
        """
        Visualize the hyperbolic embedding

        Parameters:
        -----------
        embedding : dict
            Dict mapping layer indices to (x, y) coordinates
        hierarchy_data : dict
            Dict containing hierarchical information
        layers : list
            List of layer names
        medium_font, large_font : dict
            Font configuration dictionaries
        layer_colors : dict
            Dictionary mapping layer names to colors
        coloring : str
            The coloring method to use
        node_size_factor : float
            Base size for nodes
        """
        # Apply zoom and rotation
        transformed_embedding = {}
        for i, (x, y) in embedding.items():
            # Apply zoom (in hyperbolic space)
            r = np.sqrt(x**2 + y**2)
            if r > 0:
                angle = np.arctan2(y, x)

                # Apply rotation
                angle += np.radians(self._rotation_angle)

                # Apply zoom (scale the hyperbolic radius)
                r_h = np.arctanh(r)  # Convert to hyperbolic radius
                r_h /= self._zoom_level  # Apply zoom
                r = np.tanh(r_h)  # Convert back to Poincaré disk radius

                transformed_embedding[i] = (r * np.cos(angle), r * np.sin(angle))
            else:
                transformed_embedding[i] = (0, 0)

        # Draw Poincaré disk boundary
        circle = plt.Circle(
            (0, 0), 1, fill=False, edgecolor="black", linestyle="--", alpha=0.5
        )
        self.ax.add_patch(circle)

        # Set axis limits
        self.ax.set_xlim(-1.05, 1.05)
        self.ax.set_ylim(-1.05, 1.05)

        # Draw nodes
        x_coords = [transformed_embedding[i][0] for i in range(len(layers))]
        y_coords = [transformed_embedding[i][1] for i in range(len(layers))]

        # Determine node colors based on coloring method
        node_colors = []
        if coloring == "Layer Colors":
            # Use layer colors
            for layer in layers:
                if layer in layer_colors:
                    node_colors.append(layer_colors[layer])
                else:
                    node_colors.append("skyblue")

        elif coloring == "Depth":
            # Color by hierarchical depth
            depths = [hierarchy_data["depth"].get(i, 0) for i in range(len(layers))]
            max_depth = max(depths) if depths else 1
            node_colors = [plt.cm.viridis(d / max_depth) for d in depths]

        elif coloring == "Centrality":
            # Create a graph from embedding distances
            G = nx.Graph()
            for i in range(len(layers)):
                G.add_node(i)

            for i in range(len(layers)):
                for j in range(i + 1, len(layers)):
                    # Calculate hyperbolic distance
                    x1, y1 = embedding[i]
                    x2, y2 = embedding[j]

                    # Möbius distance formula
                    d = np.acosh(
                        1
                        + 2
                        * ((x1 - x2) ** 2 + (y1 - y2) ** 2)
                        / ((1 - x1**2 - y1**2) * (1 - x2**2 - y2**2))
                    )

                    if not np.isnan(d) and d < 5:  # Only connect nearby nodes
                        G.add_edge(i, j, weight=1 / max(0.1, d))

            # Calculate centrality
            try:
                centrality = nx.eigenvector_centrality(G, weight="weight")
                max_centrality = max(centrality.values()) if centrality else 1
                node_colors = [
                    plt.cm.plasma(centrality.get(i, 0) / max_centrality)
                    for i in range(len(layers))
                ]
            except:
                # Fallback
                node_colors = ["skyblue" for _ in range(len(layers))]

        else:  # Communities
            # Color by community
            communities = hierarchy_data.get("communities", {})
            if communities:
                community_ids = [communities.get(i, 0) for i in range(len(layers))]
                unique_communities = sorted(set(community_ids))
                color_map = {
                    comm: plt.cm.tab10(i % 10)
                    for i, comm in enumerate(unique_communities)
                }
                node_colors = [
                    color_map.get(communities.get(i, 0), "skyblue")
                    for i in range(len(layers))
                ]
            else:
                node_colors = ["skyblue" for _ in range(len(layers))]

        # Determine node sizes (can vary based on centrality, depth, etc.)
        node_sizes = [node_size_factor for _ in range(len(layers))]

        # Draw nodes
        self.ax.scatter(
            x_coords,
            y_coords,
            s=node_sizes,
            c=node_colors,
            alpha=0.8,
            edgecolors="black",
            zorder=10,
        )

        # Draw edges (curved in hyperbolic space)
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                x1, y1 = transformed_embedding[i]
                x2, y2 = transformed_embedding[j]

                # Calculate hyperbolic distance
                d = np.acosh(
                    1
                    + 2
                    * ((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    / ((1 - x1**2 - y1**2) * (1 - x2**2 - y2**2))
                )

                if not np.isnan(d) and d < 2:  # Only draw edges for nearby nodes
                    # Draw a geodesic in the Poincaré disk
                    # For simplicity, we'll approximate with a straight line
                    # In a full implementation, proper hyperbolic geodesics should be used
                    self.ax.plot(
                        [x1, x2], [y1, y2], "gray", alpha=0.3, linewidth=1, zorder=1
                    )

        # Draw node labels
        for i in range(len(layers)):
            x, y = transformed_embedding[i]
            self.ax.text(
                x,
                y,
                layers[i],
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                    boxstyle="round,pad=0.1",
                ),
                zorder=20,
            )

        # Set title and remove axes
        method = self.embedding_method_dropdown.currentText()
        self.ax.set_title(f"Hyperbolic Embedding ({method})", **large_font)
        self.ax.axis("off")

        # Add a legend explaining the visualization
        if coloring == "Depth":
            legend_text = "Color represents hierarchical depth\nDeeper nodes are further from center"
        elif coloring == "Centrality":
            legend_text = "Color represents node centrality\nWarmer colors indicate higher centrality"
        elif coloring == "Communities":
            legend_text = "Color represents community membership\nNodes in same community have same color"
        else:
            legend_text = "Using layer colors from main visualization"

        self.ax.text(
            0.5,
            -0.05,
            legend_text,
            transform=self.ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7),
        )

        # Add navigation instructions
        nav_text = (
            f"Zoom: {self._zoom_level:.1f}x | Rotation: {self._rotation_angle:.0f}°\n"
            f"Use controls above to navigate"
        )
        self.ax.text(
            0.5,
            1.05,
            nav_text,
            transform=self.ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7),
        )

    def update_stats(self, data_manager):
        """Update the Hyperbolic Embedding with current data"""
        try:
            # Check if data_manager is valid
            if data_manager is None:
                self.clear_visualization("No data manager provided")
                return

            # Clear figure
            self.clear_visualization()

            # Define font sizes
            medium_font = {"fontsize": 7}
            large_font = {"fontsize": 9}

            # Store data manager and font settings for later use
            self._current_data = (data_manager, medium_font, large_font)

            # Only create visualization if enabled
            if self.enable_checkbox.isChecked():
                self.update_visualization()
            else:
                self.clear_visualization("Hyperbolic embedding disabled")

        except Exception as e:
            print(f"Error in update_stats: {e}")
            self.clear_visualization(f"Error updating stats: {str(e)}")
            return
