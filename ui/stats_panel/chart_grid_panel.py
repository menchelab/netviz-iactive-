from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QComboBox, 
                             QLabel, QPushButton, QScrollArea, QWidget, QGridLayout,
                             QSpinBox, QCheckBox, QGroupBox)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import importlib
import importlib.util  # Explicitly import importlib.util
import inspect
import logging
import sys

from .base_panel import BaseStatsPanel
from data.data_loader import get_available_diseases, load_disease_data

class ChartGridPanel(BaseStatsPanel):
    """Panel for displaying charts in a grid layout for all datasets"""
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)  # Add some spacing between elements

        # Add controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(20)  # Add spacing between control groups
        
        # Chart selection
        chart_group = QGroupBox("Chart Selection")
        chart_layout = QVBoxLayout()
        
        # Add chart dropdown
        chart_layout.addWidget(QLabel("Select Chart:"))
        self.chart_dropdown = QComboBox()
        self.chart_dropdown.currentTextChanged.connect(self.on_chart_changed)
        chart_layout.addWidget(self.chart_dropdown)
        
        chart_group.setLayout(chart_layout)
        controls_layout.addWidget(chart_group)
        
        # Grid settings
        grid_group = QGroupBox("Grid Settings")
        grid_layout = QVBoxLayout()
        
        # Add columns spinner
        columns_layout = QHBoxLayout()
        columns_layout.addWidget(QLabel("Columns:"))
        self.columns_spinner = QSpinBox()
        self.columns_spinner.setMinimum(1)
        self.columns_spinner.setMaximum(5)
        self.columns_spinner.setValue(2)
        self.columns_spinner.valueChanged.connect(self.on_grid_settings_changed)
        columns_layout.addWidget(self.columns_spinner)
        grid_layout.addLayout(columns_layout)
        
        # Add dataset filter checkbox
        self.show_all_datasets = QCheckBox("Show All Datasets")
        self.show_all_datasets.setChecked(True)
        self.show_all_datasets.stateChanged.connect(self.on_grid_settings_changed)
        grid_layout.addWidget(self.show_all_datasets)
        
        grid_group.setLayout(grid_layout)
        controls_layout.addWidget(grid_group)
        
        # Generate button
        self.generate_button = QPushButton("Generate Charts\nand wait a minute")
        self.generate_button.clicked.connect(self.generate_charts)
        self.generate_button.setEnabled(False)  # Disabled by default
        controls_layout.addWidget(self.generate_button)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Create scroll area for charts with fixed height
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(800)  # Set minimum height
        self.scroll_content = QWidget()
        self.grid_layout = QGridLayout(self.scroll_content)
        self.grid_layout.setSpacing(10)  # Reduce spacing between charts to 10px
        self.grid_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area, 1)  # 1 = stretch factor

        # Store current data
        self._current_data = None
        self._current_chart_module = None
        self._current_chart_function = None
        self._datasets_cache = {}
        
        # Now populate the chart dropdown after all UI elements are created
        self.populate_chart_dropdown()
        
    def populate_chart_dropdown(self):
        """Populate the chart dropdown with available charts"""
        self.chart_dropdown.clear()
        
        # Get all Python files from the charts directory
        # First try the absolute path
        charts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'charts')
        print(f"Charts directory path: {charts_dir}")
        
        if not os.path.exists(charts_dir):
            print(f"Charts directory not found at: {charts_dir}")
            # Try a relative path
            charts_dir = os.path.join(os.getcwd(), 'charts')
            print(f"Trying alternative path: {charts_dir}")
            
            if not os.path.exists(charts_dir):
                print(f"Charts directory not found at alternative path: {charts_dir}")
                self.chart_dropdown.addItem("No charts directory found")
                return
        
        print(f"Charts directory exists at: {charts_dir}")
            
        # Add charts directory to path if not already there
        parent_dir = os.path.dirname(charts_dir)
        if parent_dir not in sys.path:
            print(f"Adding parent directory to sys.path: {parent_dir}")
            sys.path.append(parent_dir)
        
        # Also add the current directory to sys.path
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            print(f"Adding current directory to sys.path: {current_dir}")
            sys.path.append(current_dir)
            
        print(f"Updated sys.path: {sys.path}")
        
        chart_files = []
        for filename in os.listdir(charts_dir):
            if filename.endswith('.py') and not filename.startswith('__'):
                chart_name = filename[:-3]  # Remove .py extension
                chart_files.append(chart_name)
                print(f"Found chart file: {filename}")
        
        # Sort alphabetically
        chart_files.sort()
        
        # Add to dropdown
        for chart in chart_files:
            self.chart_dropdown.addItem(chart)
            
        print(f"Added {len(chart_files)} charts to dropdown")
        
        # If we have charts, try to select the first one
        if chart_files:
            print(f"Setting current chart to: {chart_files[0]}")
            self.chart_dropdown.setCurrentText(chart_files[0])
            # Note: We don't need to call on_chart_changed here as it will be triggered by setCurrentText
    
    def on_chart_changed(self, chart_name):
        """Handle chart selection change"""
        if not chart_name or chart_name == "No charts directory found":
            print(f"Invalid chart name: {chart_name}")
            self._current_chart_module = None
            self._current_chart_function = None
            self.generate_button.setEnabled(False)
            return
            
        try:
            # Import the selected chart module
            module_name = f"charts.{chart_name}"
            print(f"Attempting to import module: {module_name}")
            
            # Try different import approaches
            try:
                # First try direct import
                import importlib  # Import here to avoid UnboundLocalError
                self._current_chart_module = importlib.import_module(module_name)
                print(f"Successfully imported module: {module_name}")
            except ImportError as e:
                print(f"Direct import failed: {e}")
                
                # Try with relative import
                try:
                    # Try importing from the current package
                    import importlib  # Import here to avoid UnboundLocalError
                    self._current_chart_module = importlib.import_module(f".{chart_name}", package="charts")
                    print(f"Successfully imported module with relative import: .{chart_name}")
                except ImportError as e2:
                    print(f"Relative import failed: {e2}")
                    
                    # Try with absolute path
                    try:
                        # Get the absolute path to the chart module
                        charts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'charts')
                        if not os.path.exists(charts_dir):
                            charts_dir = os.path.join(os.getcwd(), 'charts')
                        
                        module_path = os.path.join(charts_dir, f"{chart_name}.py")
                        print(f"Trying to load module from path: {module_path}")
                        
                        # Use importlib.util to load the module from file path
                        import importlib.util  # Import here to avoid UnboundLocalError
                        spec = importlib.util.spec_from_file_location(chart_name, module_path)
                        if spec is None:
                            raise ImportError(f"Could not create spec for {module_path}")
                            
                        self._current_chart_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(self._current_chart_module)
                        print(f"Successfully imported module from path: {module_path}")
                    except Exception as e3:
                        print(f"Path-based import failed: {e3}")
                        raise ImportError(f"All import methods failed for {chart_name}")
            
            # Find the create_*_chart function
            found_function = False
            for name, obj in inspect.getmembers(self._current_chart_module):
                # Check for all possible function name patterns
                if inspect.isfunction(obj) and name.startswith('create_') and (
                    name.endswith('_chart') or 
                    name.endswith('_charts') or 
                    name.endswith('_diagram') or
                    name.endswith('_heatmap') or  # Add heatmap pattern
                    name.endswith('_graph') or    # Add graph pattern
                    name.endswith('_network')     # Add network pattern
                ):
                    print(f"Found chart function: {name}")
                    self._current_chart_function = obj
                    found_function = True
                    # Enable the generate button
                    self.generate_button.setEnabled(True)
                    break
            
            if not found_function:
                print(f"No create_*_chart function found in {module_name}")
                logging.warning(f"No create_*_chart function found in {module_name}")
                self._current_chart_function = None
                self.generate_button.setEnabled(False)
        except Exception as e:
            print(f"Error loading chart module {chart_name}: {e}")
            logging.error(f"Error loading chart module {chart_name}: {e}")
            self._current_chart_module = None
            self._current_chart_function = None
            self.generate_button.setEnabled(False)
    
    def on_grid_settings_changed(self):
        """Handle grid settings changes"""
        # This will be used when regenerating the charts
        pass
    
    def generate_charts(self):
        """Generate charts for all datasets in a grid layout"""
        if not self._current_data or not self._current_chart_function:
            print("Cannot generate charts: missing data or chart function")
            return
            
        # Clear existing charts
        self.clear_grid_layout()
        
        # Get data manager and font settings
        data_manager, medium_font, large_font = self._current_data
        
        # Get available datasets
        data_dir = data_manager._data_dir if hasattr(data_manager, '_data_dir') else None
        if not data_dir:
            # Try to get data_dir from parent
            parent = self.parent()
            while parent:
                if hasattr(parent, 'data_dir'):
                    data_dir = parent.data_dir
                    break
                parent = parent.parent()
        
        if not data_dir:
            print("Could not find data directory")
            logging.error("Could not find data directory")
            return
            
        print(f"Using data directory: {data_dir}")
        datasets = get_available_diseases(data_dir)
        if not datasets:
            print("No datasets found")
            logging.warning("No datasets found")
            return
            
        print(f"Available datasets: {datasets}")
        
        # If not showing all datasets, only use the current one
        if not self.show_all_datasets.isChecked():
            current_dataset = None
            parent = self.parent()
            while parent:
                if hasattr(parent, 'disease_combo'):
                    current_dataset = parent.disease_combo.currentText()
                    break
                parent = parent.parent()
            
            if current_dataset and current_dataset in datasets:
                datasets = [current_dataset]
                print(f"Using only current dataset: {current_dataset}")
            else:
                print(f"Current dataset not found, using all datasets")
        
        # Get number of columns
        num_columns = self.columns_spinner.value()
        print(f"Using {num_columns} columns for grid layout")
        
        # Create a figure for each dataset
        for i, dataset in enumerate(datasets):
            row = i // num_columns
            col = i % num_columns
            
            print(f"Creating chart for dataset {dataset} at position ({row}, {col})")
            
            # Create figure with larger size
            fig = Figure(figsize=(8, 8), dpi=100)  # Make figures square and larger
            canvas = FigureCanvas(fig)
            canvas.setMinimumHeight(600)  # Set minimum height for each chart
            
            # Add to grid
            self.grid_layout.addWidget(canvas, row, col)
            
            # Load dataset if not already cached
            if dataset not in self._datasets_cache:
                try:
                    print(f"Loading dataset: {dataset}")
                    dataset_data = load_disease_data(data_dir, dataset)
                    
                    # Check if we got all the expected data
                    if dataset_data and len(dataset_data) >= 7:
                        self._datasets_cache[dataset] = dataset_data
                    else:
                        print(f"Incomplete data for dataset {dataset}")
                        ax = fig.add_subplot(111)
                        ax.text(0.5, 0.5, f"Incomplete data for {dataset}", ha='center', va='center')
                        ax.axis('off')
                        canvas.draw()
                        continue
                except Exception as e:
                    print(f"Error loading dataset {dataset}: {e}")
                    logging.error(f"Error loading dataset {dataset}: {e}")
                    ax = fig.add_subplot(111)
                    ax.text(0.5, 0.5, f"Error loading {dataset}: {str(e)}", ha='center', va='center')
                    ax.axis('off')
                    canvas.draw()
                    continue
            
            dataset_data = self._datasets_cache[dataset]
            if not dataset_data:
                print(f"No data for dataset {dataset}")
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f"No data for {dataset}", ha='center', va='center')
                ax.axis('off')
                canvas.draw()
                continue
                
            # Extract data - handle different return formats
            try:
                # Try to unpack with the expected format
                if len(dataset_data) >= 11:  # Full data with colors and origins
                    node_positions, link_pairs, link_colors, node_ids, layers, node_clusters, unique_clusters, node_colors, node_origins, unique_origins, layer_colors = dataset_data
                elif len(dataset_data) >= 9:  # Data with colors but no layer colors
                    node_positions, link_pairs, link_colors, node_ids, layers, node_clusters, unique_clusters, node_colors, node_origins = dataset_data
                    unique_origins = []
                    layer_colors = {}
                elif len(dataset_data) >= 7:  # Basic data
                    node_positions, link_pairs, link_colors, node_ids, layers, node_clusters, unique_clusters = dataset_data
                    node_colors = ['#FFFFFF'] * len(node_positions)
                    node_origins = {}
                    unique_origins = []
                    layer_colors = {}
                else:
                    raise ValueError(f"Dataset has unexpected format: {len(dataset_data)} items")
                
                print(f"Successfully unpacked data for {dataset}: {len(layers)} layers, {len(node_positions)} nodes")
                
                # Calculate additional parameters needed by various chart functions
                
                # Calculate nodes_per_layer (total nodes / number of layers)
                nodes_per_layer = len(node_positions) // len(layers) if layers else 0
                
                # Calculate visible_links (all links for now, could be filtered later)
                visible_links = link_pairs
                
                # Calculate intralayer and interlayer connections
                intralayer_connections = {}
                interlayer_connections = {}
                
                # Count connections per node
                for start_idx, end_idx in link_pairs:
                    # Get base node names
                    start_node = node_ids[start_idx].split('_')[0] if start_idx < len(node_ids) else f"unknown_{start_idx}"
                    end_node = node_ids[end_idx].split('_')[0] if end_idx < len(node_ids) else f"unknown_{end_idx}"
                    
                    # Check if this is an intralayer or interlayer connection
                    start_layer = start_idx // nodes_per_layer if nodes_per_layer else 0
                    end_layer = end_idx // nodes_per_layer if nodes_per_layer else 0
                    
                    if start_layer == end_layer:  # Intralayer connection
                        intralayer_connections[start_node] = intralayer_connections.get(start_node, 0) + 1
                        intralayer_connections[end_node] = intralayer_connections.get(end_node, 0) + 1
                    else:  # Interlayer connection
                        interlayer_connections[start_node] = interlayer_connections.get(start_node, 0) + 1
                        interlayer_connections[end_node] = interlayer_connections.get(end_node, 0) + 1
                
                # Divide by 2 to avoid double counting within the same layer
                for node in intralayer_connections:
                    intralayer_connections[node] = intralayer_connections[node] // 2
                
            except Exception as e:
                print(f"Error unpacking data for {dataset}: {e}")
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Error processing data for {dataset}: {str(e)}", ha='center', va='center')
                ax.axis('off')
                canvas.draw()
                continue
            
            # Create axes for the chart
            if self._current_chart_function.__name__ == 'create_layer_communities_chart':
                # Special case for layer_communities_chart which needs two axes
                gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
                heatmap_ax = fig.add_subplot(gs[0])
                network_ax = fig.add_subplot(gs[1])
                
                # Calculate layer connections
                layer_connections = np.zeros((len(layers), len(layers)), dtype=int)
                for start_idx, end_idx in link_pairs:
                    start_layer = start_idx // nodes_per_layer if nodes_per_layer else 0
                    end_layer = end_idx // nodes_per_layer if nodes_per_layer else 0
                    layer_connections[start_layer, end_layer] += 1
                    if start_layer != end_layer:
                        layer_connections[end_layer, start_layer] += 1
                
                # Call the chart function
                try:
                    print(f"Calling {self._current_chart_function.__name__} for {dataset}")
                    self._current_chart_function(
                        heatmap_ax, network_ax, layer_connections, layers, 
                        medium_font, large_font
                    )
                    fig.suptitle(dataset, fontsize=12)
                    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title
                except Exception as e:
                    print(f"Error generating chart for {dataset}: {e}")
                    logging.error(f"Error generating chart for {dataset}: {e}")
                    fig.clear()
                    ax = fig.add_subplot(111)
                    ax.text(0.5, 0.5, f"Error generating chart for {dataset}: {str(e)}", 
                           ha='center', va='center', wrap=True)
                    ax.axis('off')
            elif self._current_chart_function.__name__ == 'create_node_connections_charts':
                # Special case for node_connections_charts which needs two axes
                gs = gridspec.GridSpec(1, 2)
                intralayer_ax = fig.add_subplot(gs[0])
                interlayer_ax = fig.add_subplot(gs[1])
                
                # Call the chart function
                try:
                    print(f"Calling {self._current_chart_function.__name__} for {dataset}")
                    self._current_chart_function(
                        intralayer_ax, interlayer_ax, visible_links, node_ids,
                        nodes_per_layer, small_font=medium_font, medium_font=large_font
                    )
                    fig.suptitle(dataset, fontsize=12)
                    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title
                except Exception as e:
                    print(f"Error generating chart for {dataset}: {e}")
                    logging.error(f"Error generating chart for {dataset}: {e}")
                    fig.clear()
                    ax = fig.add_subplot(111)
                    ax.text(0.5, 0.5, f"Error generating chart for {dataset}: {str(e)}", 
                           ha='center', va='center', wrap=True)
                    ax.axis('off')
            elif self._current_chart_function.__name__ == 'create_layer_influence_chart':
                # Special case for layer_influence_chart which needs two axes
                gs = gridspec.GridSpec(1, 2)
                bar_ax = fig.add_subplot(gs[0])
                network_ax = fig.add_subplot(gs[1])
                
                # Calculate layer connections
                layer_connections = np.zeros((len(layers), len(layers)), dtype=int)
                for start_idx, end_idx in link_pairs:
                    start_layer = start_idx // nodes_per_layer if nodes_per_layer else 0
                    end_layer = end_idx // nodes_per_layer if nodes_per_layer else 0
                    layer_connections[start_layer, end_layer] += 1
                    if start_layer != end_layer:
                        layer_connections[end_layer, start_layer] += 1
                
                # Call the chart function
                try:
                    print(f"Calling {self._current_chart_function.__name__} for {dataset}")
                    self._current_chart_function(
                        bar_ax, network_ax, layer_connections, layers, 
                        medium_font, large_font
                    )
                    fig.suptitle(dataset, fontsize=12)
                    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title
                except Exception as e:
                    print(f"Error generating chart for {dataset}: {e}")
                    logging.error(f"Error generating chart for {dataset}: {e}")
                    fig.clear()
                    ax = fig.add_subplot(111)
                    ax.text(0.5, 0.5, f"Error generating chart for {dataset}: {str(e)}", 
                           ha='center', va='center', wrap=True)
                    ax.axis('off')
            elif self._current_chart_function.__name__ == 'create_information_flow_chart':
                # Special case for information_flow_chart which needs two axes
                gs = gridspec.GridSpec(1, 2)
                flow_ax = fig.add_subplot(gs[0])
                network_ax = fig.add_subplot(gs[1])
                
                # Calculate layer connections
                layer_connections = np.zeros((len(layers), len(layers)), dtype=int)
                for start_idx, end_idx in link_pairs:
                    start_layer = start_idx // nodes_per_layer if nodes_per_layer else 0
                    end_layer = end_idx // nodes_per_layer if nodes_per_layer else 0
                    layer_connections[start_layer, end_layer] += 1
                    if start_layer != end_layer:
                        layer_connections[end_layer, start_layer] += 1
                
                # Call the chart function
                try:
                    print(f"Calling {self._current_chart_function.__name__} for {dataset}")
                    self._current_chart_function(
                        flow_ax, network_ax, layer_connections, layers, 
                        medium_font, large_font
                    )
                    fig.suptitle(dataset, fontsize=12)
                    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title
                except Exception as e:
                    print(f"Error generating chart for {dataset}: {e}")
                    logging.error(f"Error generating chart for {dataset}: {e}")
                    fig.clear()
                    ax = fig.add_subplot(111)
                    ax.text(0.5, 0.5, f"Error generating chart for {dataset}: {str(e)}", 
                           ha='center', va='center', wrap=True)
                    ax.axis('off')
            else:
                # Generic case for other charts
                ax = fig.add_subplot(111)
                
                # Call the chart function with appropriate arguments based on its signature
                try:
                    sig = inspect.signature(self._current_chart_function)
                    params = {}
                    
                    # Map common parameter names to our available data
                    param_mapping = {
                        'ax': ax,
                        'fig': fig,
                        'node_positions': node_positions,
                        'link_pairs': link_pairs,
                        'link_colors': link_colors,
                        'node_ids': node_ids,
                        'visible_node_ids': node_ids,  # Add visible_node_ids parameter
                        'layers': layers,
                        'node_clusters': node_clusters,
                        'unique_clusters': unique_clusters,
                        'medium_font': medium_font,
                        'large_font': large_font,
                        'layer_connections': None,  # Will be calculated if needed
                        'node_colors': node_colors,
                        'node_origins': node_origins,
                        'unique_origins': unique_origins,
                        'layer_colors': layer_colors,
                        'small_font': {'fontsize': 6},  # Add small font as some charts might need it
                        'visible_layer_indices': list(range(len(layers))),  # Add visible layer indices
                        'visible_links': visible_links,  # Add visible links
                        'nodes_per_layer': nodes_per_layer,  # Add nodes per layer
                        'intralayer_connections': intralayer_connections,  # Add intralayer connections
                        'interlayer_connections': interlayer_connections,  # Add interlayer connections
                    }
                    
                    # Calculate layer connections if needed
                    if 'layer_connections' in sig.parameters and param_mapping['layer_connections'] is None:
                        print(f"Calculating layer connections for {dataset}")
                        layer_connections = np.zeros((len(layers), len(layers)), dtype=int)
                        for start_idx, end_idx in link_pairs:
                            start_layer = start_idx // nodes_per_layer if nodes_per_layer else 0
                            end_layer = end_idx // nodes_per_layer if nodes_per_layer else 0
                            layer_connections[start_layer, end_layer] += 1
                            if start_layer != end_layer:
                                layer_connections[end_layer, start_layer] += 1
                        param_mapping['layer_connections'] = layer_connections
                    
                    # Build parameters based on function signature
                    for param_name in sig.parameters:
                        if param_name in param_mapping:
                            params[param_name] = param_mapping[param_name]
                    
                    # Call the function with the appropriate parameters
                    print(f"Calling {self._current_chart_function.__name__} for {dataset} with params: {list(params.keys())}")
                    self._current_chart_function(**params)
                    fig.suptitle(dataset, fontsize=12)
                    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title
                except Exception as e:
                    print(f"Error generating chart for {dataset}: {e}")
                    logging.error(f"Error generating chart for {dataset}: {e}")
                    fig.clear()
                    ax = fig.add_subplot(111)
                    ax.text(0.5, 0.5, f"Error generating chart for {dataset}: {str(e)}", 
                           ha='center', va='center', wrap=True)
                    ax.axis('off')
            
            # Draw the canvas
            canvas.draw()
    
    def clear_grid_layout(self):
        """Clear all widgets from the grid layout"""
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
    
    def update_stats(self, data_manager):
        """Update with current data"""
        try:
            # Check if data_manager is valid
            if data_manager is None:
                print("No data manager provided to ChartGridPanel.update_stats")
                return
                
            # Define font sizes
            medium_font = {'fontsize': 8}
            large_font = {'fontsize': 10}
            
            # Store data manager and font settings for later use
            self._current_data = (data_manager, medium_font, large_font)
            
            # Clear any existing charts
            self.clear_grid_layout()
            
            # If a chart is selected, enable the generate button
            if self._current_chart_function is not None:
                print("Chart function is available, enabling Generate Charts button")
                self.generate_button.setEnabled(True)
            else:
                print("No chart function available, disabling Generate Charts button")
                self.generate_button.setEnabled(False)
                
            # If a chart is already selected in the dropdown, try to load it
            current_chart = self.chart_dropdown.currentText()
            if current_chart and current_chart != "No charts directory found":
                print(f"Current chart selected: {current_chart}")
                self.on_chart_changed(current_chart)
                
        except Exception as e:
            print(f"Error in ChartGridPanel.update_stats: {e}")
            logging.error(f"Error in update_stats: {e}")
            return 