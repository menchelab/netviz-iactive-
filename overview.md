# Network Visualization Tool Overview

This document provides an overview of the Network Visualization Tool's architecture and components, organized by analysis panels and their associated visualizations.

## Core Architecture

### Entry Point (`main.py`)
- Initializes PyQt5 GUI application with QApplication instance
- Configures logging system with timestamp-based file handlers
- Implements event loop for main visualization window
- Manages data directory paths and configuration loading

### Core UI Components
- **Main Window** (`ui/main_window.py`): 
  - QMainWindow implementation with dock widgets
  - Signal/slot connections for inter-component communication
  - Event handling for user interactions
  - Layout management with QGridLayout

- **Control Panel** (`ui/control_panel.py`): 
  - QWidget-based control interface
  - Dynamic widget generation based on available options
  - State management for filter settings
  - Event propagation to visualization components

- **Network Canvas** (`ui/network_canvas/`): 
  - Custom QGraphicsScene implementation
  - OpenGL acceleration for large networks
  - Interactive zoom and pan capabilities
  - Custom shader programs for node/edge rendering

## Stats Panel System (`stats_panel/stats_panel.py`)

The stats panel system is implemented as a QTabWidget containing specialized analysis panels. Each tab provides different network analysis capabilities.

### 1. Main Statistics Tab (`stats_panel/main_stats_panel.py`)
Comprehensive network statistics dashboard with six main visualizations:

**Left Column Charts**:
1. **Layer Connectivity Matrix**
   ```python
   def create_layer_connectivity_chart(ax, links, nodes_per_layer, layers, small_font, medium_font):
       matrix = calculate_connectivity_matrix(links, nodes_per_layer)
       return plot_connectivity_heatmap(ax, matrix, layers)
   ```
   - Connection density visualization
   - Inter-layer relationship mapping
   - Hierarchical clustering of layers

2. **Cluster Distribution**
   ```python
   def create_cluster_distribution_chart(ax, node_ids, clusters, fonts, colors):
       distribution = calculate_cluster_sizes(node_ids, clusters)
       return plot_distribution(ax, distribution, colors)
   ```
   - Cluster size analysis
   - Distribution fitting
   - Statistical metrics

3. **Layer Activity**
   - Temporal activity patterns
   - Activity density metrics
   - Cross-layer correlations

**Right Column Charts**:
1. **Betweenness Centrality**
   - Node importance metrics
   - Path analysis
   - Centrality distribution

2. **Interlayer Graph**
   - Network topology visualization
   - Layer relationship structure
   - Interactive graph layout

3. **Layer Similarity**
   - Hierarchical clustering
   - Similarity metrics
   - Dendrogram visualization

### 2. Sankey Analysis Tab (`stats_panel/sankey_panel.py`)
Flow and relationship visualization system:

**Components**:
1. **Sankey Diagram**
   ```python
   def create_sankey_diagram(flows, nodes, colors):
       layout = optimize_node_positions(nodes, flows)
       return render_sankey(layout, flows, colors)
   ```
   - Flow optimization
   - Node positioning
   - Interactive highlighting

2. **Flow Analysis**
   - Flow volume calculation
   - Path optimization
   - Bottleneck detection

### 3. Layer Graph Tab (`stats_panel/layer_graph_panel.py`)
Layer-level network structure analysis:

**Features**:
1. **Graph Metrics**
   ```python
   def calculate_layer_metrics(G):
       density = nx.density(G)
       clustering = nx.average_clustering(G)
       centrality = nx.eigenvector_centrality_numpy(G)
       return density, clustering, centrality
   ```
   - Density calculation
   - Clustering analysis
   - Centrality measures

2. **Visualization**
   - Force-directed layout
   - Edge bundling
   - Interactive navigation

### 4. Layer Communities Tab (`stats_panel/layer_communities_panel.py`)
Community detection and analysis:

**Algorithms**:
1. **Louvain Method**
   ```python
   def louvain_community_detection(G):
       communities = nx.community.louvain_communities(G)
       modularity = nx.community.modularity(G, communities)
       return communities, modularity
   ```
   - Time complexity: O(N log N)
   - Space complexity: O(N + M)
   - Resolution parameter tuning

2. **Infomap Algorithm**
   - MDL optimization
   - Hierarchical decomposition
   - Path encoding

3. **Spectral Clustering**
   - Laplacian computation
   - Eigendecomposition
   - K-means clustering

### 5. Information Flow Tab (`stats_panel/information_flow_panel.py`)
Information propagation analysis:

**Metrics**:
1. **Flow Centrality**
   ```python
   def calculate_flow_metrics(G):
       bc = nx.betweenness_centrality(G, weight='weight')
       fb = nx.current_flow_betweenness_centrality(G)
       ic = nx.information_centrality(G)
       return bc, fb, ic
   ```
   - Betweenness measures
   - Flow patterns
   - Information paths

2. **Visualizations**
   - Flow heatmaps
   - Network diagrams
   - Metric distributions

### 6. Layer Coupling Tab (`stats_panel/layer_coupling_panel.py`)
Inter-layer relationship analysis:

**Methods**:
1. **Statistical Coupling**
   ```python
   def calculate_coupling_metrics(layer1, layer2):
       pearson = np.corrcoef(layer1, layer2)[0,1]
       mi = mutual_info_score(layer1, layer2)
       cross_corr = signal.correlate2d(layer1, layer2)
       return pearson, mi, cross_corr
   ```
   - Correlation analysis
   - Mutual information
   - Cross-correlation

2. **Structural Coupling**
   - Jaccard similarity
   - Topological overlap
   - Custom metrics

### 7. Critical Structure Tab (`stats_panel/critical_structure_panel.py`)
Network vulnerability analysis:

**Analysis Methods**:
1. **Connectivity Impact**
   ```python
   def analyze_connectivity_impact(G):
       cut_vertices = list(nx.articulation_points(G))
       edge_conn = nx.edge_connectivity(G)
       components = list(nx.connected_components(G))
       return cut_vertices, edge_conn, components
   ```
   - Component analysis
   - Cut-vertex detection
   - Edge connectivity

2. **Centrality Impact**
   - Betweenness analysis
   - Eigenvector centrality
   - PageRank influence

### 8. Multi-Dataset Analysis Tab (`stats_panel/chart_grid_panel.py`)
Comparative analysis system:

**Features**:
1. **Grid Layout**
   - Dynamic chart arrangement
   - Customizable grid size
   - Synchronized views

2. **Chart Types**
   - All available visualizations
   - Comparative metrics
   - Cross-dataset analysis
