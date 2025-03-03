# LC17 Cluster Bridging Analysis

This document provides a comprehensive overview of all analysis types available in the LC17 Cluster Bridging Analysis tool. Each analysis offers a unique perspective on how clusters bridge between layers in multilayer networks.

## Existing Analyses

### 1. Bridge Score Analysis
**Purpose:** Measures how well a cluster connects different layers.

**How it works:** Calculates the ratio of interlayer to total connections within a cluster. Higher scores indicate clusters that effectively bridge between layers.

**Visualization:** Horizontal bar chart showing bridge scores for each cluster, sorted from highest to lowest.

**Applications:** Identify clusters that serve as effective bridges between different layers of the network.

**Related concept:** [Structural Holes Theory](https://en.wikipedia.org/wiki/Structural_holes)

### 2. Flow Efficiency Analysis
**Purpose:** Measures how efficiently information can flow through clusters between layers.

**How it works:** Calculates the inverse of average shortest path length between layers through each cluster. For each layer pair, it identifies the cluster that provides the most efficient path.

**Visualization:** Heatmap showing the best flow efficiency between each pair of layers, with annotations indicating which cluster provides the best flow.

**Applications:** Discover efficient interlayer communication pathways through the network.

**Related concept:** [Network Flow](https://en.wikipedia.org/wiki/Flow_network)

### 3. Layer Span Analysis
**Purpose:** Shows how clusters span across different layers.

**How it works:** Calculates the distribution of a cluster's nodes across different layers. The "span" value indicates how many different layers a cluster bridges.

**Visualization:** Stacked bar chart showing the proportion of each cluster's nodes in different layers, with span values indicated above each bar.

**Applications:** Find clusters that span multiple layers and may play important roles in cross-layer interactions.

**Related concept:** [Multiplex Networks](https://en.wikipedia.org/wiki/Multiplex_network)

### 4. Centrality Distribution Analysis
**Purpose:** Identifies which clusters serve as central hubs in the network.

**How it works:** Calculates betweenness centrality for nodes and aggregates by cluster and layer. Shows which clusters have high centrality across different layers.

**Visualization:** Heatmap with marginal bar charts showing centrality distribution by cluster and layer.

**Applications:** Understand the distribution of betweenness centrality across layers for each cluster.

**Related concept:** [Betweenness Centrality](https://en.wikipedia.org/wiki/Betweenness_centrality)

### 5. Cluster Cohesion Analysis
**Purpose:** Measures the strength of connections within each cluster.

**How it works:** Combines within-layer cohesion (average clustering coefficient) and between-layer cohesion (ratio of interlayer edges) into a single score.

**Visualization:** Horizontal bar chart showing combined cohesion scores for each cluster, with details on internal (I) and external (E) cohesion.

**Applications:** Identify clusters that effectively bridge between layers while maintaining internal structure.

**Related concept:** [Clustering Coefficient](https://en.wikipedia.org/wiki/Clustering_coefficient)

### 6. Information Flow Analysis
**Purpose:** Simulates information flow through the network to identify key bridging clusters.

**How it works:** Uses a diffusion model to simulate how information spreads from one layer to others through each cluster. Higher scores indicate better information transmission.

**Visualization:** Horizontal bar chart showing information flow scores for each cluster.

**Applications:** Analyze how information might flow between layers through specific clusters.

**Related concept:** [Diffusion Process](https://en.wikipedia.org/wiki/Diffusion_of_innovations)

## New Analyses

### 7. Structural Holes Analysis
**Purpose:** Identifies gaps in the network that can be exploited for information or control advantages.

**How it works:** Uses Burt's constraint measure to identify structural holes. Lower constraint (higher score) means more structural holes and more brokerage opportunities.

**Visualization:** Heatmap showing structural hole advantage for each cluster-layer combination.

**Applications:** Identify clusters that bridge structural holes between layers, potentially controlling information flow.

**Related concept:** [Structural Holes](https://en.wikipedia.org/wiki/Structural_holes)

### 8. Cross-Layer Influence Analysis
**Purpose:** Measures how much influence each cluster has across different layers.

**How it works:** Combines eigenvector centrality with cross-layer connectivity to measure how influential each cluster is on different layers.

**Visualization:** Heatmap showing normalized influence scores for each cluster-layer combination.

**Applications:** Identify which clusters are most influential in connecting different layers.

**Related concept:** [Eigenvector Centrality](https://en.wikipedia.org/wiki/Eigenvector_centrality)

### 9. Cluster Resilience Analysis
**Purpose:** Measures how well a cluster maintains connectivity when nodes are removed.

**How it works:** Simulates node removal and measures the change in connectivity. Smaller changes indicate higher resilience.

**Visualization:** Heatmap showing resilience scores for each cluster-layer combination.

**Applications:** Identify which clusters are more resilient to node failures or removals.

**Related concept:** [Network Resilience](https://en.wikipedia.org/wiki/Network_resilience)

### 10. Path Diversity Analysis
**Purpose:** Measures how many different paths exist between layers through each cluster.

**How it works:** Counts edge-disjoint paths between nodes in different layers through each cluster. More paths indicate higher diversity.

**Visualization:** Heatmap showing path diversity scores for each cluster and layer pair.

**Applications:** Identify clusters that provide more diverse routing options between layers.

**Related concept:** [Edge Disjoint Paths](https://en.wikipedia.org/wiki/Menger%27s_theorem)

### 11. Boundary Spanning Analysis
**Purpose:** Measures how effectively clusters connect to other clusters across layer boundaries.

**How it works:** Calculates the ratio of external connections (to other clusters) to total connections for each cluster-layer combination.

**Visualization:** Heatmap showing boundary spanning scores for each cluster-layer combination, with a bar chart showing overall scores.

**Applications:** Identify clusters that serve as bridges between different communities in the network.

**Related concept:** [Boundary Spanning](https://en.wikipedia.org/wiki/Boundary_spanning)

## Disease Network Analyses

### 12. Module Conservation Analysis
**Purpose:** Measures how well the topological structure of clusters is preserved across different layers.

**How it works:** Calculates both node conservation (using Jaccard similarity) and edge conservation between layers, combining them into a weighted score. Higher values indicate more consistent module structure.

**Visualization:** Heatmap showing module conservation scores for each cluster-layer combination.

**Applications:** Identify functionally important modules that are conserved across different data types (PPI, co-expression, etc.) in disease networks.

**Related concept:** [Module Preservation](https://en.wikipedia.org/wiki/Module_(biology))

### 13. Functional Enrichment Analysis
**Purpose:** Simulates functional enrichment by measuring the density of connections within clusters compared to between clusters.

**How it works:** Calculates internal density (connections within the cluster) and external density (connections to other clusters), then computes an enrichment score as the ratio of internal to external density.

**Visualization:** Heatmap showing functional enrichment scores for each cluster-layer combination.

**Applications:** Identify clusters that may represent functionally coherent modules in biological networks, suggesting shared biological processes.

**Related concept:** [Functional Enrichment Analysis](https://en.wikipedia.org/wiki/Enrichment_analysis)

### 14. Disease Association Analysis
**Purpose:** Measures potential disease relevance based on centrality, clustering, and cross-layer presence.

**How it works:** Combines three key metrics: centrality (important nodes tend to be disease-associated), clustering coefficient (disease genes tend to form modules), and cross-layer presence (disease genes tend to be present in multiple data types).

**Visualization:** Heatmap showing disease association scores for each cluster-layer combination.

**Applications:** Identify clusters that may be associated with disease processes in biological networks, prioritizing targets for further investigation.

**Related concept:** [Disease Module](https://en.wikipedia.org/wiki/Disease_module)

### 15. Co-expression Correlation Analysis
**Purpose:** Measures similarity of connectivity patterns between co-expression and other layers.

**How it works:** Compares the neighborhood structure of nodes that appear in both co-expression layers (coexMSG, coexKDN, coexSPL) and other layers (PPI, HP) using Jaccard similarity of neighbor sets.

**Visualization:** Heatmap showing correlation scores for each cluster and layer-pair combination.

**Applications:** Identify clusters with consistent gene relationships across different data types, revealing potential regulatory relationships.

**Related concept:** [Co-expression Network](https://en.wikipedia.org/wiki/Gene_co-expression_network)

### 16. Pathway Alignment Analysis
**Purpose:** Measures how well clusters maintain pathway-like structures across layers.

**How it works:** Evaluates three metrics that indicate pathway coherence: clustering coefficient (higher = more pathway-like), network diameter (lower = more pathway-like), and edge density (higher = more pathway-like).

**Visualization:** Dual visualization with a heatmap showing pathway coherence by layer and a bar chart showing overall pathway alignment scores.

**Applications:** Identify clusters that may represent conserved biological pathways in disease networks, suggesting functional units relevant to the disease process.

**Related concept:** [Pathway Analysis](https://en.wikipedia.org/wiki/Pathway_analysis)

## Applications of LC17 Analyses

These analyses can be used for various purposes in network analysis:

1. **Community Detection:** Identify important bridging communities in multilayer networks
2. **Information Flow Analysis:** Understand how information propagates across different layers
3. **Network Vulnerability Assessment:** Identify critical clusters whose removal would significantly impact network connectivity
4. **Strategic Positioning:** Find optimal positions for new nodes or connections to enhance network performance
5. **Comparative Network Analysis:** Compare the bridging structure of different networks
6. **Temporal Network Analysis:** Track how bridging roles change over time in dynamic networks
7. **Intervention Planning:** Identify key points for interventions in social, biological, or technological networks
8. **Disease Module Identification:** Discover modules associated with specific diseases across multiple data types
9. **Drug Target Prioritization:** Identify potential therapeutic targets based on network position and module membership
10. **Biomarker Discovery:** Find clusters that may contain potential biomarkers for disease diagnosis or prognosis
11. **Pathway Reconstruction:** Reconstruct biological pathways from network data by identifying coherent modules
12. **Multi-omics Integration:** Integrate data from different omics layers to gain comprehensive understanding of disease mechanisms

## References

- Burt, R. S. (1992). Structural Holes: The Social Structure of Competition. Harvard University Press.
- Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
- Kivelä, M., Arenas, A., Barthelemy, M., Gleeson, J. P., Moreno, Y., & Porter, M. A. (2014). Multilayer networks. Journal of Complex Networks, 2(3), 203-271.
- Wasserman, S., & Faust, K. (1994). Social Network Analysis: Methods and Applications. Cambridge University Press.
- Barabási, A. L., Gulbahce, N., & Loscalzo, J. (2011). Network medicine: a network-based approach to human disease. Nature Reviews Genetics, 12(1), 56-68.
- Menche, J., Sharma, A., Kitsak, M., Ghiassian, S. D., Vidal, M., Loscalzo, J., & Barabási, A. L. (2015). Uncovering disease-disease relationships through the incomplete interactome. Science, 347(6224), 1257601.
- Langfelder, P., & Horvath, S. (2008). WGCNA: an R package for weighted correlation network analysis. BMC Bioinformatics, 9(1), 559.