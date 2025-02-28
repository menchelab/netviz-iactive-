import collections


def create_cluster_distribution_chart(
    ax, visible_node_ids, node_clusters, small_font, medium_font, cluster_colors=None
):
    """Create pie chart showing cluster distribution"""
    cluster_counts = collections.Counter(
        [node_clusters[node_id] for node_id in visible_node_ids]
    )

    if cluster_counts:
        clusters, counts = zip(*cluster_counts.most_common())

        # Prepare colors for the pie chart
        colors = None
        if cluster_colors:
            colors = [cluster_colors.get(cluster, None) for cluster in clusters]
            # Filter out None values (clusters without defined colors)
            if all(color is None for color in colors):
                colors = None  # Use default colors if none are defined

        ax.pie(
            counts,
            labels=clusters,
            autopct="%1.1f%%",
            startangle=90,
            textprops=small_font,
            colors=colors,
        )
        ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_title("Cluster Distribution", **medium_font)
    else:
        ax.text(
            0.5,
            0.5,
            "No clusters to display",
            horizontalalignment="center",
            verticalalignment="center",
            **small_font,
        )
