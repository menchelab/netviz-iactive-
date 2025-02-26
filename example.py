import json
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Markers
import logging
from datetime import datetime
from tqdm import tqdm
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QApplication, QHBoxLayout
from PyQt5.QtCore import Qt

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def load_json_data(file_path):
    logger = logging.getLogger(__name__)
    start_time = datetime.now()

    logger.info(f"Starting to load data from {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)

    logger.info("extracting node positions and ids")
    nodes = data['nodes']
    node_positions = np.array([node['pos'] for node in nodes])
    node_ids = [node['id'] for node in nodes]

    # lookup dict for speed brrrr
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}

    logger.info("extracting links")
    links = data['links']
    total_links = len(links)

    # Pre-allocate arrays for better performance
    link_pairs = np.zeros((total_links, 2), dtype=np.int32)
    link_colors = []

    for i, link in enumerate(tqdm(links, desc="Processing links")):
        link_pairs[i] = [
            node_id_to_index[link['source']], 
            node_id_to_index[link['target']]
        ]
        link_colors.append(link['linkcolor'])

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("Network statistics:")
    logger.info(f"  - Number of nodes: {len(nodes)}")
    logger.info(f"  - Number of links: {len(links)}")
    logger.info(f"Data loading completed in {duration:.2f} seconds")

    return node_positions, link_pairs, link_colors, node_ids

def netviz(node_positions, link_pairs, link_colors, node_ids):
    logger = logging.getLogger(__name__)
    start_time = datetime.now()

    logger.info("Starting network visualization optimization")

    # using z values to group nodes (since each z value is a layer I guess)
    # Get unique Z values and create initial mask
    unique_z = np.unique(node_positions[:, 2])
    logger.info(f"Found Z levels: {unique_z}")
    z_masks = {z: True for z in unique_z}

    # Create the main Qt widget
    main_widget = QWidget()
    layout = QHBoxLayout(main_widget)  # Changed to QHBoxLayout for side-by-side

    # Create control panel
    control_panel = QWidget()
    control_layout = QVBoxLayout(control_panel)
    control_panel.setMaximumWidth(200) 

    # Create checkboxes for each Z value
    for z in sorted(unique_z):
        checkbox = QCheckBox(f"Z Level: {z:.2f}")
        checkbox.setChecked(True)

        def make_callback(z_val):
            def callback(state):
                z_masks[z_val] = (state == Qt.Checked)
                update_visibility()
                canvas.update()
            return callback

        checkbox.stateChanged.connect(make_callback(z))
        control_layout.addWidget(checkbox)

    # Add control panel to main layout (left side)
    layout.addWidget(control_panel)

    # Create the vispy canvas for the network
    canvas = scene.SceneCanvas(
        keys='interactive',
        size=(1200, 900),
        show=False,  # Don't show yet
        title='datadivr - network viewer'
    )

    # Create view for the network
    view = canvas.central_widget.add_grid()
    network_view = view.add_view(row=0, col=0)

    layout.addWidget(canvas.native)

    # Normalize positions (in opengl all coordinates are between -1 and 1)
    pos_min = np.min(node_positions, axis=0)
    pos_max = np.max(node_positions, axis=0)
    nodepos = (node_positions - pos_min) / (pos_max - pos_min)

    # (not required)fancy colors for nodes based on pos
    num_nodes = len(node_positions)
    normalized_G = (nodepos[:, 2] * 255).astype(float)
    normalized_B = (nodepos[:, 1] * 255).astype(float)
    nodecol = np.column_stack([
        np.full(num_nodes, 255, dtype=float),
        normalized_G,
        normalized_B,
        np.full(num_nodes, 100, dtype=float),
    ]) / 255.0

    # Create node markers
    node_markers = scene.visuals.Markers()
    node_markers.set_gl_state('translucent', blend=True, depth_test=True)

    # Create lines
    start_positions = nodepos[link_pairs[:, 0]]
    end_positions = nodepos[link_pairs[:, 1]]
    line_positions = np.vstack([start_positions, end_positions]).reshape(-1, 3)

    def hex_to_rgba(hex_color):
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (*[x/255.0 for x in rgb], 0.4)

    # should be using rgb tuple from the start actually
    linkcol = np.array([hex_to_rgba(color) for color in link_colors])
    linkcol = np.repeat(linkcol, 2, axis=0)[:len(line_positions)]

    lines = scene.visuals.Line(pos=line_positions, color=linkcol, width=0.5, connect="segments")

    def update_visibility():
        # Create node mask based on Z values
        # we use numpy masks because its much faster for vizp to mask vis. than recreate nodes/links
        node_mask = np.zeros(len(nodepos), dtype=bool)
        for z, is_visible in z_masks.items():
            if is_visible:
                node_mask |= (node_positions[:, 2] == z)

        # Update node visibility
        visible_nodepos = nodepos[node_mask]
        visible_nodecol = nodecol[node_mask]
        node_markers.set_data(
            pos=visible_nodepos,
            face_color=visible_nodecol,
            size=3,
            edge_color=None
        )

        # Update link visibility
        link_mask = node_mask[link_pairs[:, 0]] & node_mask[link_pairs[:, 1]]
        visible_start_pos = start_positions[link_mask]
        visible_end_pos = end_positions[link_mask]
        visible_line_pos = np.vstack([visible_start_pos, visible_end_pos]).reshape(-1, 3)

        # Create the repeated mask for the line colors
        repeated_mask = np.repeat(link_mask, 2)
        visible_linkcol = linkcol[repeated_mask]

        # actually applying to the viz
        lines.set_data(pos=visible_line_pos, color=visible_linkcol)

    # Add visuals to view
    network_view.add(node_markers)
    network_view.add(lines)

    # Initial visibility update
    update_visibility()

    # Camera setup
    network_view.camera = 'arcball'
    network_view.camera.fov = 45
    network_view.camera.distance = 2.5
    network_view.camera.set_range()

    main_widget.show()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Visualization optimization completed in {duration:.2f} seconds")

    return main_widget

def main():
    # Create QApplication instance first (NOT after the netviz function)
    # qt only way i found to implement UI somehow smoothly
    app_instance = QApplication([])

    logger = setup_logging()
    start_time = datetime.now()

    file_path = "Overlapping_Endotypes_Multiplex_Network.json"
    # load data from json

    # not sure what you use, but at this point we want/need numpy arrays
    # lets talk in person for a good efficient data structure that supports all annotations
    # but i guess it will be something like numpyarray for each attribute
    # see how i do it for now here: https://github.com/menchelab/datadivr/blob/main/datadivr/project/model.py
    #
    # also think about potential filtering etc
    # maybe additionally to checkboxes, there are single select dropdowns, value cutoff sliders (eg for edge weights)

    # would like to have the UI filtering automatically generated based on data & hints
    # so you would just say viz.addcheckbox(attribute_name), addslider(attribute_name)
    # so basically data driven UI
    # instead of hardcoded like now with the checkboxes
    node_positions, link_pairs, link_colors, node_ids = load_json_data(file_path)

    # create  actual viz
    main_widget = netviz(node_positions, link_pairs, link_colors, node_ids)


    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    logger.info(f"Total setup time: {total_duration:.2f} seconds")

    logger.info("Starting viz loop")
    # Use Qt's event loop instead of vispy's
    app_instance.exec_()

if __name__ == "__main__":
    main()
