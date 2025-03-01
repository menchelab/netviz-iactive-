import numpy as np
from vispy import scene
from vispy.scene.visuals import Markers, Line, Text
import logging
import time
from vispy import app

from .node_manager import NodeManager
from .edge_manager import EdgeManager
from .label_manager import LabelManager
from .visibility_manager import VisibilityManager
from utils.animation import CameraAnimator


class NetworkCanvas:
    def __init__(self, parent=None, data_manager=None):
        logger = logging.getLogger(__name__)
        logger.info("Creating canvas...")

        # Store reference to data manager
        self.data_manager = data_manager
        if not data_manager:
            logger.error("NetworkCanvas requires a data_manager")
            raise ValueError("NetworkCanvas requires a data_manager")

        # Create canvas and view
        self.canvas = scene.SceneCanvas(keys="interactive", size=(1200, 768))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "turntable"
        self.view.camera.fov = 45
        self.view.camera.distance = 7  # for non orthographic view
        self.view.camera.scale_factor = 4  # for orthographic view

        # Position camera higher on z-axis
        initial_center = self.view.camera.center
        self.view.camera.center = np.array(
            [initial_center[0], initial_center[1], initial_center[2] + 1.5]
        )

        # Set up key event handling
        self.canvas.events.key_press.connect(self._on_key_press)
        # Store initial camera state for reset
        self._initial_camera_state = {
            "center": self.view.camera.center,
            "distance": self.view.camera.distance,
            "elevation": self.view.camera.elevation,
            "azimuth": self.view.camera.azimuth,
            "scale_factor": self.view.camera.scale_factor,
        }

        # Initialize camera animator
        self.camera_animator = CameraAnimator(self.view)

        # Create visuals
        self.scatter = Markers()
        self.view.add(self.scatter)

        # Create separate line visuals for intralayer and interlayer edges
        self.intralayer_lines = Line(
            pos=np.zeros((0, 3)), color=np.zeros((0, 4)), connect="segments", width=1
        )
        self.view.add(self.intralayer_lines)

        self.interlayer_lines = Line(
            pos=np.zeros((0, 3)), color=np.zeros((0, 4)), connect="segments", width=1
        )
        self.view.add(self.interlayer_lines)

        # Create text labels for nodes
        self.node_labels = Text(
            pos=np.array([[0, 0, 0]]), text=[""], color="white", font_size=6
        )
        self.node_labels.visible = False
        self.view.add(self.node_labels)

        # Create line visuals for bar charts
        self.node_count_bars = Line(
            pos=np.zeros((0, 3)), color=np.zeros((0, 4)), connect="segments", width=3
        )
        self.node_count_bars.visible = False
        self.view.add(self.node_count_bars)

        self.edge_count_bars = Line(
            pos=np.zeros((0, 3)), color=np.zeros((0, 4)), connect="segments", width=3
        )
        self.edge_count_bars.visible = False
        self.view.add(self.edge_count_bars)

        # Store current visibility state
        self.current_node_mask = None
        self.current_edge_mask = None
        self.visible_layers = None

        # Initialize managers
        self.node_manager = NodeManager(self)
        self.edge_manager = EdgeManager(self)
        self.label_manager = LabelManager(self)
        self.visibility_manager = VisibilityManager(self)

    def _on_key_press(self, event):
        """Handle key press events for navigation and view control"""

        move_amount = 0.5
        if event.key == "c":
            self._center_view()
        elif event.key == "r":
            self._reset_camera_and_rotation()
        # Camera animation keys
        elif event.key == " ":  # Spacebar - Cosmic Zoom
            self.camera_animator.cosmic_zoom()
        elif event.key == "u":  # Orbit Flyby
            self.camera_animator.orbit_flyby()
        elif event.key == "i":  # Spiral Dive
            self.camera_animator.spiral_dive()
        elif event.key == "o":  # Bounce Zoom
            self.camera_animator.bounce_zoom()
        elif event.key == "j":  # Swing Around
            self.camera_animator.swing_around()
        elif event.key == "k":  # Pulse Zoom
            self.camera_animator.pulse_zoom()
        elif event.key == "l":  # Matrix Effect
            self.camera_animator.matrix_effect()
        elif event.key == "m":  # Slow Matrix Effect
            self.camera_animator.matrix_effect_slow()
        # Arrow keys and WASD for rotation
        elif event.key in ("Left", "a"):
            self._rotate_z(-move_amount)  # Rotate left around z-axis
        elif event.key in ("Right", "d"):
            self._rotate_z(move_amount)  # Rotate right around z-axis
        elif event.key in ("Up", "w"):
            self._rotate_x(move_amount)  # Rotate up around x-axis
        elif event.key in ("Down", "s"):
            self._rotate_x(-move_amount)  # Rotate down around x-axis
        # Q and E for y-axis rotation
        elif event.key == "q":
            self._rotate_y(-move_amount)  # Rotate left around y-axis
        elif event.key == "e":
            self._rotate_y(move_amount)  # Rotate right around y-axis
        # Axis view keys
        elif event.key == "x":
            self._look_along_axis("x")
        elif event.key == "y":
            self._look_along_axis("y")
        elif event.key == "z":
            self._look_along_axis("z")

    def _center_view(self):
        self.view.camera.center = self._initial_camera_state["center"]
        self.view.camera.distance = self._initial_camera_state["distance"]
        self.canvas.update()

    def _reset_camera_and_rotation(self):
        """Reset camera to an isometric diagonal view"""
        # Reset to initial camera state for position and distance
        self.view.camera.center = self._initial_camera_state["center"]
        self.view.camera.distance = self._initial_camera_state["distance"]
        self.view.camera.scale_factor = self._initial_camera_state["scale_factor"]

        # Set to isometric view angles
        # Azimuth: 45 degrees (halfway between x and y axes)
        # Elevation: 35.264 degrees (approximately arctan(1/sqrt(2)), the isometric angle)
        self.view.camera.azimuth = 45
        self.view.camera.elevation = 45

        # Reset roll if available
        if hasattr(self.view.camera, "roll"):
            self.view.camera.roll = 0

        self.canvas.update()

    def _reset_camera(self):
        """Reset camera to initial state"""
        self.view.camera.center = self._initial_camera_state["center"]
        self.view.camera.distance = self._initial_camera_state["distance"]
        self.view.camera.elevation = self._initial_camera_state["elevation"]
        self.view.camera.azimuth = self._initial_camera_state["azimuth"]

    def _rotate_z(self, degrees):
        """Rotate the view around the z-axis"""
        self.view.camera.azimuth += degrees
        self.canvas.update()

    def _rotate_x(self, degrees):
        """Rotate the view around the x-axis"""
        self.view.camera.elevation += degrees
        self.canvas.update()

    def _rotate_y(self, degrees):
        """Rotate the view around the y-axis (roll)"""
        if hasattr(self.view.camera, "roll"):
            self.view.camera.roll += degrees
            self.canvas.update()

    def _look_along_axis(self, axis):
        """Set the camera to look along the specified axis toward the center"""
        # Store current center and distance
        center = self.view.camera.center
        distance = self.view.camera.distance

        # Reset camera orientation
        if axis == "x":
            # Look along x-axis (from positive x)
            self.view.camera.azimuth = 0
            self.view.camera.elevation = 0
        elif axis == "y":
            # Look along y-axis (from positive y)
            self.view.camera.azimuth = 90
            self.view.camera.elevation = 0
        elif axis == "z":
            # Look along z-axis (from top)
            self.view.camera.azimuth = 0
            self.view.camera.elevation = 90

        # Restore center and distance
        self.view.camera.center = center
        self.view.camera.distance = distance

        # Update the view
        self.canvas.update()

    def load_data(
        self,
        node_positions=None,
        link_pairs=None,
        link_colors=None,
        node_colors=None,
        node_ids=None,
    ):
        return self.node_manager.load_data(
            node_positions=node_positions,
            link_pairs=link_pairs,
            link_colors=link_colors,
            node_colors=node_colors,
            node_ids=node_ids,
        )

    def set_layer_colors(self, layer_colors):
        """Set the layer colors mapping"""
        return self.node_manager.set_layer_colors(layer_colors)

    def update_visibility(
        self,
        node_mask=None,
        edge_mask=None,
        show_intralayer=True,
        show_nodes=True,
        show_labels=True,
        bottom_labels_only=True,
        show_stats_bars=False,
    ):
        """Update the visibility of nodes and edges based on masks"""
        return self.visibility_manager.update_visibility(
            node_mask=node_mask,
            edge_mask=edge_mask,
            show_intralayer=show_intralayer,
            show_nodes=show_nodes,
            show_labels=show_labels,
            bottom_labels_only=bottom_labels_only,
            show_stats_bars=show_stats_bars,
        )

    def set_projection_mode(self, orthographic=True):
        if orthographic:
            self.view.camera.fov = 0
        else:
            self.view.camera.fov = 45

        self.canvas.update()
