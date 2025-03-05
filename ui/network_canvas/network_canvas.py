import numpy as np
from vispy import scene
from vispy.scene.visuals import Markers, Line, Text
import logging
import time
from vispy import app
from datetime import datetime
import os
from vispy.scene import Node
import imageio.v3 as imageio

from .node_manager import NodeManager
from .edge_manager import EdgeManager
from .label_manager import LabelManager
from .visibility_manager import VisibilityManager
from utils.anim.animation_manager import AnimationManager

logger = logging.getLogger(__name__)


class NetworkCanvas:
    def __init__(self, parent=None, data_manager=None):
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

        # Initialize animation manager
        self.animation_manager = AnimationManager(self.view)

        # Create visuals
        self.scatter = Markers(spherical=True)
        self.view.add(self.scatter)

        # Create separate line visuals for intralayer and interlayer edges
        self.intralayer_lines = Line(
            pos=np.zeros((0, 3)),
            color=np.zeros((0, 4)),
            connect="segments",
            width=1,
            antialias=True,
            method="gl",
        )
        self.view.add(self.intralayer_lines)

        self.interlayer_lines = Line(
            pos=np.zeros((0, 3)),
            color=np.zeros((0, 4)),
            connect="segments",
            width=1,
            antialias=True,
            method="gl",
        )
        self.view.add(self.interlayer_lines)

        self.intralayer_lines.set_gl_state("additive")
        self.interlayer_lines.set_gl_state("additive")
        self.scatter.set_gl_state("additive")

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
        elif event.key == "b":
            self._save_simple_screenshot()
        # Camera animation keys
        elif event.key == " ":  # Spacebar - Cosmic Zoom
            self.animation_manager.play_animation(AnimationManager.COSMIC_ZOOM)
        elif event.key == "u":  # Orbit Flyby
            self.animation_manager.play_animation(AnimationManager.ORBIT_FLYBY)
        elif event.key == "i":  # Spiral Dive
            self.animation_manager.play_animation(AnimationManager.SPIRAL_DIVE)
        elif event.key == "o":  # Bounce Zoom
            self.animation_manager.play_animation(AnimationManager.BOUNCE_ZOOM)
        elif event.key == "j":  # Swing Around
            self.animation_manager.play_animation(AnimationManager.SWING_AROUND)
        elif event.key == "k":  # Pulse Zoom
            self.animation_manager.play_animation(AnimationManager.PULSE_ZOOM)
        elif event.key == "l":  # Matrix Effect
            self.animation_manager.play_animation(AnimationManager.MATRIX_EFFECT)
        elif event.key == "m":  # Slow Matrix Effect
            self.animation_manager.play_animation(AnimationManager.MATRIX_EFFECT_SLOW)
        # Screenshot key
        elif event.key == "v":
            self._save_screenshot()
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
        intralayer_width=1.0,
        interlayer_width=1.0,
        intralayer_opacity=1.0,
        interlayer_opacity=1.0,
        node_size=1.0,
        node_opacity=1.0,
        antialias=True,
        gl_state="additive",
    ):
        """Update the visibility of nodes and edges based on masks and display settings"""
        return self.visibility_manager.update_visibility(
            node_mask=node_mask,
            edge_mask=edge_mask,
            show_intralayer=show_intralayer,
            show_nodes=show_nodes,
            show_labels=show_labels,
            bottom_labels_only=bottom_labels_only,
            show_stats_bars=show_stats_bars,
            intralayer_width=intralayer_width,
            interlayer_width=interlayer_width,
            intralayer_opacity=intralayer_opacity,
            interlayer_opacity=interlayer_opacity,
            node_size=node_size,
            node_opacity=node_opacity,
            antialias=antialias,
            gl_state=gl_state,
        )

    def set_projection_mode(self, orthographic=True):
        if orthographic:
            self.view.camera.fov = 0
        else:
            self.view.camera.fov = 45

        self.canvas.update()

    def _save_screenshot(self):
        """Save high-resolution screenshots with additive blending and different depth/blend combinations"""
        import numpy as np
        from vispy.scene import Node
        from vispy.gloo import set_state
        import imageio.v3 as imageio

        # Store original size and configuration
        original_size = self.canvas.size

        # Define combinations to test (all with additive blending)
        combinations = {
            "depth_blend": {
                "depth_test": True,
                "blend": True,
                "blend_func": ("src_alpha", "one", "one", "one"),  # Additive
                "blend_equation": "func_add",
            },
            "depth_only": {
                "depth_test": True,
                "blend": False,
                "blend_func": ("src_alpha", "one", "one", "one"),
                "blend_equation": "func_add",
            },
            "blend_only": {
                "depth_test": False,
                "blend": True,
                "blend_func": ("src_alpha", "one", "one", "one"),
                "blend_equation": "func_add",
            },
            "neither": {
                "depth_test": False,
                "blend": False,
                "blend_func": ("src_alpha", "one", "one", "one"),
                "blend_equation": "func_add",
            },
        }

        try:
            # Calculate high-res dimensions (4x)
            high_res_width = self.canvas.size[0] * 4
            high_res_height = self.canvas.size[1] * 4

            # Create screenshots directory if it doesn't exist
            os.makedirs("screenshots", exist_ok=True)

            # Get timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Use a single set of visual parameters
            line_width = 3.0

            # Create a temporary parent node
            temp_parent = Node(parent=self.view.scene)

            for combo_name, gl_state in combinations.items():
                # Create temporary line visuals
                temp_intralayer = Line(
                    pos=self.intralayer_lines._pos,
                    color=self.intralayer_lines._color,
                    connect="segments",
                    width=line_width,
                    antialias=True,
                    parent=temp_parent,
                    method="gl",
                )

                temp_interlayer = Line(
                    pos=self.interlayer_lines._pos,
                    color=self.interlayer_lines._color,
                    connect="segments",
                    width=line_width,
                    antialias=True,
                    parent=temp_parent,
                    method="gl",
                )

                # Apply GL state
                temp_intralayer.set_gl_state(**gl_state)
                temp_interlayer.set_gl_state(**gl_state)

                # Hide original lines
                self.intralayer_lines.visible = False
                self.interlayer_lines.visible = False

                # Set canvas size and background
                self.canvas.size = (high_res_width, high_res_height)
                self.canvas.bgcolor = (0, 0, 0, 1)

                # Update visuals
                temp_intralayer.update()
                temp_interlayer.update()
                self.scatter.update()
                self.canvas.update()

                # Render
                img = self.canvas.render()

                # Save the image
                filepath = f"screenshots/export_additive_{combo_name}_{timestamp}.png"
                imageio.imwrite(filepath, img)
                print(f"Saved additive {combo_name} to {os.path.abspath(filepath)}")

                # Clean up current visuals
                temp_parent.parent = None
                temp_parent = Node(parent=self.view.scene)

        except Exception as e:
            print(f"Error saving screenshots: {str(e)}")
            import logging

            logging.getLogger(__name__).error(f"Screenshot error: {str(e)}")

        finally:
            # Restore original settings
            self.canvas.size = original_size
            self.canvas.bgcolor = (0, 0, 0, 1)
            self.intralayer_lines.visible = True
            self.interlayer_lines.visible = True
            self.canvas.update()

    def _save_simple_screenshot(self):
        """Save a simple high-resolution screenshot with 4x resolution"""
        try:
            original_size = self.canvas.size

            high_res_width = self.canvas.size[0] * 4
            high_res_height = self.canvas.size[1] * 4

            os.makedirs("screenshots", exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            self.canvas.size = (high_res_width, high_res_height)
            self.canvas.update()

            # Render and save
            img = self.canvas.render()
            filepath = f"screenshots/screenshot_{timestamp}.png"
            imageio.imwrite(filepath, img)
            print(f"Saved screenshot to {os.path.abspath(filepath)}")

        except Exception as e:
            print(f"Error saving screenshot: {str(e)}")
            logger.error(f"Screenshot error: {str(e)}")

        finally:
            self.canvas.size = original_size
            self.canvas.update()
