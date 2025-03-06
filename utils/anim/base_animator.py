import numpy as np
from vispy import app
import logging

logger = logging.getLogger(__name__)


class BaseAnimator:
    """Base class for camera animations"""

    # Default animation duration in seconds
    DEFAULT_DURATION = 1.0
    # Frame rate for animations
    FRAME_RATE = 60

    def __init__(self, view):
        self.view = view
        self._animation_timer = None
        self._animation_in_progress = False

    def _duration_to_frames(self, duration=None):
        """Convert duration in seconds to total frames"""
        if duration is None:
            duration = self.DEFAULT_DURATION
        return int(duration * self.FRAME_RATE)

    def is_animating(self):
        """Check if an animation is currently in progress"""
        return self._animation_in_progress

    def _store_camera_state(self):
        """Store the current camera state for animation reference"""
        return {
            "center": np.array(self.view.camera.center),
            "distance": self.view.camera.distance,
            "elevation": self.view.camera.elevation,
            "azimuth": self.view.camera.azimuth,
            "scale_factor": self.view.camera.scale_factor,
        }

    def _schedule_next_frame(self, animation_func, params, current_frame):
        """Schedule the next frame of an animation"""
        # Increment the current frame
        params["current_frame"] = current_frame + 1

        # Schedule the next frame
        self._animation_timer = app.Timer(
            interval=1 / 60, connect=lambda _: animation_func(**params), iterations=1
        )
        self._animation_timer.start()

    def _restore_camera_state(self, state):
        """Restore the camera to its original state"""
        self.view.camera.scale_factor = state["scale_factor"]
        self.view.camera.elevation = state["elevation"]
        self.view.camera.azimuth = state["azimuth"] % 360
        self.view.camera.center = state["center"]
        self.view.camera.distance = state["distance"]
        self.view.canvas.update()

    # ===== EASING FUNCTIONS =====

    @staticmethod
    def ease_out_cubic(t):
        """Cubic ease out: 1 - (1-t)^3"""
        return 1 - (1 - t) ** 3

    @staticmethod
    def ease_in_cubic(t):
        """Cubic ease in: t^3"""
        return t**3

    @staticmethod
    def ease_in_out_cubic(t):
        """Cubic ease in-out"""
        return 3 * t**2 - 2 * t**3

    @staticmethod
    def ease_out_elastic(t):
        """Elastic ease out"""
        p = 0.3
        return 2 ** (-10 * t) * np.sin((t - p / 4) * (2 * np.pi) / p) + 1

    @staticmethod
    def ease_out_bounce(t):
        """Bounce ease out"""
        if t < 1 / 2.75:
            return 7.5625 * t**2
        elif t < 2 / 2.75:
            t -= 1.5 / 2.75
            return 7.5625 * t**2 + 0.75
        elif t < 2.5 / 2.75:
            t -= 2.25 / 2.75
            return 7.5625 * t**2 + 0.9375
        else:
            t -= 2.625 / 2.75
            return 7.5625 * t**2 + 0.984375
