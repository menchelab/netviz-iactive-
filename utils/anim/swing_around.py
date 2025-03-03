import numpy as np
from vispy import app
from .base_animator import BaseAnimator


class SwingAroundAnimator(BaseAnimator):
    """Swing around animation that swings around the network in an arc"""

    def animate(self):
        """
        Swing around animation - camera swings around the network in an arc,
        creating a dynamic perspective change.
        """
        if self._animation_in_progress:
            return

        self._animation_in_progress = True
        current_state = self._store_camera_state()

        # Determine if we're in orthographic mode by checking fov
        is_orthographic = self.view.camera.fov == 0

        # Animation parameters
        params = {
            "current_state": current_state,
            "is_orthographic": is_orthographic,
            "total_frames": 60,  # Half as long (was 180)
            "swing_angle": 90,  # Angle to swing through
            "elevation_change": 90,  # Maximum elevation change
            "current_frame": 0,
            "restore_original": True,  # Return to original position at end
        }

        # Start the animation
        self._animation_timer = app.Timer(
            interval=1 / 60,
            connect=lambda _: self._animation_step(**params),
            iterations=1,
        )
        self._animation_timer.start()

    def _animation_step(
        self,
        current_state,
        is_orthographic,
        total_frames,
        swing_angle,
        elevation_change,
        current_frame,
        **kwargs,
    ):
        """Execute a single step of the swing around animation"""
        if current_frame >= total_frames:
            # Animation complete, restore original state exactly
            self._restore_camera_state(current_state)
            self._animation_in_progress = False
            return

        # Calculate progress (0 to 1)
        progress = current_frame / total_frames

        # Ensure we end at exactly the original position
        if progress < 0.8:  # First 70% for main swing
            # First 70% - normal swing
            swing_progress = progress / 0.8

            # Use a sine-based easing for a pendulum-like motion
            # This creates an ease-in, then ease-out effect
            angle_progress = np.sin(swing_progress * np.pi)

            # Calculate azimuth change (swing from -angle/2 to +angle/2)
            azimuth_offset = (
                swing_angle * angle_progress * 2
            )  # Multiply by 0.5 to center the swing

            # Calculate elevation change (rise up in the middle of the swing)
            elevation_offset = elevation_change * np.sin(swing_progress * np.pi)

            # Calculate zoom variation (zoom out slightly in the middle of the swing)
            zoom_variation = 1 + 0.3 * np.sin(swing_progress * np.pi)
        else:
            # Last 30% - smooth return to original position
            final_segment = (progress - 0.8) / 0.2
            eased_final = self.ease_in_out_cubic(final_segment)

            # Get the values at the 70% mark to ensure smooth transition
            prev_angle_progress = np.sin(1.0 * np.pi)  # Value at 70%
            prev_azimuth = swing_angle * prev_angle_progress * 2

            # Smoothly interpolate back to 0
            azimuth_offset = prev_azimuth * (1 - eased_final)
            elevation_offset = elevation_change * np.sin(np.pi) * (1 - eased_final)
            zoom_variation = 1  # + (0.999 * (1 - eased_final))

        # Update camera parameters based on camera mode
        if is_orthographic:
            # For orthographic mode, adjust scale_factor
            # For scale_factor, LARGER values zoom in, SMALLER values zoom out
            self.view.camera.scale_factor = current_state["scale_factor"] * (
                1 / zoom_variation
            )
        else:
            # For perspective mode, adjust distance
            # For distance, larger values zoom out
            self.view.camera.distance = current_state["distance"] * zoom_variation

        self.view.camera.azimuth = current_state["azimuth"] + azimuth_offset
        self.view.camera.elevation = current_state["elevation"] + elevation_offset

        # Update the view
        self.view.canvas.update()

        # Schedule next frame with a clean set of parameters
        next_params = {
            "current_state": current_state,
            "is_orthographic": is_orthographic,
            "total_frames": total_frames,
            "swing_angle": swing_angle,
            "elevation_change": elevation_change,
            "current_frame": current_frame,
        }
        next_params.update(kwargs)

        self._schedule_next_frame(self._animation_step, next_params, current_frame)
