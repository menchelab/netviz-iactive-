import numpy as np
from vispy import app
from .base_animator import BaseAnimator


class PulseZoomAnimator(BaseAnimator):
    """Pulse zoom animation that pulses in and out with a rhythmic motion"""

    def animate(self, duration=None):
        """
        Pulse zoom animation - camera pulses in and out while rotating.
        Creates a rhythmic zooming effect combined with gentle rotation.

        Args:
            duration (float, optional): Duration of the animation in seconds.
                                      If None, uses the default duration.
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
            "total_frames": self._duration_to_frames(duration),
            "pulse_cycles": 3,  # Number of pulse cycles
            "rotation_amount": 180,  # Total rotation amount in degrees
            "current_frame": 0,
            "restore_original": True,
        }

        # Start the animation
        self._animation_timer = app.Timer(
            interval=1 / self.FRAME_RATE,
            connect=lambda _: self._animation_step(**params),
            iterations=1,
        )
        self._animation_timer.start()

    def _animation_step(
        self,
        current_state,
        is_orthographic,
        total_frames,
        pulse_cycles,
        rotation_amount,
        current_frame,
        **kwargs,
    ):
        """Execute a single step of the pulse zoom animation"""
        if current_frame >= total_frames:
            # Animation complete, restore original state exactly
            self._restore_camera_state(current_state)
            self._animation_in_progress = False
            return

        # Calculate progress (0 to 1)
        progress = current_frame / total_frames

        # Split animation into main movement (85%) and smooth landing (15%)
        if progress < 0.85:
            # Main animation phase (85% of total time)
            main_progress = progress / 0.85

            # Calculate pulsing zoom factor using sine wave
            pulse_factor = 1 + 2 * (
                0.5 + 0.5 * np.sin(main_progress * np.pi * 2 * pulse_cycles)
            )

            # Calculate rotation with easing
            azimuth_change = rotation_amount * self.ease_in_out_cubic(main_progress)

            # Add subtle elevation changes synchronized with the pulse
            elevation_change = 10 * np.sin(main_progress * np.pi * 2 * pulse_cycles)
        else:
            # Smooth landing phase (last 15%)
            final_progress = (progress - 0.85) / 0.15
            eased_final = self.ease_out_cubic(final_progress)

            # Get the values at the 85% mark
            prev_main_progress = 1.0  # End of main phase
            prev_pulse = 1 + 2 * (
                0.5 + 0.5 * np.sin(prev_main_progress * np.pi * 2 * pulse_cycles)
            )
            prev_azimuth = rotation_amount * self.ease_in_out_cubic(prev_main_progress)
            prev_elevation = 10 * np.sin(prev_main_progress * np.pi * 2 * pulse_cycles)

            # Smoothly interpolate all parameters back to original values
            pulse_factor = 1 + (prev_pulse - 1) * (1 - eased_final)
            azimuth_change = (
                prev_azimuth + (rotation_amount - prev_azimuth) * eased_final
            )
            elevation_change = prev_elevation * (1 - eased_final)

        # Update camera parameters based on camera mode
        if is_orthographic:
            # For orthographic mode, adjust scale_factor
            self.view.camera.scale_factor = current_state["scale_factor"] * (
                1 / pulse_factor
            )
        else:
            # For perspective mode, adjust distance
            # For distance, larger values zoom out
            self.view.camera.distance = current_state["distance"] * pulse_factor

        self.view.camera.azimuth = current_state["azimuth"] + azimuth_change
        self.view.camera.elevation = current_state["elevation"] + elevation_change

        # Update the view
        self.view.canvas.update()

        # Schedule next frame with a clean set of parameters
        next_params = {
            "current_state": current_state,
            "is_orthographic": is_orthographic,
            "total_frames": total_frames,
            "pulse_cycles": pulse_cycles,
            "rotation_amount": rotation_amount,
            "current_frame": current_frame,
        }
        next_params.update(kwargs)

        self._schedule_next_frame(self._animation_step, next_params, current_frame)
