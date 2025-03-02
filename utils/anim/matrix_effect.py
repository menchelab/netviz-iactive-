import numpy as np
from vispy import app
from .base_animator import BaseAnimator


class MatrixEffectAnimator(BaseAnimator):
    """Matrix-style camera effect with rapid perspective changes"""

    def animate(self):
        """
        Matrix-style camera effect - rapidly changing perspectives with a digital feel.
        Creates a dramatic sequence of camera movements like in the movie "The Matrix".
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
            "total_frames": 150,  # Half as long (was 300)
            "num_glitches": 6,  # Fewer glitches for shorter animation
            "max_scale_change": 1.0,  # Maximum scale factor change
            "max_elevation_change": 70,  # Maximum elevation change
            "max_azimuth_change": 180,  # Maximum azimuth change
            "current_frame": 0,
            "restore_original": True,  # Return to original view at end
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
        num_glitches,
        max_scale_change,
        max_elevation_change,
        max_azimuth_change,
        current_frame,
        **kwargs,
    ):
        """Execute a single step of the matrix effect animation"""
        if current_frame >= total_frames:
            # Animation complete, check if we need to restore original state
            if kwargs.get("restore_original", True):
                self._restore_camera_state(current_state)
            self._animation_in_progress = False
            return

        # Calculate progress (0 to 1)
        progress = current_frame / total_frames

        # Create a series of "glitches" - sudden camera changes
        # Use a sawtooth pattern to create distinct phases
        glitch_phase = (progress * num_glitches) % 1.0

        # Different behavior based on glitch phase
        if glitch_phase < 0.1:  # Quick transition (10% of each glitch cycle)
            # Random-looking but deterministic changes based on current frame
            # This creates a "digital glitch" effect
            phase_progress = glitch_phase / 0.1  # 0 to 1 within this phase

            # Use sine functions with different frequencies for pseudo-random but smooth movement
            scale_change = (
                0.5 * max_scale_change * np.sin(current_frame * 0.1) * phase_progress
            )
            azimuth_offset = (
                max_azimuth_change * np.sin(current_frame * 0.2) * phase_progress
            )
            elevation_offset = (
                max_elevation_change * np.sin(current_frame * 0.3) * phase_progress
            )

        else:  # Hold the new perspective (90% of each glitch cycle)
            # Calculate which glitch we're in
            glitch_index = int(progress * num_glitches)

            # Create deterministic but seemingly random values based on glitch index
            scale_change = 0.5 * max_scale_change * np.sin(glitch_index * 1.5)
            azimuth_offset = max_azimuth_change * np.sin(glitch_index * 2.7)
            elevation_offset = max_elevation_change * np.sin(glitch_index * 3.9)

        # Add a subtle continuous motion
        continuous_azimuth = 20 * np.sin(progress * np.pi * 2)

        # Final approach to original state
        if progress > 0.9:
            # Ease back to original in last 10%
            final_progress = (progress - 0.9) / 0.1
            eased_final = self.ease_in_out_cubic(final_progress)

            scale_change = scale_change * (1 - eased_final)
            azimuth_offset = azimuth_offset * (1 - eased_final)
            elevation_offset = elevation_offset * (1 - eased_final)
            continuous_azimuth = continuous_azimuth * (1 - eased_final)

        # Update camera parameters based on camera mode
        if is_orthographic:
            # For orthographic mode, adjust scale_factor
            self.view.camera.scale_factor = current_state["scale_factor"] * (
                1 + scale_change
            )
        else:
            # For perspective mode, adjust distance
            self.view.camera.distance = current_state["distance"] * (1 + scale_change)

        self.view.camera.azimuth = (
            current_state["azimuth"] + azimuth_offset + continuous_azimuth
        )
        self.view.camera.elevation = current_state["elevation"] + elevation_offset

        # Update the view
        self.view.canvas.update()

        # Schedule next frame with a clean set of parameters
        next_params = {
            "current_state": current_state,
            "is_orthographic": is_orthographic,
            "total_frames": total_frames,
            "num_glitches": num_glitches,
            "max_scale_change": max_scale_change,
            "max_elevation_change": max_elevation_change,
            "max_azimuth_change": max_azimuth_change,
            "current_frame": current_frame,
        }
        next_params.update(kwargs)

        self._schedule_next_frame(self._animation_step, next_params, current_frame)


class MatrixEffectSlowAnimator(BaseAnimator):
    """A slower variant of the matrix effect with continuous rotation"""

    def animate(self):
        """
        A slower variant of the matrix effect with continuous 360° rotation and
        dramatic perspective changes. Features extended holds at key viewpoints.
        Includes minimum 8-frame holds at top and side views.
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
            "total_frames": 60,
            "num_scenes": 4,  # Four distinct scenes including holds
            "current_frame": 0,
            "restore_original": True,
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
        num_scenes,
        current_frame,
        **kwargs,
    ):
        """Execute a single step of the slow matrix effect animation"""
        if current_frame >= total_frames:
            # Animation complete, restore original state exactly
            self._restore_camera_state(current_state)
            self._animation_in_progress = False
            return

        # Calculate progress (0 to 1)
        progress = current_frame / total_frames

        # Calculate continuous 360° rotation across all scenes
        base_rotation = 360 * progress

        # Define scene boundaries (as percentages)
        SCENE_1_END = 0.3  # Initial zoom and rotation
        SCENE_2_END = 0.6  # Top view hold
        SCENE_3_END = 0.9  # Side view hold
        # Remaining 10% for smooth return

        # Determine current scene and scene-specific progress
        if progress < SCENE_1_END:
            # Scene 1: Initial zoom in while rotating
            scene_progress = progress / SCENE_1_END
            eased_scene = self.ease_in_out_cubic(scene_progress)

            elevation = (
                current_state["elevation"]
                + (90 - current_state["elevation"]) * eased_scene
            )
            zoom_factor = 1 + (2 * eased_scene)  # Zoom in to 3x

        elif progress < SCENE_2_END:
            # Scene 2: Hold at top view while continuing rotation
            scene_progress = (progress - SCENE_1_END) / (SCENE_2_END - SCENE_1_END)

            elevation = 90  # Hold at top view
            zoom_factor = 3 - (
                scene_progress * 0.5
            )  # Slight zoom adjustment during hold

        elif progress < SCENE_3_END:
            # Scene 3: Move to and hold side view
            scene_progress = (progress - SCENE_2_END) / (SCENE_3_END - SCENE_2_END)

            if scene_progress < 0.3:  # Quick transition to side view
                transition_progress = scene_progress / 0.3
                eased_transition = self.ease_in_out_cubic(transition_progress)
                elevation = 90 - (
                    90 * eased_transition
                )  # Move from top (90°) to side (0°)
            else:  # Hold at side view
                elevation = 0  # Hold at side view

            zoom_factor = 2.5  # Maintain consistent zoom during side view

        else:
            # Scene 4: Return to original position
            scene_progress = (progress - SCENE_3_END) / (1 - SCENE_3_END)
            eased_scene = self.ease_out_cubic(scene_progress)

            elevation = current_state["elevation"] * eased_scene
            zoom_factor = 2.5 - (1.5 * eased_scene)  # Return to original zoom

        # Add a subtle digital glitch effect during major transitions
        glitch_intensity = 0
        if (
            (progress > SCENE_1_END - 0.05 and progress < SCENE_1_END)
            or (progress > SCENE_2_END - 0.05 and progress < SCENE_2_END)
            or (progress > SCENE_3_END - 0.05 and progress < SCENE_3_END)
        ):
            glitch_progress = min(
                1.0, (0.05 - abs(progress - int(progress / 0.3) * 0.3)) * 20
            )
            glitch_intensity = np.sin(current_frame * 0.5) * glitch_progress
            elevation += glitch_intensity * 2

        # Update camera parameters based on camera mode
        if is_orthographic:
            self.view.camera.scale_factor = current_state["scale_factor"] * (
                1 / zoom_factor
            )
        else:
            self.view.camera.distance = current_state["distance"] * zoom_factor

        self.view.camera.elevation = elevation
        self.view.camera.azimuth = current_state["azimuth"] + base_rotation

        # Update the view
        self.view.canvas.update()

        # Schedule next frame with a clean set of parameters
        next_params = {
            "current_state": current_state,
            "is_orthographic": is_orthographic,
            "total_frames": total_frames,
            "num_scenes": num_scenes,
            "current_frame": current_frame,
        }
        next_params.update(kwargs)

        self._schedule_next_frame(self._animation_step, next_params, current_frame)
