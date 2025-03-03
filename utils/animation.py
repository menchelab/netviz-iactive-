import numpy as np
from vispy import app
import logging

logger = logging.getLogger(__name__)


class CameraAnimator:
    """
    Handles camera animations for the network visualization.
    Provides various animation effects for camera movement.
    """

    def __init__(self, canvas_view):
        """
        Initialize the animator with a reference to the canvas view.

        Parameters:
        -----------
        canvas_view : vispy.scene.widgets.ViewBox
            The view to animate
        """
        self.view = canvas_view
        self._animation_timer = None
        self._animation_in_progress = False

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

    # ===== ANIMATION FUNCTIONS =====

    def cosmic_zoom(self):
        """
        Cosmic zoom animation - starts from very far away and zooms in with rotation.
        Gives a sense of diving into the network from outer space.

        Works in both orthographic mode (adjusts scale_factor) and
        non-orthographic mode (adjusts distance).
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
            "rotation_amount": 360,
            "elevation_change": 60,
            "current_frame": 0,
            "restore_original": True,
        }

        # Start the animation
        self._animation_timer = app.Timer(
            interval=1 / 60,
            connect=lambda _: self._cosmic_zoom_step(**params),
            iterations=1,
        )
        self._animation_timer.start()

    def _cosmic_zoom_step(
        self,
        current_state,
        is_orthographic,
        total_frames,
        rotation_amount,
        elevation_change,
        current_frame,
        **kwargs,
    ):
        """Execute a single step of the cosmic zoom animation"""
        if current_frame >= total_frames:
            # Animation complete, restore original state exactly
            self._restore_camera_state(current_state)
            self._animation_in_progress = False
            return

        # Calculate progress (0 to 1)
        progress = current_frame / total_frames

        # Custom easing with faster start and gradual slowdown at the end
        if progress < 0.2:
            # First 50% - fast ease-in cubic
            eased_progress = (
                self.ease_in_cubic(progress * 2) * 0.7
            )  # Get to 70% of the way quickly
        else:
            # Last 50% - gradual slowdown
            segment_progress = (progress - 0.5) / 0.5
            eased_progress = 0.7 + (self.ease_out_cubic(segment_progress) * 0.3)

        # Handle zoom differently based on camera mode
        if is_orthographic:
            # For orthographic mode, we use scale_factor
            # Start with a large scale factor (zoomed out) and decrease it (zoom in)
            start_scale_factor = (
                current_state["scale_factor"] * 10
            )  # Start extremely zoomed out

            # Create a zoom curve that goes in, then slightly out, then back to original
            zoom_progress = 1.0
            if progress < 0.8:
                zoom_progress = eased_progress / 0.8  # Zoom in faster
            else:
                # Return to original in last 20%
                final_segment = (progress - 0.8) / 0.2
                zoom_progress = 1.0

            target_scale = (
                start_scale_factor * (1 - zoom_progress)
                + current_state["scale_factor"] * zoom_progress
            )
            self.view.camera.scale_factor = target_scale
        else:
            # For perspective mode, we use distance
            # Start with a large distance (zoomed out) and decrease it (zoom in)
            start_distance = current_state["distance"] * 30  # Start extremely far away

            # Create a zoom curve that goes in, then slightly out, then back to original
            zoom_progress = 1.0
            if progress < 0.1:
                zoom_progress = eased_progress / 0.2  # Zoom in faster
            else:
                # Return to original in last 20%
                final_segment = (progress - 0.8) / 0.2
                zoom_progress = 1.0

            target_distance = (
                start_distance * (1 - zoom_progress)
                + current_state["distance"] * zoom_progress
            )
            self.view.camera.distance = target_distance

        # Calculate rotation - complete most rotation early, then slow down
        if progress < 0.7:
            rotation_progress = progress / 0.7
            azimuth_change = rotation_amount * self.ease_in_out_cubic(rotation_progress)
        else:
            # Slow down rotation at the end and return to original
            final_segment = (progress - 0.7) / 0.3
            azimuth_change = rotation_amount * (
                1 - self.ease_in_out_cubic(final_segment)
            )

        # Calculate elevation change (arc motion)
        if progress < 0.8:
            # Rise up in first 80%
            elevation_segment = progress / 0.8
            elevation_change_current = elevation_change * np.sin(
                elevation_segment * np.pi
            )
        else:
            # Return to original elevation in last 20%
            final_segment = (progress - 0.8) / 0.2
            elevation_change_current = (
                elevation_change * np.sin(np.pi) * (1 - final_segment)
            )

        # Add oscillation that reduces as we zoom in
        oscillation_amplitude = 5 * (1 - eased_progress)
        oscillation = np.sin(progress * np.pi * 6) * oscillation_amplitude

        # Update camera parameters
        self.view.camera.azimuth = current_state["azimuth"] + azimuth_change
        self.view.camera.elevation = (
            current_state["elevation"] + elevation_change_current + oscillation
        )

        # Update the view
        self.view.canvas.update()

        # Schedule next frame with a clean set of parameters
        next_params = {
            "current_state": current_state,
            "is_orthographic": is_orthographic,
            "total_frames": total_frames,
            "rotation_amount": rotation_amount,
            "elevation_change": elevation_change,
            "current_frame": current_frame,
        }
        next_params.update(kwargs)

        self._schedule_next_frame(self._cosmic_zoom_step, next_params, current_frame)

    def matrix_effect(self):
        """
        Matrix-style camera effect - rapidly changing perspectives with a digital feel.
        Creates a dramatic sequence of camera movements like in the movie "The Matrix".

        Works in both orthographic mode (adjusts scale_factor) and
        non-orthographic mode (adjusts distance).
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
            connect=lambda _: self._matrix_effect_step(**params),
            iterations=1,
        )
        self._animation_timer.start()

    def _matrix_effect_step(
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

        self._schedule_next_frame(self._matrix_effect_step, next_params, current_frame)

    def matrix_effect_slow(self):
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
            connect=lambda _: self._matrix_effect_slow_step(**params),
            iterations=1,
        )
        self._animation_timer.start()

    def _matrix_effect_slow_step(
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
            
            elevation = current_state["elevation"] + (90 - current_state["elevation"]) * eased_scene
            zoom_factor = 1 + (2 * eased_scene)  # Zoom in to 3x
            
        elif progress < SCENE_2_END:
            # Scene 2: Hold at top view while continuing rotation
            scene_progress = (progress - SCENE_1_END) / (SCENE_2_END - SCENE_1_END)
            
            elevation = 90  # Hold at top view
            zoom_factor = 3 - (scene_progress * 0.5)  # Slight zoom adjustment during hold
            
        elif progress < SCENE_3_END:
            # Scene 3: Move to and hold side view
            scene_progress = (progress - SCENE_2_END) / (SCENE_3_END - SCENE_2_END)
            
            if scene_progress < 0.3:  # Quick transition to side view
                transition_progress = scene_progress / 0.3
                eased_transition = self.ease_in_out_cubic(transition_progress)
                elevation = 90 - (90 * eased_transition)  # Move from top (90°) to side (0°)
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
        if (progress > SCENE_1_END - 0.05 and progress < SCENE_1_END) or \
           (progress > SCENE_2_END - 0.05 and progress < SCENE_2_END) or \
           (progress > SCENE_3_END - 0.05 and progress < SCENE_3_END):
            glitch_progress = min(1.0, (0.05 - abs(progress - int(progress / 0.3) * 0.3)) * 20)
            glitch_intensity = np.sin(current_frame * 0.5) * glitch_progress
            elevation += glitch_intensity * 2
            
        # Update camera parameters based on camera mode
        if is_orthographic:
            self.view.camera.scale_factor = current_state["scale_factor"] * (1 / zoom_factor)
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

        self._schedule_next_frame(self._matrix_effect_slow_step, next_params, current_frame)

    def orbit_flyby(self):
        """
        Orbit flyby animation - camera flies around the network in an elliptical orbit,
        changing elevation to create a dynamic flyby effect.

        Works in both orthographic mode (adjusts scale_factor) and
        non-orthographic mode (adjusts distance).
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
            "total_frames": 60,  # Reduced to 60 frames
            "orbit_cycles": 1.0,  # Exactly one full orbit
            "elevation_amplitude": 30,  # Maximum elevation change
            "distance_factor": 1.2,  # Increase distance slightly for better view
            "current_frame": 0,
            "restore_original": True,  # Return to original position at end
        }

        # Start the animation
        self._animation_timer = app.Timer(
            interval=1 / 60,
            connect=lambda _: self._orbit_flyby_step(**params),
            iterations=1,
        )
        self._animation_timer.start()

    def _orbit_flyby_step(
        self,
        current_state,
        is_orthographic,
        total_frames,
        orbit_cycles,
        elevation_amplitude,
        distance_factor,
        current_frame,
        **kwargs,
    ):
        """Execute a single step of the orbit flyby animation"""
        if current_frame >= total_frames:
            # Animation complete, restore original state exactly
            self._restore_camera_state(current_state)
            self._animation_in_progress = False
            return

        # Calculate progress (0 to 1)
        progress = current_frame / total_frames

        # Calculate azimuth change (full orbits)
        # Use sine easing to start and end smoothly at the original position
        if progress < 0.7:  # Changed from 0.9 to 0.7 to give more time for ending
            # First 70% - complete the orbit
            orbit_progress = progress / 0.7
            # Add ease-out to make it smoother
            eased_orbit = self.ease_out_cubic(orbit_progress)
            azimuth_change = 360 * orbit_cycles * eased_orbit
        else:
            # Last 30% - return to original azimuth more gradually
            final_segment = (progress - 0.7) / 0.3
            # Use ease-in-out for smoother return
            eased_final = self.ease_in_out_cubic(final_segment)
            azimuth_change = 360 * orbit_cycles * (1 - eased_final)

        # Calculate elevation using a sine wave for up-down motion
        # Ensure we start and end at 0 elevation change
        elevation_offset = elevation_amplitude * np.sin(progress * np.pi * 2)

        # Ensure elevation returns to original more gradually
        if progress > 0.7:
            final_segment = (progress - 0.7) / 0.3
            eased_final = self.ease_in_out_cubic(final_segment)
            elevation_offset = elevation_offset * (1 - eased_final)

        # Calculate distance/scale variation (closer and further) using a sine wave
        variation_factor = 0.2 * np.sin(progress * np.pi * 3)

        # Ensure variation returns to 0 more gradually
        if progress > 0.7:
            final_segment = (progress - 0.7) / 0.3
            eased_final = self.ease_in_out_cubic(final_segment)
            variation_factor = variation_factor * (1 - eased_final)

        # Update camera parameters based on camera mode
        if is_orthographic:
            # For orthographic mode, adjust scale_factor
            # For scale_factor, LARGER values zoom in, SMALLER values zoom out
            zoom_factor = distance_factor + variation_factor

            # Ensure we return to original scale gradually
            if progress > 0.7:
                final_segment = (progress - 0.7) / 0.3
                eased_final = self.ease_in_out_cubic(final_segment)
                zoom_factor = zoom_factor * (1 - eased_final) + 1 * eased_final

            self.view.camera.scale_factor = current_state["scale_factor"] / zoom_factor
        else:
            # For perspective mode, adjust distance
            # For distance, larger values zoom out
            distance_change = (
                current_state["distance"] * distance_factor - current_state["distance"]
            ) + (current_state["distance"] * variation_factor)

            # Ensure we return to original distance gradually
            if progress > 0.7:
                final_segment = (progress - 0.7) / 0.3
                eased_final = self.ease_in_out_cubic(final_segment)
                distance_change = distance_change * (1 - eased_final)

            self.view.camera.distance = current_state["distance"] + distance_change

        self.view.camera.azimuth = current_state["azimuth"] + azimuth_change
        self.view.camera.elevation = current_state["elevation"] + elevation_offset

        # Update the view
        self.view.canvas.update()

        # Schedule next frame with a clean set of parameters
        next_params = {
            "current_state": current_state,
            "is_orthographic": is_orthographic,
            "total_frames": total_frames,
            "orbit_cycles": orbit_cycles,
            "elevation_amplitude": elevation_amplitude,
            "distance_factor": distance_factor,
            "current_frame": current_frame,
        }
        next_params.update(kwargs)

        self._schedule_next_frame(self._orbit_flyby_step, next_params, current_frame)

    def spiral_dive(self):
        """
        Spiral dive animation - camera spirals down into the network,
        creating a tornado-like diving effect.

        Works in both orthographic mode (adjusts scale_factor) and
        non-orthographic mode (adjusts distance).
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
            "spiral_rotations": 1,  # Number of spiral rotations
            "current_frame": 0,
            "restore_original": True,  # Return to original position at end
        }

        # Start the animation
        self._animation_timer = app.Timer(
            interval=1 / 60,
            connect=lambda _: self._spiral_dive_step(**params),
            iterations=1,
        )
        self._animation_timer.start()

    def _spiral_dive_step(
        self,
        current_state,
        is_orthographic,
        total_frames,
        spiral_rotations,
        current_frame,
        **kwargs,
    ):
        """Execute a single step of the spiral dive animation"""
        if current_frame >= total_frames:
            # Animation complete, restore original state exactly
            self._restore_camera_state(current_state)
            self._animation_in_progress = False
            return

        # Calculate progress (0 to 1)
        progress = current_frame / total_frames

        # Use elastic easing for a bouncy finish
        if progress < 0.65:
            segment_progress = progress / 0.65
            eased_progress = self.ease_out_elastic(segment_progress)
        else:
            # Last 15% - return to original position
            segment_progress = (progress - 0.45) / 0.55
            eased_progress = 1.0 - segment_progress  # Linear return to start

        # Calculate zoom factor
        # Start zoomed out (5x), end at current zoom level
        zoom_factor = 2 * (1 - eased_progress) + 1 * eased_progress

        # Calculate spiral rotation
        if progress < 0.35:
            # First 85% - complete the spiral
            rotation_progress = progress / 0.65
            azimuth_change = 360 * spiral_rotations * rotation_progress
        else:
            # Last 15% - return to original azimuth
            segment_progress = (progress - 0.65) / 0.35
            azimuth_change = 360 * spiral_rotations * (1 - segment_progress)

        # Calculate elevation change (start high, spiral down)
        start_elevation = 60  # Start looking from above (less extreme)

        if progress < 0.85:
            # First 85% - spiral down
            elevation_current = (
                start_elevation
                - (start_elevation - current_state["elevation"]) * eased_progress
            )
        else:
            # Last 15% - return to original elevation
            segment_progress = (progress - 0.85) / 0.15
            elevation_offset = (start_elevation - current_state["elevation"]) * (
                1 - segment_progress
            )
            elevation_current = current_state["elevation"] + elevation_offset

        # Update camera parameters based on camera mode
        if is_orthographic:
            # For orthographic mode, adjust scale_factor
            # For scale_factor, LARGER values zoom in, SMALLER values zoom out
            self.view.camera.scale_factor = current_state["scale_factor"] * (
                1 / zoom_factor
            )
        else:
            # For perspective mode, adjust distance
            # For distance, larger values zoom out
            self.view.camera.distance = current_state["distance"] * zoom_factor

        self.view.camera.azimuth = current_state["azimuth"] + azimuth_change
        self.view.camera.elevation = elevation_current

        # Update the view
        self.view.canvas.update()

        # Schedule next frame with a clean set of parameters
        next_params = {
            "current_state": current_state,
            "is_orthographic": is_orthographic,
            "total_frames": total_frames,
            "spiral_rotations": spiral_rotations,
            "current_frame": current_frame,
        }
        next_params.update(kwargs)

        self._schedule_next_frame(self._spiral_dive_step, next_params, current_frame)

    def bounce_zoom(self):
        """
        Bounce zoom animation - camera zooms in with a bouncy effect,
        as if it's bouncing off an elastic surface.

        Works in both orthographic mode (adjusts scale_factor) and
        non-orthographic mode (adjusts distance).
        """
        if self._animation_in_progress:
            return

        self._animation_in_progress = True
        current_state = self._store_camera_state()

        is_orthographic = self.view.camera.fov == 0

        params = {
            "current_state": current_state,
            "is_orthographic": is_orthographic,
            "total_frames": 60,
            "rotation_amount": 360,
            "current_frame": 0,
            "restore_original": True,
        }

        self._animation_timer = app.Timer(
            interval=1 / 60,
            connect=lambda _: self._bounce_zoom_step(**params),
            iterations=1,
        )
        self._animation_timer.start()

    def _bounce_zoom_step(
        self,
        current_state,
        is_orthographic,
        total_frames,
        rotation_amount,
        current_frame,
        **kwargs,
    ):
        """Execute a single step of the bounce zoom animation"""
        if current_frame >= total_frames:
            # Animation complete, restore original state exactly
            self._restore_camera_state(current_state)
            self._animation_in_progress = False
            return

        # Calculate progress (0 to 1)
        progress = current_frame / total_frames

        # Use bounce easing for a bouncy finish
        if progress < 0.85:
            # First 85% - bounce zoom with bounce easing
            segment_progress = progress / 0.85
            # Reduce frequency of bounce by adjusting the progress
            bounce_progress = segment_progress * 0.7  # Reduce bounce frequency
            eased_progress = self.ease_out_bounce(bounce_progress)
            # Reduce the zoom range to be less extreme
            zoom_factor = 2.5 * (1 - eased_progress) + 1 * eased_progress  # Reduced from 3.5 to 2.5
        else:
            # Last 15% - return to original position smoothly
            segment_progress = (progress - 0.85) / 0.15
            eased_progress = self.ease_in_out_cubic(segment_progress)
            zoom_factor = 1  # Stay at original zoom level during return

        rotation_progress = progress 
        eased_rotation = self.ease_in_out_cubic(rotation_progress)
        azimuth_change = rotation_amount * eased_rotation

        # Add some bounce to elevation with reduced amplitude
        if progress < 0.85:
            # First 85% - bounce elevation
            elevation_bounce = 10 * np.sin(progress * np.pi * 3)  # Reduced amplitude
        else:
            # Last 15% - return to original elevation smoothly
            segment_progress = (progress - 0.85) / 0.15
            eased_progress = self.ease_in_out_cubic(segment_progress)
            elevation_bounce = 10 * np.sin(progress * np.pi * 3) * (1 - eased_progress)

        # Update camera parameters based on camera mode
        if is_orthographic:
            # For orthographic mode, adjust scale_factor
            # For scale_factor, LARGER values zoom in, SMALLER values zoom out
            self.view.camera.scale_factor = current_state["scale_factor"] * (
                1 / zoom_factor
            )
        else:
            # For perspective mode, adjust distance
            # For distance, larger values zoom out
            self.view.camera.distance = current_state["distance"] * zoom_factor

        self.view.camera.azimuth = current_state["azimuth"] + azimuth_change
        self.view.camera.elevation = current_state["elevation"] + elevation_bounce

        # Update the view
        self.view.canvas.update()

        # Schedule next frame with a clean set of parameters
        next_params = {
            "current_state": current_state,
            "is_orthographic": is_orthographic,
            "total_frames": total_frames,
            "rotation_amount": rotation_amount,
            "current_frame": current_frame,
        }
        next_params.update(kwargs)

        self._schedule_next_frame(self._bounce_zoom_step, next_params, current_frame)

    def swing_around(self):
        """
        Swing around animation - camera swings around the network in an arc,
        creating a dynamic perspective change.

        Works in both orthographic mode (adjusts scale_factor) and
        non-orthographic mode (adjusts distance).
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
            connect=lambda _: self._swing_around_step(**params),
            iterations=1,
        )
        self._animation_timer.start()

    def _swing_around_step(
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
            azimuth_offset = swing_angle * angle_progress * 2  # Multiply by 0.5 to center the swing

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
            zoom_variation = 1# + (0.999 * (1 - eased_final))

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

        self._schedule_next_frame(self._swing_around_step, next_params, current_frame)

    def pulse_zoom(self):
        """
        Pulse zoom animation - camera pulses in and out with a rhythmic motion,
        creating a breathing effect.

        Works in both orthographic mode (adjusts scale_factor) and
        non-orthographic mode (adjusts distance).
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
            "total_frames": 60,  # Reduced from 90 to 60 frames
            "pulse_cycles": 1,  # Single pulse cycle
            "rotation_amount": 360,  # Full rotation (changed from 90)
            "current_frame": 0,
            "restore_original": True,  # Return to original position at end
        }

        # Start the animation
        self._animation_timer = app.Timer(
            interval=1 / 60,
            connect=lambda _: self._pulse_zoom_step(**params),
            iterations=1,
        )
        self._animation_timer.start()

    def _pulse_zoom_step(
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
            azimuth_change = prev_azimuth + (rotation_amount - prev_azimuth) * eased_final
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

        self._schedule_next_frame(self._pulse_zoom_step, next_params, current_frame)
