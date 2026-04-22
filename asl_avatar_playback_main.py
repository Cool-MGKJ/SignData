"""
Main Application for ASL Avatar Playback System – Polished Edition

Reconstructs and animates a 3D avatar from gesture dataset with smooth
interpolation, looping playback, HUD overlay, and premium visuals.
"""

import sys
import time
import math
import copy

try:
    import glfw
    GLFW_AVAILABLE = True
except ImportError:
    GLFW_AVAILABLE = False
    print("Warning: GLFW not available. Install with: pip install glfw")

try:
    import OpenGL.GL as gl
    import OpenGL.GLU as glu
    import OpenGL.GLUT as glut
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: OpenGL not available. Install with: pip install PyOpenGL")

from asl_avatar_loader import GestureLoader, BodyPose, FaceExpression, FrameData
from asl_avatar_text_parser import TextParser
from asl_avatar_renderer import AvatarRenderer
from asl_avatar_player import GesturePlayer


# ---------------------------------------------------------------------------
# Frame interpolation
# ---------------------------------------------------------------------------
def _lerp(a, b, t):
    return a + (b - a) * t


def _lerp_joints(j1, j2, t):
    """Linearly interpolate two lists of [x,y,z] joints."""
    out = []
    for a, b in zip(j1, j2):
        if len(a) >= 3 and len(b) >= 3:
            out.append([_lerp(a[0], b[0], t),
                        _lerp(a[1], b[1], t),
                        _lerp(a[2], b[2], t)])
        else:
            out.append(list(a))
    return out


def _lerp_blendshapes(bs1, bs2, t):
    """Interpolate two blendshape dicts."""
    out = {}
    keys = set(list(bs1.keys()) + list(bs2.keys()))
    for k in keys:
        out[k] = _lerp(bs1.get(k, 0.0), bs2.get(k, 0.0), t)
    return out


def interpolate_frames(f1: FrameData, f2: FrameData, t: float) -> FrameData:
    """Create an interpolated frame between f1 and f2 at parameter t in [0,1]."""
    body = BodyPose(
        joints=_lerp_joints(f1.body_pose.joints, f2.body_pose.joints, t),
        hierarchy=list(f1.body_pose.hierarchy),
    )
    left  = _lerp_joints(f1.left_hand, f2.left_hand, t)
    right = _lerp_joints(f1.right_hand, f2.right_hand, t)

    bs1 = f1.face.blendshapes if f1.face else {}
    bs2 = f2.face.blendshapes if f2.face else {}
    face = FaceExpression(blendshapes=_lerp_blendshapes(bs1, bs2, t))

    ts = _lerp(f1.timestamp, f2.timestamp, t)
    return FrameData(timestamp=ts, body_pose=body, left_hand=left,
                     right_hand=right, face=face)


# ---------------------------------------------------------------------------
# HUD helpers (bitmap font via GLUT or plain quads)
# ---------------------------------------------------------------------------
_glut_inited = False

def _ensure_glut():
    global _glut_inited
    if not _glut_inited:
        try:
            glut.glutInit(sys.argv)
            _glut_inited = True
        except Exception:
            _glut_inited = False


def draw_text_2d(x, y, text, color=(0.85, 0.85, 0.9), font=None):
    """Draw text at window pixel position (x, y) using GLUT bitmap font."""
    _ensure_glut()
    if not _glut_inited:
        return
    gl.glColor3f(*color)
    gl.glRasterPos2f(x, y)
    if font is None:
        font = glut.GLUT_BITMAP_HELVETICA_18
    for ch in text:
        glut.glutBitmapCharacter(font, ord(ch))


def draw_progress_bar(x, y, w, h, progress, bar_color=(0.3, 0.7, 0.9),
                      bg_color=(0.2, 0.2, 0.25)):
    """Draw a simple progress bar at pixel position."""
    # Background
    gl.glColor4f(*bg_color, 0.6)
    gl.glBegin(gl.GL_QUADS)
    gl.glVertex2f(x, y)
    gl.glVertex2f(x + w, y)
    gl.glVertex2f(x + w, y + h)
    gl.glVertex2f(x, y + h)
    gl.glEnd()

    # Fill
    fw = w * max(0.0, min(1.0, progress))
    if fw > 0:
        gl.glColor4f(*bar_color, 0.85)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(x, y)
        gl.glVertex2f(x + fw, y)
        gl.glVertex2f(x + fw, y + h)
        gl.glVertex2f(x, y + h)
        gl.glEnd()

    # Border
    gl.glColor4f(0.5, 0.5, 0.55, 0.5)
    gl.glLineWidth(1.0)
    gl.glBegin(gl.GL_LINE_LOOP)
    gl.glVertex2f(x, y)
    gl.glVertex2f(x + w, y)
    gl.glVertex2f(x + w, y + h)
    gl.glVertex2f(x, y + h)
    gl.glEnd()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class PlaybackApp:
    """Main playback application with polished visuals."""

    def __init__(self, dataset_dir: str = "asl_avatar_dataset"):
        if not GLFW_AVAILABLE or not OPENGL_AVAILABLE:
            print("Error: Required libraries not available.")
            print("Install with: pip install glfw PyOpenGL PyOpenGL-accelerate")
            sys.exit(1)

        if not glfw.init():
            print("Error: Failed to initialize GLFW")
            sys.exit(1)

        # MSAA 4x for smooth edges
        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

        self.width, self.height = 1280, 720
        self.window = glfw.create_window(self.width, self.height,
                                         "ASL Avatar Playback", None, None)
        if not self.window:
            glfw.terminate()
            print("Error: Failed to create window")
            sys.exit(1)

        glfw.make_context_current(self.window)
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_framebuffer_size_callback(self.window, self.resize_callback)

        # Components
        self.gesture_loader = GestureLoader(dataset_dir)
        self.text_parser = TextParser(self.gesture_loader)
        self.renderer = AvatarRenderer(show_labels=False)
        self.player = GesturePlayer(self.renderer, self.text_parser)

        # Playback state
        self.input_text = ""
        self.playback_speed = 1.0
        self.loop = True
        self.paused = False

        # For smooth interpolation we drive frames ourselves
        self._sequences = []
        self._seq_idx = 0
        self._play_start = 0.0
        self._seq_start = 0.0
        self._is_active = False
        self._current_label = ""
        self._current_frame = 0
        self._total_frames = 0

        # Camera orbit
        self._cam_angle = 0.0      # horizontal angle in degrees (0 = facing out)
        self._cam_pitch = 3.0      # vertical angle (low for straight-on view)
        self._cam_dist  = 4.8      # zoomed out a touch
        self._auto_rotate = False

        self.setup_opengl()

        print("\n+======================================+")
        print("|    ASL Avatar Playback System        |")
        print("+======================================+")
        gestures = self.gesture_loader.list_available_gestures()
        print(f"|  Gestures: {', '.join(gestures):24s} |")
        print("+--------------------------------------+")
        print("|  SPACE   Play / Pause                |")
        print("|  +/-     Speed up / down              |")
        print("|  L       Toggle loop                  |")
        print("|  R       Reset                        |")
        print("|  Left/Right  Rotate camera            |")
        print("|  O       Toggle auto-rotate           |")
        print("|  ESC     Exit                         |")
        print("+======================================+")

    # ------------------------------------------------------------------
    def setup_opengl(self):
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glEnable(gl.GL_NORMALIZE)
        self._update_projection()

    def _update_projection(self):
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = self.width / max(self.height, 1)
        glu.gluPerspective(42.0, aspect, 0.1, 100.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def resize_callback(self, window, w, h):
        self.width, self.height = max(w, 1), max(h, 1)
        gl.glViewport(0, 0, self.width, self.height)
        self._update_projection()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def key_callback(self, window, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            if self._is_active:
                self.paused = not self.paused
                if not self.paused:
                    # adjust start time so position stays
                    self._seq_start = time.time() - self._elapsed_when_paused
            else:
                # restart
                self._start_sequences()
        elif key == glfw.KEY_EQUAL or key == glfw.KEY_KP_ADD:  # +
            self.playback_speed = min(4.0, self.playback_speed + 0.25)
            print(f"  Speed: {self.playback_speed:.2f}x")
        elif key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT:  # -
            self.playback_speed = max(0.25, self.playback_speed - 0.25)
            print(f"  Speed: {self.playback_speed:.2f}x")
        elif key == glfw.KEY_L:
            self.loop = not self.loop
            print(f"  Loop: {'ON' if self.loop else 'OFF'}")
        elif key == glfw.KEY_R:
            self._start_sequences()
        elif key == glfw.KEY_LEFT:
            self._cam_angle += 5.0
        elif key == glfw.KEY_RIGHT:
            self._cam_angle -= 5.0
        elif key == glfw.KEY_O:
            self._auto_rotate = not self._auto_rotate

    # ------------------------------------------------------------------
    # Playback engine (with interpolation)
    # ------------------------------------------------------------------
    def load_text(self, text: str) -> bool:
        parsed = self.text_parser.parse_text(text)
        seqs = []
        for word, gesture in parsed:
            if gesture is not None:
                seqs.append(gesture)
            else:
                print(f"  Warning: no gesture for '{word}'")
        if seqs:
            self._sequences = seqs
            self.input_text = text
            return True
        return False

    def _start_sequences(self):
        if not self._sequences:
            return
        self._seq_idx = 0
        self._is_active = True
        self.paused = False
        self._seq_start = time.time()
        self._play_start = self._seq_start
        self._elapsed_when_paused = 0.0

    def _get_current_frame(self) -> FrameData:
        """Return the (interpolated) frame for the current time."""
        if not self._sequences or self._seq_idx >= len(self._sequences):
            return None

        seq = self._sequences[self._seq_idx]
        frames = seq.frames
        if not frames:
            return None

        if self.paused:
            elapsed = self._elapsed_when_paused
        else:
            elapsed = (time.time() - self._seq_start) * self.playback_speed
            self._elapsed_when_paused = elapsed

        # Determine the two bounding keyframes
        t0 = frames[0].timestamp
        tN = frames[-1].timestamp
        duration = tN - t0
        if duration <= 0:
            duration = len(frames) / max(seq.fps, 1)

        # Map elapsed to normalised position
        if duration > 0:
            frac = elapsed / duration
        else:
            frac = 0.0

        # If past end of this sequence, advance
        if frac >= 1.0:
            self._seq_idx += 1
            if self._seq_idx >= len(self._sequences):
                if self.loop:
                    self._seq_idx = 0
                else:
                    self._is_active = False
                    self._seq_idx = len(self._sequences) - 1
                    # return last frame of last seq
                    last_seq = self._sequences[-1]
                    self._current_frame = len(last_seq.frames) - 1
                    self._total_frames = len(last_seq.frames)
                    self._current_label = last_seq.label
                    return last_seq.frames[-1] if last_seq.frames else None
            self._seq_start = time.time()
            self._elapsed_when_paused = 0.0
            return self._get_current_frame()  # recurse into next seq

        # Find bounding keyframes
        target_time = t0 + frac * duration
        idx_a = 0
        for i in range(len(frames) - 1):
            if frames[i + 1].timestamp >= target_time:
                idx_a = i
                break
        else:
            idx_a = len(frames) - 2 if len(frames) > 1 else 0

        idx_b = min(idx_a + 1, len(frames) - 1)

        # Local interpolation parameter between frame A and B
        ta = frames[idx_a].timestamp
        tb = frames[idx_b].timestamp
        seg_dur = tb - ta
        if seg_dur > 0:
            local_t = (target_time - ta) / seg_dur
            local_t = max(0.0, min(1.0, local_t))
        else:
            local_t = 0.0

        self._current_label = seq.label
        self._current_frame = idx_a
        self._total_frames = len(frames)

        if idx_a == idx_b:
            return frames[idx_a]
        return interpolate_frames(frames[idx_a], frames[idx_b], local_t)

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------
    def _render_hud(self):
        """Render 2D overlay: label, frame info, progress bar, controls."""
        gl.glPushAttrib(gl.GL_ALL_ATTRIB_BITS)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glOrtho(0, self.width, 0, self.height, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity()

        # Gesture label (top-center)
        if self._current_label:
            label = self._current_label.upper()
            # Estimate text width (~10px per char for Helvetica 18)
            tw = len(label) * 10
            draw_text_2d(self.width / 2 - tw / 2, self.height - 35,
                         label, (0.9, 0.85, 0.7))

        # Frame counter (bottom-left)
        if self._total_frames > 0:
            info = f"Frame {self._current_frame + 1}/{self._total_frames}"
            draw_text_2d(15, 18, info, (0.6, 0.6, 0.65))

        # Speed (bottom-left, second line)
        spd = f"Speed: {self.playback_speed:.2f}x"
        draw_text_2d(15, 40, spd, (0.5, 0.5, 0.55))

        # Loop indicator
        loop_txt = "[LOOP]" if self.loop else ""
        if loop_txt:
            draw_text_2d(15, 62, loop_txt, (0.3, 0.7, 0.5))

        # Paused indicator
        if self.paused:
            pw = 4 * 10
            draw_text_2d(self.width / 2 - pw / 2, self.height / 2,
                         "PAUSED", (1.0, 1.0, 1.0))

        # Progress bar (bottom-center)
        bar_w = 300
        bar_h = 6
        bar_x = (self.width - bar_w) / 2
        bar_y = 15
        progress = 0.0
        if self._total_frames > 1:
            progress = self._current_frame / (self._total_frames - 1)
        draw_progress_bar(bar_x, bar_y, bar_w, bar_h, progress)

        # Controls hint (bottom-right)
        hint = "SPACE:play  +/-:speed  L:loop  R:reset  ESC:exit"
        hw = len(hint) * 7
        draw_text_2d(self.width - hw - 10, 18, hint, (0.4, 0.4, 0.45),
                     font=glut.GLUT_BITMAP_HELVETICA_12 if _glut_inited else None)

        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPopAttrib()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        target_dt = 1.0 / 60.0

        while not glfw.window_should_close(self.window):
            frame_start = time.time()

            # ---- Clear ----
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            # ---- Gradient background ----
            self.renderer.render_gradient_background()

            # ---- 3D scene ----
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()

            # Camera orbit
            if self._auto_rotate:
                self._cam_angle += 0.15

            rad = math.radians(self._cam_angle)
            pitch = math.radians(self._cam_pitch)
            eye_x = self._cam_dist * math.sin(rad) * math.cos(pitch)
            eye_y = self._cam_dist * math.sin(pitch) + 0.2
            eye_z = self._cam_dist * math.cos(rad) * math.cos(pitch)
            glu.gluLookAt(eye_x, eye_y, eye_z,
                          0.0, 0.0, 0.0,
                          0.0, 1.0, 0.0)

            # Floor grid
            self.renderer.render_ground_grid()

            # Avatar
            if self._is_active:
                frame = self._get_current_frame()
                if frame:
                    self.renderer._current_frame_num = self._current_frame
                    self.renderer.render_frame(
                        frame.body_pose,
                        frame.left_hand,
                        frame.right_hand,
                        frame.face,
                    )
            else:
                # Idle – show first frame if available
                if self._sequences:
                    seq = self._sequences[0]
                    if seq.frames:
                        f0 = seq.frames[0]
                        self._current_label = seq.label
                        self._current_frame = 0
                        self._total_frames = len(seq.frames)
                        self.renderer.render_frame(
                            f0.body_pose, f0.left_hand,
                            f0.right_hand, f0.face,
                        )

            # ---- 2D HUD ----
            self._render_hud()

            # ---- Swap ----
            glfw.swap_buffers(self.window)
            glfw.poll_events()

            # Frame-rate limit
            elapsed = time.time() - frame_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

        glfw.terminate()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="ASL Avatar Playback System")
    parser.add_argument("--dataset", "-d", default="asl_avatar_dataset",
                        help="Dataset directory (default: asl_avatar_dataset)")
    parser.add_argument("--text", "-t", default=None,
                        help="Text / gesture to play immediately")
    parser.add_argument("--speed", "-s", type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--loop", action="store_true", default=True,
                        help="Loop playback (default: on)")
    parser.add_argument("--no-loop", dest="loop", action="store_false",
                        help="Disable looping")

    args = parser.parse_args()

    app = PlaybackApp(args.dataset)
    app.playback_speed = args.speed
    app.loop = args.loop

    if args.text:
        if app.load_text(args.text):
            app._start_sequences()
        else:
            print(f"Error: could not load gestures for '{args.text}'")

    app.run()


if __name__ == "__main__":
    main()
