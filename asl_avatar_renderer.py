"""
3D Avatar Renderer – Polished Edition

Renders a volumetric 3D human avatar with capsule limbs, procedural face,
Phong lighting, gradient background and floor grid.
"""

import math
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

try:
    import OpenGL.GL as gl
    import OpenGL.GLU as glu
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: OpenGL not available. Install with: pip install PyOpenGL")


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COL_BODY_BONE   = (0.55, 0.55, 0.62)
COL_BODY_JOINT  = (0.82, 0.82, 0.86)
COL_LEFT_HAND   = (0.20, 0.72, 0.72)
COL_RIGHT_HAND  = (0.90, 0.45, 0.35)
COL_HAND_JOINT  = (0.92, 0.92, 0.95)
COL_HEAD_SKIN   = (0.82, 0.70, 0.60)
COL_EYE_WHITE   = (0.95, 0.95, 0.97)
COL_EYE_IRIS    = (0.18, 0.22, 0.32)
COL_MOUTH       = (0.75, 0.30, 0.30)
COL_BROW        = (0.35, 0.28, 0.22)

BG_TOP    = (0.06, 0.07, 0.12)
BG_BOTTOM = (0.12, 0.13, 0.20)
COL_GRID  = (0.25, 0.28, 0.35, 0.35)


@dataclass
class Joint:
    """3D joint position."""
    x: float
    y: float
    z: float


class AvatarRenderer:
    """Renders a polished 3D avatar from gesture data."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, show_labels=False, debug_print=False):
        self.scale = 3.0
        self.show_labels = show_labels
        self.debug_print = debug_print
        self._last_debug_frame = -1

        # Geometry sizes
        self._body_bone_radius  = 0.028
        self._body_joint_radius = 0.042
        self._hand_bone_radius  = 0.012
        self._hand_joint_radius = 0.018
        self._head_radius       = 0.18

        # Cached quadric
        self._quadric: Optional[object] = None

        # Body skeleton connections (parent -> child)
        self.body_connections = [
            (0, 1), (1, 2), (2, 3),      # spine
            (2, 4), (4, 5), (5, 6),       # left arm
            (2, 7), (7, 8), (8, 9),       # right arm
        ]

        # Hand connections (MediaPipe 21-landmark)
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),         # index
            (0, 9), (9, 10), (10, 11), (11, 12),    # middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
            (5, 9), (9, 13), (13, 17),               # palm cross
        ]

        self.body_joint_names = [
            "pelvis", "spine", "neck", "head",
            "left_shoulder", "left_elbow", "left_wrist",
            "right_shoulder", "right_elbow", "right_wrist",
        ]

    # ------------------------------------------------------------------
    # Low-level GL helpers
    # ------------------------------------------------------------------
    def _get_quadric(self):
        if self._quadric is None:
            self._quadric = glu.gluNewQuadric()
            glu.gluQuadricNormals(self._quadric, glu.GLU_SMOOTH)
        return self._quadric

    def _set_material(self, color: Tuple[float, float, float], shininess: float = 40.0):
        """Set OpenGL material for Phong shading."""
        ambient  = [c * 0.25 for c in color] + [1.0]
        diffuse  = [c for c in color] + [1.0]
        specular = [min(1.0, c + 0.3) for c in color] + [1.0]
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT, ambient)
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_DIFFUSE, diffuse)
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SPECULAR, specular)
        gl.glMaterialf(gl.GL_FRONT_AND_BACK, gl.GL_SHININESS, shininess)

    # ------------------------------------------------------------------
    # Drawing primitives
    # ------------------------------------------------------------------
    def draw_sphere(self, x, y, z, radius, color, slices=20, stacks=20):
        if not OPENGL_AVAILABLE:
            return
        gl.glPushMatrix()
        gl.glTranslatef(x, y, z)
        self._set_material(color)
        glu.gluSphere(self._get_quadric(), radius, slices, stacks)
        gl.glPopMatrix()

    def draw_capsule(self, start, end, radius, color):
        """Draw a cylinder capped with hemispheres (a capsule)."""
        if not OPENGL_AVAILABLE:
            return
        sx, sy, sz = start
        ex, ey, ez = end
        dx, dy, dz = ex - sx, ey - sy, ez - sz
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length < 1e-6:
            self.draw_sphere(sx, sy, sz, radius, color)
            return

        self._set_material(color)
        q = self._get_quadric()

        gl.glPushMatrix()
        gl.glTranslatef(sx, sy, sz)

        # Align Z-axis of the cylinder to the direction vector
        # Default gluCylinder is along +Z
        ax, ay, az = self._rotation_to_direction(dx, dy, dz, length)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, dz / length))))
        if abs(ax) > 1e-8 or abs(ay) > 1e-8 or abs(az) > 1e-8:
            gl.glRotatef(angle, ax, ay, az)
        elif dz < 0:
            gl.glRotatef(180, 1, 0, 0)

        # Bottom hemisphere
        glu.gluSphere(q, radius, 14, 14)
        # Cylinder
        glu.gluCylinder(q, radius, radius, length, 14, 1)
        # Top hemisphere
        gl.glTranslatef(0, 0, length)
        glu.gluSphere(q, radius, 14, 14)

        gl.glPopMatrix()

    @staticmethod
    def _rotation_to_direction(dx, dy, dz, length):
        """Return rotation axis to rotate Z-axis onto the given direction."""
        # Cross product of (0,0,1) x (dx,dy,dz)/length
        nx = -dy
        ny =  dx
        nz =  0.0
        nl = math.sqrt(nx*nx + ny*ny + nz*nz)
        if nl < 1e-8:
            return (1.0, 0.0, 0.0)
        return (nx / nl, ny / nl, nz / nl)

    # ------------------------------------------------------------------
    # Background & grid
    # ------------------------------------------------------------------
    def render_gradient_background(self):
        """Draw a full-screen vertical gradient quad (no depth)."""
        if not OPENGL_AVAILABLE:
            return
        gl.glPushAttrib(gl.GL_ALL_ATTRIB_BITS)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glOrtho(-1, 1, -1, 1, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity()

        gl.glBegin(gl.GL_QUADS)
        gl.glColor3f(*BG_BOTTOM)
        gl.glVertex2f(-1, -1)
        gl.glVertex2f(1, -1)
        gl.glColor3f(*BG_TOP)
        gl.glVertex2f(1, 1)
        gl.glVertex2f(-1, 1)
        gl.glEnd()

        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPopAttrib()

    def render_ground_grid(self, y=-1.35, extent=3.0, step=0.25):
        """Draw a subtle translucent floor grid."""
        if not OPENGL_AVAILABLE:
            return
        gl.glPushAttrib(gl.GL_ALL_ATTRIB_BITS)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glLineWidth(1.0)

        gl.glBegin(gl.GL_LINES)
        n = int(extent / step)
        for i in range(-n, n + 1):
            # Fade lines near the edges
            frac = 1.0 - abs(i * step) / extent
            alpha = COL_GRID[3] * max(frac, 0.0)
            gl.glColor4f(COL_GRID[0], COL_GRID[1], COL_GRID[2], alpha)
            # lines parallel to X
            gl.glVertex3f(-extent, y, i * step)
            gl.glVertex3f(extent, y, i * step)
            # lines parallel to Z
            gl.glVertex3f(i * step, y, -extent)
            gl.glVertex3f(i * step, y, extent)
        gl.glEnd()

        gl.glPopAttrib()

    # ------------------------------------------------------------------
    # Lighting
    # ------------------------------------------------------------------
    def setup_lighting(self):
        """Three-point Phong lighting."""
        if not OPENGL_AVAILABLE:
            return
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glShadeModel(gl.GL_SMOOTH)

        # Key light (warm, from upper-right-front)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [3.0, 5.0, 4.0, 0.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [0.95, 0.92, 0.88, 1.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])

        # Fill light (cool, from left)
        gl.glEnable(gl.GL_LIGHT1)
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, [-4.0, 2.0, 2.0, 0.0])
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_DIFFUSE, [0.25, 0.28, 0.35, 1.0])
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_SPECULAR, [0.1, 0.1, 0.15, 1.0])

        # Rim / back light
        gl.glEnable(gl.GL_LIGHT2)
        gl.glLightfv(gl.GL_LIGHT2, gl.GL_POSITION, [0.0, 3.0, -5.0, 0.0])
        gl.glLightfv(gl.GL_LIGHT2, gl.GL_DIFFUSE, [0.18, 0.18, 0.25, 1.0])
        gl.glLightfv(gl.GL_LIGHT2, gl.GL_SPECULAR, [0.05, 0.05, 0.08, 1.0])

        # Ambient
        gl.glLightModelfv(gl.GL_LIGHT_MODEL_AMBIENT, [0.12, 0.12, 0.15, 1.0])

    # ------------------------------------------------------------------
    # Body pose
    # ------------------------------------------------------------------
    def _body_joints_to_world(self, joints):
        """Convert body joints to world coords centred at origin."""
        valid = [j for j in joints if len(j) >= 3]
        if not valid:
            return [], (0, 0, 0)
        xs = [j[0] for j in valid]
        ys = [j[1] for j in valid]
        zs = [j[2] for j in valid]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        cz = (min(zs) + max(zs)) / 2.0

        world = []
        for j in joints:
            if len(j) >= 3:
                wx =  (j[0] - cx) * self.scale
                wy =  (cy - j[1]) * self.scale   # flip Y (MediaPipe Y goes down)
                wz = -(j[2] - cz) * self.scale   # flip Z so avatar faces camera
                world.append((wx, wy, wz))
            else:
                world.append(None)
        return world, (cx, cy, cz)

    def render_body_pose(self, joints, hierarchy, frame_num=0):
        if not OPENGL_AVAILABLE or not joints:
            return
        if len(joints) != len(hierarchy):
            return

        world, center = self._body_joints_to_world(joints)

        # Debug
        if self.debug_print and frame_num != self._last_debug_frame:
            print(f"\n=== Frame {frame_num} ===")
            for i, (j, n) in enumerate(zip(joints, hierarchy)):
                if len(j) >= 3:
                    print(f"  {i}: {n:15s}  x:{j[0]:.3f} y:{j[1]:.3f} z:{j[2]:.3f}")
            self._last_debug_frame = frame_num

        # Draw capsule bones
        for pi, ci in self.body_connections:
            if pi < len(world) and ci < len(world) and world[pi] and world[ci]:
                self.draw_capsule(world[pi], world[ci],
                                  self._body_bone_radius, COL_BODY_BONE)

        # Draw joint spheres
        for i, wp in enumerate(world):
            if wp:
                # Head gets special treatment
                if i == 3:  # head
                    self.draw_sphere(*wp, self._head_radius, COL_HEAD_SKIN, 28, 28)
                else:
                    self.draw_sphere(*wp, self._body_joint_radius, COL_BODY_JOINT, 16, 16)

    # ------------------------------------------------------------------
    # Hands
    # ------------------------------------------------------------------
    def _hand_to_world(self, landmarks, body_wrist_pos):
        """Convert 21 hand landmarks to world space anchored at body wrist.

        Uses relative offsets from the hand wrist landmark (index 0) to each
        other landmark, avoiding the coordinate-system mismatch between
        MediaPipe image-space (0-1) and the body's bounding-box world space.
        """
        if not landmarks or len(landmarks) < 21:
            return []
        wl = landmarks[0]  # hand wrist landmark (reference point)
        if len(wl) < 3:
            return []

        # Scale factor: hand landmarks are 0-1 image coords;
        # relative finger-length ~0.10 should map to ~0.12 world units.
        hs = self.scale * 0.40

        out = []
        for lm in landmarks[:21]:
            if len(lm) >= 3:
                dx = -(lm[0] - wl[0]) * hs          # X: flip to restore chirality after body Z-flip
                dy = -(lm[1] - wl[1]) * hs          # Y: flip (image Y downward)
                dz = -(lm[2] - wl[2]) * hs          # Z: flip (consistent with body Z flip)
                out.append((
                    body_wrist_pos[0] + dx,
                    body_wrist_pos[1] + dy,
                    body_wrist_pos[2] + dz,
                ))
            else:
                out.append(None)
        return out

    def render_hand(self, landmarks, wrist_position, color):
        if not OPENGL_AVAILABLE or not landmarks or len(landmarks) < 21:
            return None
        # Skip all-zero hands
        if not any(any(abs(c) > 0.001 for c in lm) for lm in landmarks if len(lm) >= 3):
            return None

        world = self._hand_to_world(landmarks, wrist_position)
        if not world:
            return None

        # Capsule bones
        for pi, ci in self.hand_connections:
            if pi < len(world) and ci < len(world) and world[pi] and world[ci]:
                self.draw_capsule(world[pi], world[ci],
                                  self._hand_bone_radius, color)
        # Joint spheres
        for wp in world:
            if wp:
                self.draw_sphere(*wp, self._hand_joint_radius, COL_HAND_JOINT, 10, 10)

        return world[0] if world else None

    # ------------------------------------------------------------------
    # Procedural face
    # ------------------------------------------------------------------
    def render_face(self, blendshapes: dict, head_pos: Tuple[float, float, float]):
        """Draw procedural eyes, mouth, and brows on the head sphere."""
        if not OPENGL_AVAILABLE or not blendshapes:
            return
        hx, hy, hz = head_pos
        r = self._head_radius

        smile_l   = blendshapes.get("mouthSmileLeft", 0.0)
        smile_r   = blendshapes.get("mouthSmileRight", 0.0)
        smile     = blendshapes.get("smile", 0.0)
        jaw_open  = blendshapes.get("jawOpen", 0.0)
        blink_l   = blendshapes.get("eyeBlinkLeft", 0.0)
        blink_r   = blendshapes.get("eyeBlinkRight", 0.0)
        brow_up   = blendshapes.get("browRaise", 0.0)
        brow_inner = blendshapes.get("browInnerUp", 0.0)

        # ---------- Eyes ----------
        eye_y  = hy + r * 0.22
        eye_sep = r * 0.32
        eye_z  = hz + r * 0.88   # on front surface (toward camera, +Z faces viewer)

        for side, blink in [(-1, blink_l), (1, blink_r)]:
            ex = hx + side * eye_sep
            # Eye white
            self._set_material(COL_EYE_WHITE, 20)
            gl.glPushMatrix()
            gl.glTranslatef(ex, eye_y, eye_z)
            sy = max(0.15, 1.0 - blink * 0.85)  # squash when blinking
            gl.glScalef(1.0, sy, 0.5)
            glu.gluSphere(self._get_quadric(), r * 0.14, 14, 14)
            gl.glPopMatrix()

            # Iris
            self._set_material(COL_EYE_IRIS, 60)
            gl.glPushMatrix()
            gl.glTranslatef(ex, eye_y, eye_z - r * 0.06)
            gl.glScalef(1.0, sy, 0.5)
            glu.gluSphere(self._get_quadric(), r * 0.07, 10, 10)
            gl.glPopMatrix()

        # ---------- Mouth ----------
        avg_smile = max(smile, (smile_l + smile_r) / 2.0)
        mouth_y = hy - r * 0.32
        mouth_z = hz + r * 0.90   # front face of head sphere
        mouth_open_amt = jaw_open * r * 0.18

        self._set_material(COL_MOUTH, 10)

        # Draw mouth as a small arc of spheres
        n_seg = 8
        mouth_width = r * 0.30
        for i in range(n_seg + 1):
            t = i / n_seg  # 0..1
            ang = math.pi * t  # 0..pi
            mx = hx + math.cos(ang) * mouth_width
            # Smile curves corners up
            smile_lift = avg_smile * 0.04 * (1.0 - abs(2 * t - 1))
            my = mouth_y - math.sin(ang) * mouth_open_amt * 0.5 + smile_lift
            mz = mouth_z
            self.draw_sphere(mx, my, mz, r * 0.032, COL_MOUTH, 6, 6)

        # If mouth is open, draw lower lip arc
        if jaw_open > 0.15:
            for i in range(n_seg + 1):
                t = i / n_seg
                ang = math.pi * t
                mx = hx + math.cos(ang) * mouth_width * 0.85
                my = mouth_y - mouth_open_amt - smile_lift * 0.3
                mz = mouth_z + r * 0.02
                self.draw_sphere(mx, my, mz, r * 0.025, COL_MOUTH, 6, 6)

        # ---------- Brows ----------
        brow_y_base = hy + r * 0.42
        brow_lift = (brow_up + brow_inner) * r * 0.08

        self._set_material(COL_BROW, 10)
        for side in [-1, 1]:
            bx = hx + side * eye_sep
            by = brow_y_base + brow_lift
            bz = eye_z + r * 0.02
            # Small capsule for brow
            b_start = (bx - side * r * 0.12, by, bz)
            b_end   = (bx + side * r * 0.06, by - 0.01, bz)
            self.draw_capsule(b_start, b_end, r * 0.025, COL_BROW)

    # ------------------------------------------------------------------
    # Full frame
    # ------------------------------------------------------------------
    def render_frame(self, body_pose, left_hand, right_hand, face):
        """Render one complete frame of animation (called by player)."""
        if not OPENGL_AVAILABLE:
            return

        # Lighting (re-apply each frame to be safe with matrix changes)
        self.setup_lighting()
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)

        # ---- body ----
        left_wrist_pos  = None
        right_wrist_pos = None
        head_pos        = None

        if body_pose and body_pose.joints and len(body_pose.joints) > 0:
            frame_num = getattr(self, '_current_frame_num', 0)
            self.render_body_pose(body_pose.joints, body_pose.hierarchy, frame_num)

            world, center = self._body_joints_to_world(body_pose.joints)

            if len(world) > 3 and world[3]:
                head_pos = world[3]
            if len(world) > 6 and world[6]:
                left_wrist_pos = world[6]
            if len(world) > 9 and world[9]:
                right_wrist_pos = world[9]

        # ---- face ----
        if head_pos and face and face.blendshapes:
            self.render_face(face.blendshapes, head_pos)

        # ---- hands ----
        if left_wrist_pos and left_hand and len(left_hand) >= 21:
            hw = self.render_hand(left_hand, left_wrist_pos, COL_LEFT_HAND)
            if hw:
                self.draw_capsule(left_wrist_pos, hw,
                                  self._hand_bone_radius, COL_LEFT_HAND)

        if right_wrist_pos and right_hand and len(right_hand) >= 21:
            hw = self.render_hand(right_hand, right_wrist_pos, COL_RIGHT_HAND)
            if hw:
                self.draw_capsule(right_wrist_pos, hw,
                                  self._hand_bone_radius, COL_RIGHT_HAND)
