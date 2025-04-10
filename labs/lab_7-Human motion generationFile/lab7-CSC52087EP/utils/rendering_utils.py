"""
This code is adapted from:
https://github.com/davrempe/humor and https://github.com/nv-tlabs/stmc
"""

from itertools import product
import os
from pathlib import Path
import time
from typing import Callable, List, Tuple

import numpy as np
import imageio
import pyrender
from smplx import SMPL
from smplx.utils import Struct
import torch
from torchtyping import TensorType
import trimesh

# ------------------------------------------------------------------------------------- #

FACES = torch.from_numpy(np.int32(np.load("./utils/smpl.faces")))
VERTEX_COLOR = [0.0390625, 0.4140625, 0.796875]
TRAJ_COLOR = [0.63137255, 0.85098039, 0.60784314]
CAM_COLOR = [0.19215686, 0.63921569, 0.32941176]
MASKED_COLOR = [0.2, 0.2, 0.2]

os.environ["PYOPENGL_PLATFORM"] = "egl"
c2c = lambda tensor: tensor.detach().cpu().numpy()  # noqa

num_frames, num_vertices = None, None

# ------------------------------------------------------------------------------------- #


class SceneRenderer(object):
    def __init__(
        self,
        pyrender: Callable,
        width: int,
        height: int,
        focal: int = 500,
        follow_camera: bool = True,
    ):
        super().__init__()

        self.pyrender = pyrender

        # mesh sequences to animate
        self.animated_seqs = []  # the actual sequence of pyrender meshes
        self.animated_seqs_type = []
        self.animated_nodes = []  # the nodes corresponding to each sequence
        self.light_nodes = []

        # Animation state
        self.animation_len = -1
        self.animation_frame_idx = 0  # current index in the animation sequence
        self.animation_render_time = time.time()  # track render time to keep steady fps

        self.scene = self.pyrender.Scene(
            bg_color=[1.0, 1.0, 1.0], ambient_light=(0.3, 0.3, 0.3)
        )
        self.viewer = self.pyrender.OffscreenRenderer(*(width, height), point_size=2.75)
        self.scene.bg_color = [1.0, 1.0, 1.0, 0.0]

        # Viewer camera
        self.follow_camera = follow_camera
        cx, cy = width / 2, height / 2
        self.fx, self.cx = focal, cx
        camera_pose = np.eye(4)
        camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
        camera = self.pyrender.camera.IntrinsicsCamera(fx=focal, fy=focal, cx=cx, cy=cy)
        self.camera_node = self.scene.add(camera, pose=camera_pose, name="pc-camera")

        # Camera and trajectory markers
        cam_marker = trimesh.load_mesh("./utils/cam_marker.stl")
        cam_marker.vertices = (cam_marker.vertices / cam_marker.vertices.max()) * 0.3
        cam_marker.vertices[:, 2] *= -1
        cam_marker.vertices[:, 2] += 0.3
        self.cam_marker = cam_marker
        self.cam_marker.visual.vertex_colors = CAM_COLOR
        self.traj_marker = trimesh.creation.uv_sphere(radius=0.05)
        self.traj_marker.visual.vertex_colors = TRAJ_COLOR

        # Viewer lighting
        light = self.pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
        self.scene.add(light, pose=camera_pose)
        self.use_raymond_lighting(3.5)

    # --------------------------------------------------------------------------------- #

    def _add_scene_object(self, object_seq: List[np.ndarray], seq_type: str):
        # Add to the list of sequences to render
        seq_id = len(self.animated_seqs)
        self.animated_seqs.append(object_seq)
        self.animated_seqs_type.append(seq_type)

        # Create the corresponding node in the scene
        anim_node = self.scene.add(object_seq[0], "anim-mesh-%2d" % (seq_id))
        self.animated_nodes.append(anim_node)

    def add_cam_seq(self, cam_seq: List[np.ndarray]):
        """Add a sequence of camera markers (visualized as pyramids)."""
        # Ensure same length as other sequences
        cur_seq_len = len(cam_seq)
        if self.animation_len != -1:
            if cur_seq_len != self.animation_len:
                print("Unexpected sequence length.")
                return
        else:
            if cur_seq_len > 0:
                self.animation_len = cur_seq_len
            else:
                print("Warning: points sequence is length 0!")
                return

        # Create sequence of pyrender meshes
        camera_markers, keyframe_markers = [], []
        for cam_index, cam_pose in enumerate(cam_seq):
            cam_marker = self.cam_marker.copy()
            camera_mesh = self.pyrender.Mesh.from_trimesh(
                cam_marker, poses=cam_pose, wireframe=True
            )
            camera_markers.append(camera_mesh)
            # ------------------------------------------------------------------------- #
            traj_marker = self.traj_marker.copy()
            sub_keyframe_markers = [
                self.pyrender.Mesh.from_trimesh(traj_marker, poses=cam_seq[k])
                for k in range(cam_index + 1)
            ]
            sub_keyframe_markers.extend([None] * (cur_seq_len - cam_index - 1))
            keyframe_markers.append(sub_keyframe_markers)
            # ------------------------------------------------------------------------- #

        # Add to the list of sequences to render
        index = len([x for x in self.animated_seqs_type if "cam_marker" in x])
        self._add_scene_object(camera_markers, f"cam_marker_{index}")

        for index, sub_keyframes in enumerate(keyframe_markers):
            self._add_scene_object(sub_keyframes, f"traj_marker_{index}")

    def add_body_seq(self, body_seq: Struct, mask: torch.Tensor = None):
        """Add a sequence of body to render."""
        # Ensure same length as other sequences
        cur_seq_len = body_seq.v.size(0)
        if self.animation_len != -1:
            if cur_seq_len != self.animation_len:
                print("Unexpected sequence length.")
                return
        else:
            if cur_seq_len > 0:
                self.animation_len = cur_seq_len
            else:
                print("Warning: mesh sequence is length 0!")
                return

        faces = c2c(body_seq.f)
        vertices = c2c(body_seq.v)
        colors = np.tile(VERTEX_COLOR, (vertices.shape[0], vertices.shape[1], 1))
        if mask is not None:
            colors[~mask] = MASKED_COLOR

        # Create sequence of pyrender meshes
        body_meshes = []
        for body_index, body_verts in enumerate(vertices):
            if body_verts.sum() == 0:
                break
            tmesh = trimesh.Trimesh(
                vertices=body_verts,
                faces=faces,
                vertex_colors=colors[body_index],
                process=False,
            )
            mesh = self.pyrender.Mesh.from_trimesh(tmesh.copy())
            body_meshes.append(mesh)

        # Add to the list of sequences to render
        index = len([x for x in self.animated_seqs_type if "char_mesh" in x])
        self._add_scene_object(body_meshes, seq_type=f"char_mesh_{index}")

    def add_ground(
        self,
        length: float = 25.0,
        color0: Tuple[float, float, float] = [0.8, 0.9, 0.9],
        color1: Tuple[float, float, float] = [0.6, 0.7, 0.7],
        tile_width: float = 0.5,
        alpha: float = 1.0,
    ):
        # Create checkerboard
        color0 = np.array(color0 + [alpha])
        color1 = np.array(color1 + [alpha])
        radius = length / 2.0
        num_rows = num_cols = int(length / tile_width)
        vertices, faces, face_colors = [], [], []
        for i, j in product(range(num_rows), range(num_cols)):
            start_loc = [-radius + j * tile_width, radius - i * tile_width]
            cur_verts = np.array(
                [
                    [start_loc[0], start_loc[1], 0.0],
                    [start_loc[0], start_loc[1] - tile_width, 0.0],
                    [start_loc[0] + tile_width, start_loc[1] - tile_width, 0.0],
                    [start_loc[0] + tile_width, start_loc[1], 0.0],
                ]
            )
            cur_faces = np.array([[0, 1, 3], [1, 2, 3]], dtype=int)
            cur_faces += 4 * (i * num_cols + j)  # the number of previously added verts
            use_color0 = (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1)
            cur_color = color0 if use_color0 else color1
            cur_face_colors = np.array([cur_color, cur_color])

            vertices.append(cur_verts)
            faces.append(cur_faces)
            face_colors.append(cur_face_colors)
        vertices = np.concatenate(vertices, axis=0)
        faces = np.concatenate(faces, axis=0)
        face_colors = np.concatenate(face_colors, axis=0)

        # Create ground mesh and add to scene
        ground_tri = trimesh.creation.Trimesh(
            vertices=vertices, faces=faces, face_colors=face_colors, process=False
        )
        ground_mesh = self.pyrender.Mesh.from_trimesh(ground_tri, smooth=False)
        self.scene.add(ground_mesh, "ground-mesh")

    def add_coordinate_axes(self, size=1.0):
        """Add coordinate axes to the scene (X: red / Y: green / Z: blue)."""
        axes = trimesh.creation.axis(axis_length=size)
        axes_mesh = self.pyrender.Mesh.from_trimesh(axes, smooth=False)
        self.scene.add(axes_mesh, name="coordinate_axes")

    # --------------------------------------------------------------------------------- #

    def update_frame(self):
        """
        Update frame to show the current self.animation_frame_idx
        """
        for seq_idx in range(len(self.animated_seqs)):
            cur_mesh = self.animated_seqs[seq_idx][self.animation_frame_idx]
            if cur_mesh is None:
                continue

            # Replace the old mesh
            anim_node = list(self.scene.get_nodes(name="anim-mesh-%2d" % (seq_idx)))
            anim_node = anim_node[0]
            anim_node.mesh = cur_mesh

        if self.follow_camera:
            # According to the camera and character centroids
            frame_index = self.animation_frame_idx
            centroids = []
            for seqs_type in self.animated_seqs_type:
                if "cam_marker" in seqs_type or "char_mesh" in seqs_type:
                    index = self.animated_seqs_type.index(seqs_type)
                    centroids.append(self.animated_seqs[index][frame_index].centroid)

            centroids = np.stack(centroids)
            distance = 5.0
            centroid = centroids[0]
            perpendicular = np.array([1.0, -1.0, 0.5])
            perpendicular /= np.linalg.norm(perpendicular)

            cam_trans = centroid + distance * perpendicular
            z_cam = -(centroid - cam_trans)
            z_cam /= np.linalg.norm(z_cam)
            x_cam = np.cross(np.array([0.0, 0.0, 1.0]), z_cam)
            x_cam /= np.linalg.norm(x_cam)
            y_cam = np.cross(z_cam, x_cam)
            y_cam /= np.linalg.norm(z_cam)
            cam_rot = np.stack([x_cam, y_cam, z_cam]).T

            camera_pose = np.eye(4)
            camera_pose[:3, 3] = cam_trans
            camera_pose[:3, :3] = cam_rot

            cam_node = list(self.scene.get_nodes(name="pc-camera"))
            cam_node = cam_node[0]
            self.scene.set_pose(cam_node, camera_pose)

    def render_frame(self) -> np.ndarray:
        RenderFlags = self.pyrender.constants.RenderFlags
        flags = RenderFlags.SHADOWS_DIRECTIONAL
        rendered_frame, _ = self.viewer.render(self.scene, flags=flags)
        return rendered_frame

    def render_sequence(self, progress_bar: Callable = None) -> np.ndarray:
        """
        Starts animating any given mesh sequences. This should be called last after adding
        all desired components to the scene as it is a blocking operation and will run
        until the user exits (or the full video is rendered if offline).
        """
        # Set up init frame
        self.update_frame()

        assert self.animation_frame_idx == 0

        iterator = (
            progress_bar(list(range(self.animation_len)), desc="SMPL rendering")
            if progress_bar is not None
            else range(self.animation_len)
        )

        rendered_frames = []
        for frame_idx in iterator:
            self.animation_frame_idx = frame_idx
            rendered_frames.append(self.render_frame())

            if self.animation_frame_idx + 1 >= self.animation_len:
                continue  # last iteration anyway

            self.animation_render_time = time.time()
            self.animation_frame_idx = self.animation_frame_idx + 1

            self.update_frame()

        self.animation_frame_idx = 0
        rendered_frames = np.stack(rendered_frames)

        return rendered_frames

    # --------------------------------------------------------------------------------- #

    def _add_raymond_light(self) -> List[Callable]:
        DirectionalLight = self.pyrender.light.DirectionalLight
        Node = self.pyrender.node.Node

        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []
        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                Node(
                    light=DirectionalLight(color=np.ones(3), intensity=1.0),
                    matrix=matrix,
                )
            )
        return nodes

    def use_raymond_lighting(self, intensity: float):
        for node in self._add_raymond_light():
            node.light.intensity = intensity / 3.0
            if not self.scene.has_node(node):
                self.scene.add_node(node)
            self.light_nodes.append(node)


# ------------------------------------------------------------------------------------- #


def launch_rendering(
    pyrender: Callable,
    body: Struct,
    body_mask: TensorType["num_frames", "num_vertices"],
    frame_width: int,
    frame_height: int,
    progress_bar: Callable,
):
    # Initialize the mesh viewer
    renderer = SceneRenderer(pyrender, width=frame_width, height=frame_height)

    # Add coordinate axes
    renderer.add_coordinate_axes(size=1.0)  # Adjust size as needed

    if body is not None:
        # Add character meshes
        if isinstance(body, list):
            for b, m in zip(body, body_mask):
                renderer.add_body_seq(b, m)
        else:
            renderer.add_body_seq(body, body_mask)
        # Add ground
        renderer.add_ground()

    # Render the sequence
    frames = renderer.render_sequence(progress_bar=progress_bar)

    return frames


def render_frames(
    vertices: TensorType["num_frames", "num_vertices", 3],
    frame_width: int,
    frame_height: int,
    body_mask: TensorType["num_frames", "num_vertices"] = None,
    progress_bar: Callable = None,
):
    if vertices is not None:
        _vertices = vertices.clone()
        # Put the vertices and cameras at the floor level
        ground = _vertices[..., 2].min()
        _vertices[..., 2] -= ground
        # Create a body object
        body_pred = Struct(v=_vertices, f=FACES)
    else:
        body_pred = None
        ground = torch.zeros(1)

    # Render the animation
    frames = launch_rendering(
        pyrender, body_pred, body_mask, frame_width, frame_height, progress_bar
    )
    return frames


def smpl_to_mp4(vertices: SMPL, out_path: Path):
    frames = render_frames(vertices, 368, 368, None)
    imageio.mimwrite(out_path, frames.astype(np.uint8), fps=30)
