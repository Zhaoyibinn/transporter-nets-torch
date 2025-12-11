# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Insertion Tasks."""

import numpy as np
from ravens_torch.tasks.task import Task
from ravens_torch.utils import utils
import os
from pathlib import Path

import trimesh
import open3d as o3d

import pybullet as p


class BlockInsertion(Task):
    """Insertion Task - Base Variant."""

    def __init__(self):
        super().__init__()
        self.max_steps = 3

    def reset(self, env):
        super().reset(env)
        self._add_instance(env)

    def _add_instance(self, env):
        # block_id = self.add_block(env)
        # targ_pose = self.add_fixture(env)
        
        targ_pose = self.add_fixture_ret(env)

        own_obj_id = self.add_block_own(env)
        self.own_obj_id = own_obj_id
        # self.goals.append(
        #     ([block_id], [2 * np.pi], [[0]], [targ_pose], 'pose', None, 1.))
        # self.goals.append(([(block_id, (2 * np.pi, None))], np.int32([[1]]),
        #                    [targ_pose], False, True, 'pose', None, 1))
        self.goals.append(([(own_obj_id, (2 * np.pi, None))], np.int32([[1]]),
                           [targ_pose], False, True, 'pose', None, 1))

    def add_block(self, env):
        """Add L-shaped block."""
        size = (0.1, 0.1, 0.04)
        urdf = 'insertion/ell.urdf'
        pose = self.get_random_pose(env, size)


        return env.add_object(urdf, pose)
    
    def add_block_own(self, env):
        size_own = (0.1, 0.1, 0.04)
        # scene = "milk"
        # self.urdf_own = f'insertion/{scene}/fuse_post.urdf'
        # # self.GS_own = 'GS/point_cloud.ply'
        # self.GS_own = f'insertion/{scene}/point_cloud.ply'
        # self.extra_pose_own = f'insertion/{scene}/T.txt'
        # mesh_own = f'insertion/{scene}/fuse_post.ply'
        if not os.path.exists(env.assets_root + f'/insertion/{self.scene}/fuse_post_trans.stl'):
            self.trans_mesh(self.extra_pose_own,self.mesh_own,env)
        self.pose_own = self.get_random_pose_own(env, size_own)
        # self.pose_own = ((0.0,0.0,0.0),(0.0,0.0,0.0,0.0))
        return env.add_object(self.urdf_own, self.pose_own)

    def trans_mesh(self, extra_pose_own, mesh_own, env):
        asset_root = Path(env.assets_root)
        mesh_path = asset_root / mesh_own
        extra_pose_path = asset_root / extra_pose_own
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        if not extra_pose_path.exists():
            raise FileNotFoundError(f"Extra pose file not found: {extra_pose_path}")

        extra_pose = np.loadtxt(extra_pose_path)
        translation = np.asarray(extra_pose[0], dtype=np.float32)
        euler_deg = np.asarray(extra_pose[1], dtype=np.float32)
        if extra_pose.shape[0] > 2:
            scale_vals = np.asarray(extra_pose[2], dtype=np.float32).ravel()
            scale = float(scale_vals[0]) if scale_vals.size else 1.0
        else:
            scale = 1.0
        rotation_rad = np.deg2rad(euler_deg)

        mesh = trimesh.load(mesh_path, force='mesh').copy()

        # Convert Y-up assets into PyBullet's Z-up coordinate system.

        transform = trimesh.transformations.euler_matrix(
            rotation_rad[0], rotation_rad[1], rotation_rad[2])
        transform[:3, 3] = translation

        mesh.apply_transform(transform)
        if not np.isclose(scale, 1.0):
            mesh.apply_scale(scale)

        mesh = self._simplify_mesh(mesh, target_faces=4000)

        y_up_to_z_up = trimesh.transformations.euler_matrix(np.pi / 2, 0, 0)
        mesh.apply_transform(y_up_to_z_up)

        output_path = mesh_path.with_name('fuse_post_trans.stl')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(output_path)
        return str(output_path)

    @staticmethod
    def _simplify_mesh(mesh, target_faces):
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError('Loaded geometry is not a single mesh.')

        if mesh.faces.shape[0] <= target_faces:
            return mesh

        if hasattr(mesh, 'simplify_quadratic_decimation'):
            simplified = mesh.simplify_quadratic_decimation(target_faces)
            if isinstance(simplified, trimesh.Trimesh):
                return simplified
            raise RuntimeError('Mesh simplification failed to return a mesh instance.')

        if o3d is None:
            raise AttributeError(
                'Current trimesh build lacks simplify_quadratic_decimation and open3d is not installed. '
                'Install open3d to enable mesh downsampling.')

        o3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices),
            o3d.utility.Vector3iVector(mesh.faces))
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_faces)
        o3d_mesh.remove_degenerate_triangles()
        o3d_mesh.remove_duplicated_triangles()
        o3d_mesh.remove_non_manifold_edges()
        return trimesh.Trimesh(
            vertices=np.asarray(o3d_mesh.vertices),
            faces=np.asarray(o3d_mesh.triangles),
            process=True)

    def add_fixture(self, env):
        """Add L-shaped fixture to place block."""
        size = (0.1, 0.1, 0.04)
        urdf = 'insertion/fixture.urdf'
        pose = self.get_random_pose(env, size)
        env.add_object(urdf, pose, 'fixed')
        return pose
    
    def add_fixture_ret(self, env):
        """改为一个长方形的容器"""
        size = (0.1, 0.12, 0.08)
        urdf = 'insertion/fixture_ret.urdf'
        pose = self.get_random_pose(env, size)
        env.add_object(urdf, pose, 'fixed')
        return pose

class BlockInsertionTranslation(BlockInsertion):
    """Insertion Task - Translation Variant."""

    def get_random_pose(self, env, obj_size):
        pose = super(BlockInsertionTranslation,
                     self).get_random_pose(env, obj_size)
        pos, rot = pose
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, np.pi / 2))
        return pos, rot

    # Visualization positions.
    # block_pos = (0.40, -0.15, 0.02)
    # fixture_pos = (0.65, 0.10, 0.02)


class BlockInsertionEasy(BlockInsertionTranslation):
    """Insertion Task - Easy Variant."""

    def add_block(self, env):
        """Add L-shaped block in fixed position."""
        # size = (0.1, 0.1, 0.04)
        urdf = 'insertion/ell.urdf'
        pose = ((0.5, 0, 0.02), p.getQuaternionFromEuler((0, 0, np.pi / 2)))
        return env.add_object(urdf, pose)


class BlockInsertionSixDof(BlockInsertion):
    """Insertion Task - 6DOF Variant."""

    def __init__(self):
        super().__init__()
        self.sixdof = True
        self.pos_eps = 0.02

    def add_fixture(self, env):
        """Add L-shaped fixture to place block."""
        size = (0.1, 0.1, 0.04)
        urdf = 'insertion/fixture.urdf'
        pose = self.get_random_pose_6dof(env, size)
        env.add_object(urdf, pose, 'fixed')
        return pose

    def get_random_pose_6dof(self, env, obj_size):
        pos, rot = super(BlockInsertionSixDof,
                         self).get_random_pose(env, obj_size)
        z = (np.random.rand() / 10) + 0.03
        pos = (pos[0], pos[1], obj_size[2] / 2 + z)
        roll = (np.random.rand() - 0.5) * np.pi / 2
        pitch = (np.random.rand() - 0.5) * np.pi / 2
        yaw = np.random.rand() * 2 * np.pi
        rot = utils.eulerXYZ_to_quatXYZW((roll, pitch, yaw))
        return pos, rot


class BlockInsertionNoFixture(BlockInsertion):
    """Insertion Task - No Fixture Variant."""

    def add_fixture(self, env):
        """Add target pose to place block."""
        size = (0.1, 0.1, 0.04)
        # urdf = 'insertion/fixture.urdf'
        pose = self.get_random_pose(env, size)
        return pose

    # def reset(self, env, last_info=None):
    #   self.num_steps = 1
    #   self.goal = {'places': {}, 'steps': []}

    #   # Add L-shaped block.
    #   block_size = (0.1, 0.1, 0.04)
    #   block_urdf = 'insertion/ell.urdf'
    #   block_pose = self.get_random_pose(env, block_size)
    #   block_id = env.add_object(block_urdf, block_pose)
    #   self.goal['steps'].append({block_id: (2 * np.pi, [0])})

    #   # Add L-shaped target pose, but without actually adding it.
    #   if self.goal_cond_testing:
    #     assert last_info is not None
    #     self.goal['places'][0] = self._get_goal_info(last_info)
    #     # print('\nin insertion reset, goal: {}'.format(self.goal['places'][0]))
    #   else:
    #     hole_pose = self.get_random_pose(env, block_size)
    #     self.goal['places'][0] = hole_pose
    #     # print('\nin insertion reset, goal: {}'.format(hole_pose))

    # def _get_goal_info(self, last_info):
    #   """Used to determine the goal given the last `info` dict."""
    #   position, rotation, _ = last_info[4]  # block ID=4
    #   return (position, rotation)
