"""URDF visualizer using MeshCat.

This module implements a lightweight URDF parser and visualizer using
meshcat only. It intentionally avoids any dependency on placo or
pinocchio. It supports:
 - loading basic visuals from URDF (box, sphere, cylinder)
 - displaying link frames (axes)
 - a simple animation mode that sinusoidally drives revolute/continuous
   joints to demonstrate motion

Usage:
  python3 ik_visualization.py path/to/robot.urdf [--frames frame1 frame2] [--animate]

Notes:
 - Mesh geometries referenced via mesh filename are skipped (placeholder used).
 - This is a pragmatic viewer for debugging and quick visualization when
   pinocchio/placo are not available.
"""

from __future__ import annotations

import argparse
import time
import math
import os
import xml.etree.ElementTree as ET
from typing import Dict, Tuple, List, Optional

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np


def parse_origin(elem: ET.Element) -> Tuple[np.ndarray, np.ndarray]:
    """Parse an <origin> element and return (xyz, rpy) as numpy arrays.

    If element is missing, return zeros.
    """
    xyz = np.zeros(3)
    rpy = np.zeros(3)
    if elem is None:
        return xyz, rpy
    if 'xyz' in elem.attrib:
        xyz = np.fromstring(elem.attrib['xyz'], sep=' ')
    if 'rpy' in elem.attrib:
        rpy = np.fromstring(elem.attrib['rpy'], sep=' ')
    return xyz, rpy


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Convert rpy (roll,pitch,yaw) to a 4x4 transform matrix."""
    roll, pitch, yaw = rpy
    R = tf.euler_matrix(roll, pitch, yaw, 'sxyz')
    return R


def make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = tf.translation_matrix(xyz)
    T = T @ rpy_to_matrix(rpy)
    return T


class URDFModel:
    """Simple URDF model containing links and joints minimal info."""

    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # collect links
        self.links: Dict[str, Dict] = {}
        for link in root.findall('link'):
            name = link.attrib.get('name')
            visuals = []
            for vis in link.findall('visual'):
                geom = vis.find('geometry')
                origin = vis.find('origin')
                xyz, rpy = parse_origin(origin)
                if geom is None:
                    continue
                if geom.find('box') is not None:
                    size = np.fromstring(geom.find('box').attrib.get('size', '0 0 0'), sep=' ')
                    visuals.append(('box', size, xyz, rpy))
                elif geom.find('sphere') is not None:
                    radius = float(geom.find('sphere').attrib.get('radius', 0.0))
                    visuals.append(('sphere', radius, xyz, rpy))
                elif geom.find('cylinder') is not None:
                    cyl = geom.find('cylinder')
                    length = float(cyl.attrib.get('length', 0.0))
                    radius = float(cyl.attrib.get('radius', 0.0))
                    visuals.append(('cylinder', (length, radius), xyz, rpy))
                elif geom.find('mesh') is not None:
                    # mesh references filename; extract scale too
                    mesh = geom.find('mesh')
                    fname = mesh.attrib.get('filename', '')
                    scale_str = mesh.attrib.get('scale', '1 1 1')
                    scale = np.fromstring(scale_str, sep=' ')
                    if len(scale) == 1:
                        scale = np.array([scale[0], scale[0], scale[0]])
                    elif len(scale) != 3:
                        scale = np.array([1.0, 1.0, 1.0])
                    visuals.append(('mesh', (fname, scale), xyz, rpy))
            self.links[name] = {'visuals': visuals}

        # collect joints
        self.joints: Dict[str, Dict] = {}
        child_links = set()
        for joint in root.findall('joint'):
            name = joint.attrib.get('name')
            jtype = joint.attrib.get('type')
            parent = joint.find('parent').attrib.get('link')
            child = joint.find('child').attrib.get('link')
            origin = joint.find('origin')
            xyz, rpy = parse_origin(origin)
            axis_elem = joint.find('axis')
            axis = np.array([1.0, 0.0, 0.0])
            if axis_elem is not None and 'xyz' in axis_elem.attrib:
                axis = np.fromstring(axis_elem.attrib['xyz'], sep=' ')
            self.joints[name] = {'type': jtype, 'parent': parent, 'child': child, 'origin_xyz': xyz, 'origin_rpy': rpy, 'axis': axis}
            child_links.add(child)

        # find root link (a link that is never a child)
        roots = [ln for ln in self.links.keys() if ln not in child_links]
        self.root = roots[0] if roots else next(iter(self.links.keys()), None)

        # build adjacency from parent->child joints for kinematics
        self.children: Dict[str, List[str]] = {}
        for jn, jinfo in self.joints.items():
            p = jinfo['parent']
            self.children.setdefault(p, []).append(jn)


class URDFMeshcatViewer:
    def __init__(self, model: URDFModel, open_browser: bool = True):
        self.model = model
        try:
            self.vis = meshcat.Visualizer().open() if open_browser else meshcat.Visualizer()
        except Exception as e:
            print(f"Warning: Failed to open browser automatically: {e}")
            print("You can still access the visualizer at the URL shown above.")
            self.vis = meshcat.Visualizer()
        self.link_nodes: Dict[str, str] = {}

        # create visuals
        for lname, linfo in model.links.items():
            node = f'links/{lname}'
            self.link_nodes[lname] = node
            # create a group node; geometry will be placed under node+'/geom'
            for k, visinfo in enumerate(linfo['visuals']):
                geom_type = visinfo[0]
                if geom_type == 'box':
                    size, xyz, rpy = visinfo[1], visinfo[2], visinfo[3]
                    obj = g.Box(size.tolist())
                    self.vis[node + f'/geom{k}'].set_object(obj)
                    self.vis[node + f'/geom{k}'].set_transform(make_transform(xyz, rpy))
                elif geom_type == 'sphere':
                    radius, xyz, rpy = visinfo[1], visinfo[2], visinfo[3]
                    obj = g.Sphere(radius)
                    self.vis[node + f'/geom{k}'].set_object(obj)
                    self.vis[node + f'/geom{k}'].set_transform(tf.translation_matrix(xyz))
                elif geom_type == 'cylinder':
                    (length, radius), xyz, rpy = visinfo[1], visinfo[2], visinfo[3]
                    obj = g.Cylinder(length, radius)
                    self.vis[node + f'/geom{k}'].set_object(obj)
                    self.vis[node + f'/geom{k}'].set_transform(make_transform(xyz, rpy))
                elif geom_type == 'mesh':
                    # load actual mesh file
                    (fname, scale), xyz, rpy = visinfo[1], visinfo[2], visinfo[3]
                    try:
                        # meshcat can handle dae files directly
                        if os.path.exists(fname):
                            obj = g.DaeMeshGeometry.from_file(fname)
                            T = make_transform(xyz, rpy)
                            # apply scale
                            scale_matrix = np.diag([scale[0], scale[1], scale[2], 1.0])
                            T = T @ scale_matrix
                            self.vis[node + f'/geom{k}'].set_object(obj)
                            self.vis[node + f'/geom{k}'].set_transform(T)
                        else:
                            # fallback to small box if file not found
                            obj = g.Box([0.05, 0.05, 0.05])
                            self.vis[node + f'/geom{k}'].set_object(obj)
                            self.vis[node + f'/geom{k}'].set_transform(make_transform(xyz, rpy))
                    except Exception as e:
                        print(f"Warning: Could not load mesh {fname}: {e}")
                        # fallback to small box on error
                        obj = g.Box([0.05, 0.05, 0.05])
                        self.vis[node + f'/geom{k}'].set_object(obj)
                        self.vis[node + f'/geom{k}'].set_transform(make_transform(xyz, rpy))

            # frame axes
            axes_node = node + '/frame'
            # x (red)
            self.vis[axes_node + '/x'].set_object(g.Cylinder(0.1, 0.01), g.MeshLambertMaterial(color=0xFF0000))
            # y (green)
            self.vis[axes_node + '/y'].set_object(g.Cylinder(0.1, 0.01), g.MeshLambertMaterial(color=0x00FF00))
            # z (blue)
            self.vis[axes_node + '/z'].set_object(g.Cylinder(0.1, 0.01), g.MeshLambertMaterial(color=0x0000FF))

            self.motor_map = {
            "LF_HAA": 7,
            "LF_HFE": 8,
            "LF_KFE": 9,
            "RF_HAA": 10,
            "RF_HFE": 11,
            "RF_KFE": 12,
            "LH_HAA": 13,
            "LH_HFE": 14,
            "LH_KFE": 15,
            "RH_HAA": 16,
            "RH_HFE": 17,
            "RH_KFE": 18
        }

    def set_link_transform(self, link_name: str, T: np.ndarray):
        node = self.link_nodes[link_name]
        self.vis[node].set_transform(T)

    def display_static(self):
        # compute transforms by traversing joint tree using zero joint angles
        model = self.model

        def recurse(link_name: str, parent_T: np.ndarray):
            # set link transform
            self.set_link_transform(link_name, parent_T)
            # recursively apply children joints
            for jn in model.children.get(link_name, []):
                jinfo = model.joints[jn]
                origin_T = make_transform(jinfo['origin_xyz'], jinfo['origin_rpy'])
                # with zero joint angle, joint rotation is identity
                child_T = parent_T @ origin_T
                recurse(jinfo['child'], child_T)

        root = model.root
        recurse(root, np.eye(4))

    def animate(self, duration: float = 10.0, rate: float = 60.0):
        # For revolute or continuous joints, drive angle = sin(t)
        model = self.model
        t0 = time.time()
        while True:
            t = time.time() - t0
            # build dict of joint angles
            angles: Dict[str, float] = {}
            for jn, jinfo in model.joints.items():
                if jinfo['type'] in ('revolute', 'continuous'):
                    angles[jn] = math.sin(t)
                    print(f'Joint {jn} angle: {angles[jn]:.3f}')
                else:
                    angles[jn] = 0.0

            # traverse and compute transforms
            def recurse(link_name: str, parent_T: np.ndarray):
                self.set_link_transform(link_name, parent_T)
                for jn in model.children.get(link_name, []):
                    jinfo = model.joints[jn]
                    origin_T = make_transform(jinfo['origin_xyz'], jinfo['origin_rpy'])
                    angle = angles.get(jn, 0.0)
                    axis = jinfo['axis']
                    # build rotation about axis
                    R = tf.rotation_matrix(angle, axis)
                    child_T = parent_T @ origin_T @ R
                    recurse(jinfo['child'], child_T)

            recurse(model.root, np.eye(4))
            time.sleep(1.0 / rate)

    def animate_state(self, state_desired=np.zeros(12), rate: float = 60.0):
            # For revolute or continuous joints, drive angle = sin(t)
            model = self.model
            # while True:
            # build dict of joint angles
            angles: Dict[str, float] = {}
            for jn, jinfo in model.joints.items():
                if jinfo['type'] in ('revolute', 'continuous'):
                    angles[jn] = state_desired[self.motor_map[jn]]
                    # print(f'Joint {jn} angle: {angles[jn]:.3f}')
                else:
                    angles[jn] = 0.0

            # traverse and compute transforms
            def recurse(link_name: str, parent_T: np.ndarray):
                self.set_link_transform(link_name, parent_T)
                for jn in model.children.get(link_name, []):
                    jinfo = model.joints[jn]
                    origin_T = make_transform(jinfo['origin_xyz'], jinfo['origin_rpy'])
                    angle = angles.get(jn, 0.0)
                    axis = jinfo['axis']
                    # build rotation about axis
                    R = tf.rotation_matrix(angle, axis)
                    child_T = parent_T @ origin_T @ R
                    recurse(jinfo['child'], child_T)

            recurse(model.root, np.eye(4))
            time.sleep(1.0 / rate)


def main():
    parser = argparse.ArgumentParser(description='Simple URDF MeshCat viewer (no pinocchio/placo)')
    parser.add_argument('path', help='Path to URDF file')
    parser.add_argument('--frames', nargs='+', help='Frame names to display (unused currently)')
    parser.add_argument('--animate', action='store_true', help='Animate joints (sinusoidal)')
    parser.add_argument('--no-browser', action='store_true', help='Do not try to open browser automatically')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise FileNotFoundError(args.path)

    model = URDFModel(args.path)
    viewer = URDFMeshcatViewer(model, open_browser=not args.no_browser)
    viewer.display_static()

    if args.animate:
        print('Starting animation (Ctrl-C to stop)')
        viewer.animate()


if __name__ == '__main__':
    main()