import numpy as np
import matplotlib.pyplot as plt
from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz
from scipy.spatial.transform import Rotation as R

surface_width = 25


def create_box_surface(x, y, z, axis, degree_rad):
    xp, yp, zp = np.meshgrid(x, y, z)
    xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
    box_xyz = []
    for row in xyz:
        if x[0] < row[0] < x[0] + surface_width:
            box_xyz.append(row)
        if x[-1] > row[0] > x[-1] - surface_width:
            box_xyz.append(row)
        if y[0] < row[1] < y[0] + surface_width:
            box_xyz.append(row)
        if y[-1] > row[1] > y[-1] - surface_width:
            box_xyz.append(row)
        if z[0] < row[2] < z[0] + surface_width:
            box_xyz.append(row)
        if z[-1] > row[2] > z[-1] - surface_width:
            box_xyz.append(row)

    rotation = R.from_euler(axis, degree_rad, degrees=True).as_matrix()
    box_xyz = np.asarray([np.dot(rotation, i) for i in box_xyz])
    return np.asarray(box_xyz)


def create_sphere_surface(x, y, z, radius):
    xp, yp, zp = np.meshgrid(x, y, z)
    xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
    sphere_xyz = []
    for row in xyz:
        if row[0] ** 2 + row[1] ** 2 + row[2] ** 2 < radius ** 2 and row[0] ** 2 + row[1] ** 2 + \
                row[2] ** 2 > (radius - surface_width) ** 2:
            sphere_xyz.append(row)
    return np.asarray(sphere_xyz)


def create_octree_mesh(domain, cellwidth, points, method='surface'):
    # domain dimensions
    x_length = np.abs(domain[0][0] - domain[0][1])
    y_length = np.abs(domain[1][0] - domain[1][1])
    z_length = np.abs(domain[2][0] - domain[2][1])

    # number of cells needed in each dimension
    nbx = 2 ** int(np.ceil(np.log(x_length / cellwidth) / np.log(2.0)))
    nby = 2 ** int(np.ceil(np.log(y_length / cellwidth) / np.log(2.0)))
    nbz = 2 ** int(np.ceil(np.log(z_length / cellwidth) / np.log(2.0)))

    # define base mesh 
    hx = cellwidth * np.ones(nbx)
    hy = cellwidth * np.ones(nby)
    hz = cellwidth * np.ones(nbz)
    mesh = TreeMesh([hx, hy, hz], origin=[domain[0][0], domain[1][0], domain[2][0]])

    # refine mesh around the given surface points
    mesh = refine_tree_xyz(mesh, points, octree_levels=[2, 2, 2], method=method, max_distance=10,
                           finalize=False)

    return mesh


def refine_at_locations(mesh, locations):
    mesh = refine_tree_xyz(
        mesh, locations, octree_levels=[2, 4], method="radial", finalize=False
    )
    return mesh


def plot_mesh_slice(mesh, axis, index):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    v = mesh.cell_volumes
    mesh.plot_slice(np.log10(v), normal=axis.upper(), ax=ax, ind=index, grid=True)
    ax.set_title(f"Cell Log-Volumes at {axis.upper()} = {index}")
    # plt.savefig(f"Block {axis.upper()} = {index}")
    plt.show()
