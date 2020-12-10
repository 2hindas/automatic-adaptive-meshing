import numpy as np
import matplotlib.pyplot as plt
from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz
from scipy.spatial.transform import Rotation as R

domain= ((-1000, 1000), (-1000, 1000), (-1000, 1000))

x = np.linspace(-500, 500, 50)
y = np.linspace(-500, 500, 50)
z = np.linspace(-500, 500, 50)
xp, yp, zp = np.meshgrid(x, y, z)
xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]

# Rotated block
# rotation = R.from_euler('x', 10, degrees=True).as_matrix()
# xyz = np.asarray([np.dot(rotation, i) for i in xyz])
# mesh = refine_tree_xyz(mesh, np.asarray(xyz), octree_levels=[2, 2], method="surface", finalize=False)

# Sphere 
sphere_xyz = []
for row in xyz:
    if (row[0])**2 + (row[1])**2 + (row[2])**2 < 500**2 and (row[0])**2 + (row[1])**2 + (row[2])**2 > 480**2:
        sphere_xyz.append(row)

def create_octree_mesh(domain, cellwidth, points):
    
    # domain dimensions
    x_length = np.abs(domain[0][0] - domain[0][1]) 
    y_length = np.abs(domain[1][0] - domain[1][1]) 
    z_length = np.abs(domain[2][0] - domain[2][1]) 

    # number of cells needed in each dimension
    nbx = 2 ** int(np.round(np.log(x_length / cellwidth) / np.log(2.0)))
    nby = 2 ** int(np.round(np.log(y_length / cellwidth) / np.log(2.0)))
    nbz = 2 ** int(np.round(np.log(z_length / cellwidth) / np.log(2.0)))

    # define base mesh 
    hx = cellwidth * np.ones(nbx)
    hy = cellwidth * np.ones(nby)
    hz = cellwidth * np.ones(nbz)
    mesh = TreeMesh([hx, hy, hz], x0="CCC")

    # refine mesh around the given surface points
    mesh = refine_tree_xyz(mesh, points, octree_levels=[2, 2, 2], method="surface", max_distance=10, finalize=True)

    return mesh


def plot_mesh_slice(mesh, axis, index):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    v = mesh.cell_volumes
    mesh.plot_slice(np.log10(v), normal=axis.upper(), ax=ax, ind=index, grid=True)
    ax.set_title(f"Cell Log-Volumes at {axis.upper()} = {index}")
    plt.show()


