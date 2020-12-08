import numpy as np
from discretize.utils import mkvc, refine_tree_xyz
from scipy.spatial.transform import Rotation as R
from Meshing import * 


domain= ((-1000, 1000), (-1000, 1000), (-1000, 1000))

x = np.linspace(-500, 500, 50)
y = np.linspace(-500, 500, 50)
z = np.linspace(-500, 500, 50)
xp, yp, zp = np.meshgrid(x, y, z)
xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]

# Rotated block
cube_xyz = []
for row in xyz:
    if abs(row[0]) == 500 or abs(row[1]) == 500 or abs(row[2]) == 500:
        cube_xyz.append(row)
xyz = np.asarray(cube_xyz)
rotation = R.from_euler('x', 10, degrees=True).as_matrix()
xyz = np.asarray([np.dot(rotation, i) for i in xyz])

# Sphere
"""
sphere_xyz = []
for row in xyz:
    if (row[0])**2 + (row[1])**2 + (row[2])**2 < 500**2 and (row[0])**2 + (row[1])**2 + (row[2])**2 > 480**2:
        sphere_xyz.append(row)
xyz = np.asarray(sphere_xyz)
"""

mesh = create_octree_mesh(domain, 16, xyz)
plot_mesh_slice(mesh, 'X', 64)


