import numpy as np
import matplotlib.pyplot as plt
from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz
from scipy.spatial.transform import Rotation as R

domain= ((-500, 4500), (-1000, 1000), (-1200, 200))

# minimum cell width in each dimension
dx = 20
dy = 20
dz = 20

# domain dimensions
x_length = np.abs(domain[0][0] - domain[0][1]) 
y_length = np.abs(domain[1][0] - domain[1][1]) 
z_length = np.abs(domain[2][0] - domain[2][1]) 

# number of cells necessary in each dimension
nbx = 2 ** int(np.round(np.log(x_length / dx) / np.log(2.0)))
nby = 2 ** int(np.round(np.log(y_length / dy) / np.log(2.0)))
nbz = 2 ** int(np.round(np.log(z_length / dz) / np.log(2.0)))

# Define base mesh (domain and finest discretization)
hx = dx * np.ones(nbx)
hy = dy * np.ones(nby)
hz = dz * np.ones(nbz)
mesh = TreeMesh([hx, hy, hz], origin=[-500,-1000,-1200])

# Define rotation matrix
# 10 degrees rotation around the x-axis
rotation = R.from_euler('x', 10, degrees=True).as_matrix()

# Define inner points for rectangular box
x = np.linspace(0, 4000, 30)
y = np.linspace(-500, 500, 30)
z = np.linspace(-1200, -1000, 30)
xp, yp, zp = np.meshgrid(x, y, z)
xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]

# Uncomment this part to show the rotated block domain 
xyz = np.asarray([np.dot(rotation, i) for i in xyz])
mesh = refine_tree_xyz(mesh, np.asarray(xyz), octree_levels=[2, 2], method="surface", finalize=False)

# Uncomment this part to show a sphere
"""
sphere_xyz = []
for row in xyz:
    if (row[0] - 3000)**2 + (row[1] + 200)**2 + (row[2] + 1150)**2 < 500**2:
        sphere_xyz.append(row)
mesh = refine_tree_xyz(mesh, np.asarray(sphere_xyz), octree_levels=[2, 2], method="surface", finalize=False)
"""

mesh.finalize()

# Plot cell volumes
v = mesh.cell_volumes
fig = plt.figure(figsize=(10, 2.5))
ax = fig.add_subplot(111)
index = 10  # the index of the Z-slice that is plotted 
mesh.plot_slice(np.log10(v), normal="Z", ax=ax, ind=index, grid=True)
ax.set_title(f"Cell Log-Volumes at Z = {index}")
plt.show()
