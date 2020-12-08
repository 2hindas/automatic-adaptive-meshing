import numpy as np
import matplotlib.pyplot as plt
from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz
from scipy.spatial.transform import Rotation as R

domain= ((-1000, 1000), (-1000, 1000), (-1000, 1000))

# minimum cell width in each dimension
dx = 16
dy = 16
dz = 16

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
#  mesh = TreeMesh([hx, hy, hz], origin=[-1000,-1000,-1000])
mesh = TreeMesh([hx, hy, hz], x0="CCC")

# Define rotation matrix
# 10 degrees rotation around the x-axis
# rotation = R.from_euler('x', 10, degrees=True).as_matrix()

x = np.linspace(-500, 500, 50)
y = np.linspace(-500, 500, 50)
z = np.linspace(-500, 500, 50)
xp, yp, zp = np.meshgrid(x, y, z)
xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]

# Uncomment this part to show the rotated block domain 
# xyz = np.asarray([np.dot(rotation, i) for i in xyz])
# mesh = refine_tree_xyz(mesh, np.asarray(xyz), octree_levels=[2, 2], method="surface", finalize=False)

# Uncomment this part to show a sphere
sphere_xyz = []
for row in xyz:
    if (row[0])**2 + (row[1])**2 + (row[2])**2 < 500**2 and (row[0])**2 + (row[1])**2 + (row[2])**2 > 480**2:
        sphere_xyz.append(row)
mesh = refine_tree_xyz(mesh, np.asarray(sphere_xyz), octree_levels=[2, 2, 2], method="surface", max_distance=10, finalize=False)

mesh.finalize()

# Plot cell volumes
v = mesh.cell_volumes
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
index = 64 # the index of the Z-slice that is plotted 
mesh.plot_slice(np.log10(v), normal="Z", ax=ax, ind=index, grid=True)
ax.set_title(f"Cell Log-Volumes at Z = {index}")
plt.show()
