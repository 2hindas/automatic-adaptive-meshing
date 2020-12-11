import numpy as np
from discretize.utils import mkvc, refine_tree_xyz
from scipy.spatial.transform import Rotation as R
from src.Meshing import *

domain = ((-1000, 1000), (-1000, 1000), (-1000, 1000))

x = np.linspace(-500, 500, 50)
y = np.linspace(-500, 500, 50)
z = np.linspace(-500, 500, 50)

surface = create_box_surface(x, y, z, 'x', 30)
mesh = create_octree_mesh(domain, 16, surface)
plot_mesh_slice(mesh, 'x', 65)
