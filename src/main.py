import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from discretize.utils import mkvc
import numpy as np
import src.Meshing as M
import src.Utils as utils
from SimPEG.utils import surface2ind_topo
from SimPEG import maps
from src.LapentaEstimator import iterator

domain = ((-500, 4500), (-1000, 1000), (-1200, 200))

# Surface coordinates
xx, yy = np.meshgrid(np.linspace(-500, 4500, 101), np.linspace(-1000, 1000, 101))
zz = np.zeros(np.shape(xx))
surface = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

x = np.linspace(200, 600, 30)
y = np.linspace(-500, 500, 30)
z = np.linspace(-1000, -800, 30)
xp, yp, zp = np.meshgrid(x, y, z)
block = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
cell_width = 50
mesh = M.create_octree_mesh(domain, cell_width, block, 'surface')

# plot_mesh_slice(mesh, 'z', 0)

# Defining transmitter location
xtx, ytx, ztx = np.meshgrid([0], [0], [0])
source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
ntx = np.size(xtx)
print("Number of transmitters", len(source_locations))

# Define receiver locations
N = 21
xrx, yrx, zrx = np.meshgrid(np.linspace(2000, 4000, N), [0], [0])
receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]
frequencies = [1.0]  # Frequency (Hz)
omegas = [2.0 * np.pi]  # Radial frequency (Hz)
print("Number of receivers", len(receiver_locations))

survey = utils.define_survey(frequencies, receiver_locations, source_locations, ntx)

# Refine at certain locations
# M.refine_at_locations(mesh, surface)
M.refine_at_locations(mesh, source_locations)
M.refine_at_locations(mesh, receiver_locations)

mesh.finalize()

print(mesh)

print("Total number of cells", mesh.nC)
print("Total number of cell faces", mesh.n_faces)
print("Total number of cell edges", mesh.n_edges)

# Resistivity in Ohm m
res_background = 1.0
res_block = 100.0

# Conductivity in S/m
conductivity_background = 1 / 100.0
conductivity_block = 1 / 1.0

# Find cells that are active in the forward modeling (cells below surface)
ind_active = surface2ind_topo(mesh, surface)

# Define mapping from model to active cells
model_map = maps.InjectActiveCells(mesh, ind_active, res_background)

# Define model. Models in SimPEG are vector arrays
model = res_background * np.ones(ind_active.sum())

ind_block = utils.get_ind_block(mesh, ind_active, x, y, z)
model[ind_block] = res_block

mesh = iterator(mesh, domain, surface, cell_width, block, M.create_box_surface, x, y, z
                , receiver_locations, source_locations, survey
                , res_background, res_block, model_map
                , model, ind_block, lim_iterations=3)
print(mesh)

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cells_to_refine[:, 0], cells_to_refine[:, 1], cells_to_refine[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
# refine cells
'''