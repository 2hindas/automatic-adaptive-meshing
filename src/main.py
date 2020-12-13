import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from src.Meshing import *
from src.Setup import *
from SimPEG.utils import surface2ind_topo
from SimPEG import maps
from src.LapentaEstimator import estimate_error

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
mesh = create_octree_mesh(domain, cell_width, block, 'surface')

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

survey = define_survey(frequencies, receiver_locations, source_locations, ntx)

refine_at_locations(mesh, source_locations)
refine_at_locations(mesh, source_locations)

mesh.finalize()

print("Total number of cells", mesh.nC)
print("Total number of cell faces", mesh.n_faces)
print("Total number of cell edges", mesh.n_edges)

# Resistivity in Ohm m
res_background = 1.0
res_block = 100.0
conductivity_background = 1 / 100.0
conductivity_block = 1 / 1.0

# Find cells that are active in the forward modeling (cells below surface)
ind_active = surface2ind_topo(mesh, surface)

# Define mapping from model to active cells
model_map = maps.InjectActiveCells(mesh, ind_active, res_background)

# Define model. Models in SimPEG are vector arrays
model = res_background * np.ones(ind_active.sum())
ind_block = (
        (mesh.gridCC[ind_active, 0] <= 600.0)
        & (mesh.gridCC[ind_active, 0] >= 200.0)
        & (mesh.gridCC[ind_active, 1] <= 500.0)
        & (mesh.gridCC[ind_active, 1] >= -500.0)
        & (mesh.gridCC[ind_active, 2] <= -800.0)
        & (mesh.gridCC[ind_active, 2] >= -1000.0)
)
model[ind_block] = res_block

cells = mesh.cell_centers

search_area = cells[(cells[:, 0] > (0 - cell_width)) & (cells[:, 0] < (4000 + cell_width))
                    & (cells[:, 1] > (-500 - cell_width)) & (cells[:, 1] < (500 + cell_width))
                    & (cells[:, 2] > (-1000 - cell_width)) & (cells[:, 2] < (-800 + cell_width))]

num_iterations = 1
for i in range(0, num_iterations):
    cells_to_refine = estimate_error(mesh, survey, model_map, model, search_area, 'linear',
                                     frequencies[0], omegas[0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cells_to_refine[:, 0], cells_to_refine[:, 1], cells_to_refine[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    # refine cells


