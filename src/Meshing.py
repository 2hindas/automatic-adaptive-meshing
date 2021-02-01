import numpy as np
import matplotlib.pyplot as plt
from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz
from scipy.spatial.transform import Rotation as R


def create_box_surface(coordinates, cellwidth, axis='x', degree_rad=0):
    """Creates a list of coordinates of points on the surface of a box.

    Parameters
    ----------
    coordinates : tuple
        ((xmin, xmax),(ymin, ymax),(zmin, zmax)) coordinates of box
    cellwidth : int
        width of smallest cell in the mesh
    axis : string
        'x', 'y' or 'z' to denote the axis of rotation
    degree_rad : float
        the number of degrees to rotate the cube with


    Returns
    -------
    np.ndarray
        a numpy array of the box surface coordinates
    """

    x1 = coordinates[0][0]
    x2 = coordinates[0][1]
    y1 = coordinates[1][0]
    y2 = coordinates[1][1]
    z1 = coordinates[2][0]
    z2 = coordinates[2][1]

    x_num = int(np.ceil((abs(x1-x2)) / cellwidth))
    y_num = int(np.ceil((abs(y1-y2)) / cellwidth))
    z_num = int(np.ceil((abs(z1-z2)) / cellwidth))
    
    x_coords = np.linspace(x1, x2, x_num)
    y_coords = np.linspace(y1, y2, y_num)
    z_coords = np.linspace(z1, z2, z_num)

    xp, yp, zp = np.meshgrid(x_coords, y_coords, z_coords)
    xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
    surface = []
    for row in xyz:
        if row[0] == x1 or row[0] == x2 or row[1] == y1 or row[1] == y2 or row[2] == z1 or row[2] == z2:
            surface.append(row)

    rotation = R.from_euler(axis, degree_rad, degrees=True).as_matrix()
    surface = np.asarray([np.dot(rotation, i) for i in surface])
    return np.asarray(surface)


def create_sphere_surface(origin, radius, num_points):
    """Creates a list of coordinates of points on the surface of a sphere.

        Parameters
        ----------
        origin : tuple
            (x0, y0, z0) origin of the sphere
        radius : float
            radius of the sphere
        num_points : int
            the density of the sphere surface mesh


        Returns
        -------
        np.ndarray
            a numpy array of the sphere surface coordinates
    """

    theta_range = np.linspace(0, np.pi, num_points)
    phi_range = np.linspace(0, 2*np.pi, 2 * num_points)
    surface = []

    for theta in theta_range:
        for phi in phi_range:
            x = radius * np.sin(theta) * np.cos(phi) + origin[0]
            y = radius * np.sin(theta) * np.sin(phi) + origin[1]
            z = radius * np.cos(theta) + origin[2]
            surface.append([x, y, z])

    return np.asarray(surface)


def create_octree_mesh(domain, cellwidth, points, method='surface'):
    """Creates an octree mesh and refines at specified points.

        Parameters
        ----------
        domain : tuple
            ((xmin, xmax),(ymin, ymax),(zmin, zmax)) coordinates of the domain
        cellwidth : int
            width of smallest cell in the mesh
        points : np.ndarray
            array of points that will be refined in the mesh
        method : string
            the discretize method that will be used for refinement. Available are 'surface',
            'radial' and 'box'


        Returns
        -------
        discretize.mesh
            a mesh where the points are refined
    """

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
    mesh = refine_tree_xyz(mesh, points, octree_levels=[1, 1, 1], method=method, max_distance=1,
                           finalize=False)

    return mesh


def refine_at_locations(mesh, locations):
    """Refines a mesh at the given locations.

         Parameters
         ----------
         mesh : discretize.mesh
             a not yet finalized discretize mesh
         locations : np.ndarray
             an array with coordinates of points to refine the mesh

         Returns
         -------
         discretize.mesh
             a mesh where the new points are also refined
    """

    mesh = refine_tree_xyz(
        mesh, locations, octree_levels=[1, 1], method="radial", finalize=False
    )
    return mesh


def plot_mesh_slice(mesh, axis, index, save=False):
    """Plots a single slice of a mesh.

         Parameters
         ----------
         mesh : discretize.mesh
             a finalized discretize mesh
         axis : str
             the axis that is the normal of a slice to be shown
         index : int
             the index of the slice in the mesh to plot
         save : boolean
             whether the slice is saved to a figure or shown live, default is False.
    """

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    v = mesh.cell_volumes
    mesh.plot_slice(np.log10(v), normal=axis.upper(), ax=ax, ind=index, grid=True)
    ax.set_title(f"Cell Log-Volumes at {axis.upper()} = {index}")
    if save:
        plt.savefig(f"Block {axis.upper()} = {index}")
    else:
        plt.show()
