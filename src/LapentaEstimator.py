#!/usr/bin/env python3


import numpy as np
import SimPEG.electromagnetics.frequency_domain as fdem
from scipy.interpolate import Rbf, LinearNDInterpolator, NearestNDInterpolator
import numdifftools as nd

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


def interpolate_rbf(x, y, z, x_val, y_val, z_val):
    x_curl_interpolated = Rbf(x[:, 0], x[:, 1], x[:, 2], x_val)
    y_curl_interpolated = Rbf(y[:, 0], y[:, 1], y[:, 2], y_val)
    z_curl_interpolated = Rbf(z[:, 0], z[:, 1], z[:, 2], z_val)

    return x_curl_interpolated, y_curl_interpolated, z_curl_interpolated


def interpolate_nearest(x, y, z, x_val, y_val, z_val):
    x_curl_interpolated = NearestNDInterpolator(x, x_val)
    y_curl_interpolated = NearestNDInterpolator(y, y_val)
    z_curl_interpolated = NearestNDInterpolator(z, z_val)

    return x_curl_interpolated, y_curl_interpolated, z_curl_interpolated


def interpolate_linear(x, y, z, x_val, y_val, z_val):
    x_curl_interpolated = LinearNDInterpolator(x, x_val)
    y_curl_interpolated = LinearNDInterpolator(y, y_val)
    z_curl_interpolated = LinearNDInterpolator(z, z_val)

    return x_curl_interpolated, y_curl_interpolated, z_curl_interpolated


def estimate_curl_electric_field(mesh, survey, model_map, model, interpolation='rbf', frequency=1.0,
                                 omega=2 * np.pi):
    x_faces = mesh.faces_x
    y_faces = mesh.faces_y
    z_faces = mesh.faces_z

    x_edges = mesh.edges_x
    y_edges = mesh.edges_y
    z_edges = mesh.edges_z

    # Solution by forward modelling for magnetic flux density and electric field
    # This uses a rhoMap (resistivity)
    simulation = fdem.simulation.Simulation3DMagneticFluxDensity(
        mesh, survey=survey, rhoMap=model_map, Solver=Solver
    )

    simulationelectricfield = fdem.simulation.Simulation3DElectricField(
        mesh, survey=survey, rhoMap=model_map, Solver=Solver
    )

    # Compute magnetic flux density
    fields = simulation.fields(model)
    magnetic_flux_density = fields[:, 'bSolution']

    # Source field
    sources = simulation.getSourceTerm(frequency)
    Sm = sources[0]

    # Curl of Electric field computed on the cell faces
    curl = Sm - 1j * omega * magnetic_flux_density
    curl = np.reshape(curl, len(curl))

    x_curl = curl[0:mesh.n_faces_x]
    y_curl = curl[mesh.n_faces_x:mesh.n_faces_x + mesh.n_faces_y]
    z_curl = curl[mesh.n_faces_x + mesh.n_faces_y:mesh.n_faces_x + mesh.n_faces_y + mesh.n_faces_z]

    if interpolation == 'rbf':
        interpolator = interpolate_rbf
    elif interpolation == 'linear':
        interpolator = interpolate_linear
    else:
        interpolator = interpolate_nearest

    curl_x_inter, curl_y_inter, curl_z_inter = interpolator(x_faces, y_faces, z_faces, x_curl,
                                                            y_curl, z_curl)

    # Electric field solution
    fieldselectric = simulationelectricfield.fields(model)
    EF = fieldselectric[:, 'eSolution']
    EF = np.reshape(EF, len(EF))

    EF_x = EF[0:mesh.n_edges_x]
    EF_y = EF[mesh.n_edges_x:mesh.n_edges_x + mesh.n_edges_y]
    EF_z = EF[
           mesh.n_edges_x + mesh.n_edges_y:mesh.n_edges_x + mesh.n_edges_y + mesh.n_edges_z]

    EF_x_inter, EF_y_inter, EF_z_inter = interpolator(x_edges, y_edges, z_edges, EF_x, EF_y, EF_z)

    return curl_x_inter, curl_y_inter, curl_z_inter, EF_x_inter, EF_y_inter, EF_z_inter


def compute_cell_error(cell, curl_x, curl_y, curl_z, ef_x, ef_y, ef_z):

    def ef_interpolator(x):
        return np.array([ef_x(*x), ef_y(*x), ef_z(*x)])

    jacobian = nd.Jacobian(ef_interpolator)(cell)
    curl = np.array([jacobian[2, 1] - jacobian[1, 2], jacobian[0, 2] -
                     jacobian[2, 0], jacobian[1, 0] - jacobian[0, 1]])

    curl_field = np.array([curl_x(*cell), curl_y(*cell), curl_z(*cell)])
    error = np.linalg.norm(curl_field - curl)
    return error


def estimate_error(mesh, survey, model_map, model, search_area, interpolation='rbf', frequency=1.0,
                   omega=2 * np.pi, refine_percentage=0.05):
    curl_x, curl_y, curl_z, ef_x, ef_y, ef_z = estimate_curl_electric_field(mesh, survey, model_map,
                                                                            model, interpolation,
                                                                            frequency, omega)

    cell_errors = []
    for cell in search_area:
        error = compute_cell_error(cell, curl_x, curl_y, curl_z, ef_x, ef_y, ef_z)
        cell_errors.append(error)

    np.save('error.npy', np.asarray(cell_errors))
    n_refine_cells = int(np.ceil(refine_percentage * len(search_area)))
    cells_to_refine = search_area[np.argpartition(cell_errors, -n_refine_cells)[-n_refine_cells:]]
    return cells_to_refine
