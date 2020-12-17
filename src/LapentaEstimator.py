#!/usr/bin/env python3


import numpy as np
import SimPEG.electromagnetics.frequency_domain as fdem
from scipy.interpolate import Rbf, LinearNDInterpolator, NearestNDInterpolator
import numdifftools as nd
from src.Meshing import refine_at_locations, create_octree_mesh
from SimPEG.utils import surface2ind_topo
from src.Setup import search_area_receivers, search_area_object
from SimPEG import maps

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver


# Input of this function are the x-,y- and z-faces and the curl values of the electric field on the faces
# Or x-,y- and z-edges and the values of the electric field
# Radial basis function interpolation
def interpolate_rbf(x, y, z, x_val, y_val, z_val):
    x_interpolated = Rbf(x[:, 0], x[:, 1], x[:, 2], x_val)
    y_interpolated = Rbf(y[:, 0], y[:, 1], y[:, 2], y_val)
    z_interpolated = Rbf(z[:, 0], z[:, 1], z[:, 2], z_val)

    return x_interpolated, y_interpolated, z_interpolated


# Input of this function are the x-,y- and z-faces and the curl values of the electric field on the faces
# Or x-,y- and z-edges and the values of the electric field
# Nearest neighbour interpolation
def interpolate_nearest(x, y, z, x_val, y_val, z_val):
    x_interpolated = NearestNDInterpolator(x, x_val)
    y_interpolated = NearestNDInterpolator(y, y_val)
    z_interpolated = NearestNDInterpolator(z, z_val)

    return x_interpolated, y_interpolated, z_interpolated


# Input of this function are the x-,y- and z-faces and the curl values of the electric field on the faces
# Or x-,y- and z-edges and the values of the electric field
# Linear interpolation
def interpolate_linear(x, y, z, x_val, y_val, z_val):
    x_interpolated = LinearNDInterpolator(x, x_val)
    y_interpolated = LinearNDInterpolator(y, y_val)
    z_interpolated = LinearNDInterpolator(z, z_val)

    return x_interpolated, y_interpolated, z_interpolated


def estimate_curl_electric_field(mesh, survey, model_map, model, interpolation='rbf', frequency=1.0,
                                 omega=2 * np.pi, parameter='resistivity'):
    x_faces = mesh.faces_x
    y_faces = mesh.faces_y
    z_faces = mesh.faces_z

    x_edges = mesh.edges_x
    y_edges = mesh.edges_y
    z_edges = mesh.edges_z

    # Solution by forward modelling for magnetic flux density and electric field
    # This uses a rhoMap (resistivity)
    if parameter == 'resistivity':
        simulation = fdem.simulation.Simulation3DMagneticFluxDensity(
            mesh, survey=survey, rhoMap=model_map, Solver=Solver
        )

        simulationelectricfield = fdem.simulation.Simulation3DElectricField(
            mesh, survey=survey, rhoMap=model_map, Solver=Solver
        )
    # This uses a sigmaMap (conductivity)
    else:
        simulation = fdem.simulation.Simulation3DMagneticFluxDensity(
            mesh, survey=survey, sigmaMap=model_map, Solver=Solver
        )

        simulationelectricfield = fdem.simulation.Simulation3DElectricField(
            mesh, survey=survey, sigmaMap=model_map, Solver=Solver
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
    jacobian[np.isnan(jacobian)] = 0  # handle NaN-values in the jacobian
    curl = np.array([jacobian[2, 1] - jacobian[1, 2], jacobian[0, 2] -
                     jacobian[2, 0], jacobian[1, 0] - jacobian[0, 1]])

    curl_field = np.array([curl_x(*cell), curl_y(*cell), curl_z(*cell)])
    error = np.linalg.norm(curl_field - curl)
    return error


def estimate_error(mesh, survey, model_map, model, search_area, curl_x, curl_y, curl_z
                   , ef_x, ef_y, ef_z
                   , interpolation='rbf', frequency=1.0,
                   omega=2 * np.pi, refine_percentage=0.05, parameter='resistivity'):
    cell_errors = []
    for cell in search_area:
        error = compute_cell_error(cell, curl_x, curl_y, curl_z, ef_x, ef_y, ef_z)
        cell_errors.append(error)

    np.save('error.npy', np.asarray(cell_errors))
    n_refine_cells = int(np.ceil(refine_percentage * len(search_area)))
    cells_to_refine = search_area[np.argpartition(cell_errors, -n_refine_cells)[-n_refine_cells:]]
    return cells_to_refine


def iterator(mesh, domain, surface, cell_width, objct, create_surface
             , x_object, y_object, z_object
             , receiver_locations, source_locations, survey, par_background, par_object,
             model_map, model, ind_object, frequency=1, omega=2 * np.pi
             , parameter='resistivity', interpolation='rbf', type_object='box'
             , lim_iterations=5, factor_object=2, factor_receiver=3
             , refine_percentage=0.05, axis='x', degrees_rad=0, radius=1):
    diff = 10
    i = 0

    def ef_interpolator(x):
        return np.array([ef_x(*x), ef_y(*x), ef_z(*x)])

    def ef_old_interpolator(x):
        return np.array([ef_old_x(*x), ef_old_y(*x), ef_old_z(*x)])

    # -------------------------------------------------------------------------------------------------
    while diff > 0.01 and i < lim_iterations:
        # -------------------------------------------------------------------------------------------------
        # Define search areas
        search_area_obj = search_area_object(mesh, objct, factor=factor_object)
        search_area_receiv = search_area_receivers(mesh, receiver_locations,
                                                         factor=factor_receiver)
        # Interpolate curl and electric field
        curl_x, curl_y, curl_z, ef_x, ef_y, ef_z = estimate_curl_electric_field(mesh, survey,
                                                                                model_map,
                                                                                model,
                                                                                interpolation=interpolation,
                                                                                frequency=frequency,
                                                                                omega=omega
                                                                                ,
                                                                                parameter=parameter)
        # Compare electric field values until relative difference falls below 1%
        if i > 0:
            relative_difference_Efield = []
            for cell in search_area_obj:
                # This equation is sensitive to catastrophic failure
                form = np.abs(
                    (ef_old_interpolator(cell) - ef_interpolator(cell)) / ef_old_interpolator(cell))
                relative_difference_Efield.append(np.linalg.norm(form))
            for cell in search_area_receiv:
                # This equation is sensitive to catastrophic failure
                form = np.abs(
                    (ef_old_interpolator(cell) - ef_interpolator(cell)) / ef_old_interpolator(cell))
                relative_difference_Efield.append(np.linalg.norm(form))
            diff = max(relative_difference_Efield)
            print("Maximum relative difference is ", diff)
        ef_old_x = ef_x
        ef_old_y = ef_y
        ef_old_z = ef_z

        # Define cells to refine near object
        cells_to_refine_object = estimate_error(mesh, survey, model_map, model, search_area_obj
                                                , curl_x, curl_y, curl_z
                                                , ef_x, ef_y, ef_z
                                                , interpolation=interpolation
                                                , frequency=frequency, omega=omega
                                                , refine_percentage=refine_percentage
                                                , parameter=parameter)
        # Define cells to refine near receivers
        cells_to_refine_receivers = estimate_error(mesh, survey, model_map, model,
                                                   search_area_receiv
                                                   , curl_x, curl_y, curl_z
                                                   , ef_x, ef_y, ef_z
                                                   , interpolation=interpolation
                                                   , frequency=frequency, omega=omega
                                                   , refine_percentage=refine_percentage
                                                   , parameter=parameter)
        # Refine the mesh
        mesh = create_octree_mesh(domain, cell_width, objct, 'surface')
        if type_object == 'box':
            surface_object = create_surface(x_object, y_object, z_object, axis, degrees_rad)
        if type_object == 'sphere':
            surface_object = create_surface(x_object, y_object, z_object, radius)
        else:
            pass
        refine_at_locations(mesh, surface_object)
        refine_at_locations(mesh, source_locations)
        refine_at_locations(mesh, receiver_locations)
        refine_at_locations(mesh, cells_to_refine_object)
        refine_at_locations(mesh, cells_to_refine_receivers)
        mesh.finalize()

        # Find cells that are active in the forward modeling (cells below surface)
        ind_active = surface2ind_topo(mesh, surface)

        # Define mapping from model to active cells
        model_map = maps.InjectActiveCells(mesh, ind_active, par_background)

        # Define model. Models in SimPEG are vector arrays
        model = par_background * np.ones(ind_active.sum())
        model[ind_object(mesh, ind_active)] = par_object
        i += 1
    return mesh