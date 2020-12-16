#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 19:30:30 2020

@author: larslaheij
"""
import numpy as np
import Meshing as M
from SimPEG.utils import surface2ind_topo
from SimPEG import maps
from LapentaEstimator import estimate_error, estimate_curl_electric_field
import search_area as sa

def iterator(mesh,domain,surface,cell_width,objct,create_surface
             ,x_object,y_object,z_object
             ,receiver_locations,source_locations,survey,par_background,par_object,
             model_map,model,ind_object,frequency=1,omega=2*np.pi
             ,parameter='resistivity',interpolation='rbf',type_object='box'
             ,lim_iterations=5,factor_object=2,factor_receiver=3
             ,refine_percentage=0.05,axis='x',degrees_rad=0,radius=1):
    diff = 10
    i = 0
    def ef_interpolator(x):
        return np.array([ef_x(*x), ef_y(*x), ef_z(*x)])
    def ef_old_interpolator(x):
        return np.array([ef_old_x(*x), ef_old_y(*x), ef_old_z(*x)])
    #-------------------------------------------------------------------------------------------------
    while diff>0.01 and i < lim_iterations:
    #-------------------------------------------------------------------------------------------------
        #Define search areas
        search_area_object = sa.search_area_object(mesh, objct,factor=factor_object)
        search_area_receivers = sa.search_area_receivers(mesh, receiver_locations,factor=factor_receiver)
        #Interpolate curl and electric field
        curl_x, curl_y, curl_z, ef_x, ef_y, ef_z = estimate_curl_electric_field(mesh, survey, model_map,
                                                                            model, interpolation=interpolation,
                                                                            frequency=frequency, omega=omega
                                                                            , parameter=parameter)
        #Compare electric field values until relative difference falls below 1%
        if i>0:
            relative_difference_Efield = []
            for cell in search_area_object:
                #This equation is sensitive to catastrophic failure
                form = np.abs((ef_old_interpolator(cell)-ef_interpolator(cell))/ef_old_interpolator(cell))
                relative_difference_Efield.append(np.linalg.norm(form))
            for cell in search_area_receivers:
                #This equation is sensitive to catastrophic failure
                form = np.abs((ef_old_interpolator(cell)-ef_interpolator(cell))/ef_old_interpolator(cell))
                relative_difference_Efield.append(np.linalg.norm(form))
            diff = max(relative_difference_Efield)
            print("Maximum relative difference is ",diff)
        ef_old_x = ef_x
        ef_old_y = ef_y
        ef_old_z = ef_z
            
        #Define cells to refine near object
        cells_to_refine_object = estimate_error(mesh, survey, model_map, model, search_area_object
                                                , curl_x, curl_y, curl_z
                                                , ef_x, ef_y, ef_z
                                                ,interpolation=interpolation
                                                ,frequency=frequency, omega=omega
                                                ,refine_percentage=refine_percentage
                                                ,parameter=parameter)
        #Define cells to refine near receivers
        cells_to_refine_receivers = estimate_error(mesh, survey, model_map, model, search_area_receivers
                                                , curl_x, curl_y, curl_z
                                                , ef_x, ef_y, ef_z
                                                ,interpolation=interpolation
                                                ,frequency=frequency, omega=omega
                                                ,refine_percentage=refine_percentage
                                                ,parameter=parameter)
        #Refine the mesh
        mesh = M.create_octree_mesh(domain, cell_width, objct, 'surface')
        if type_object=='box':
            surface_object = create_surface(x_object,y_object,z_object,axis,degrees_rad)
        if type_object=='sphere':
            surface_object = create_surface(x_object,y_object,z_object,radius)
        else:
            pass
        M.refine_at_locations(mesh, surface_object)
        M.refine_at_locations(mesh, source_locations)
        M.refine_at_locations(mesh, receiver_locations)
        M.refine_at_locations(mesh, cells_to_refine_object)
        M.refine_at_locations(mesh, cells_to_refine_receivers)
        mesh.finalize()
    
        # Find cells that are active in the forward modeling (cells below surface)
        ind_active = surface2ind_topo(mesh, surface)
        
        # Define mapping from model to active cells
        model_map = maps.InjectActiveCells(mesh, ind_active, par_background)
        
        # Define model. Models in SimPEG are vector arrays
        model = par_background * np.ones(ind_active.sum())
        model[ind_object(mesh,ind_active)] = par_object
        i+= 1
    return mesh
    
    