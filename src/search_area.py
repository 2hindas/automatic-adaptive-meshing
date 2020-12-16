#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:11:19 2020

@author: larslaheij
"""
def search_area_object(mesh,objct,factor=2):
    #Factor determines the search range around the object
    #Search area is always a rectangle, regardless of the shape of the object
    cells = mesh.cell_centers
    cell_width_X = min(mesh.h[0]) #minimum cell width in x-direction
    cell_width_Y = min(mesh.h[1]) #minimum cell width in y-direction
    cell_width_Z = min(mesh.h[2]) #minimum cell width in z-direction
    Left_X = min(objct[:,0]) #Left X
    Right_X = max(objct[:,0]) #Right X
    Left_Y = min(objct[:,1]) #Left Y
    Right_Y = max(objct[:,1]) #Right Y
    Lower_Z = min(objct[:,2]) #Lower Z
    Upper_Z = max(objct[:,2]) #Upper Z
    search_area = cells[(cells[:, 0] > (Left_X - factor*cell_width_X)) & (cells[:, 0] < (Right_X + factor*cell_width_X))
                    & (cells[:, 1] > (Left_Y - factor*cell_width_Y)) & (cells[:, 1] < (Right_Y + factor*cell_width_Y))
                    & (cells[:, 2] > (Lower_Z - factor*cell_width_Z)) & (cells[:, 2] < (Upper_Z + factor*cell_width_Z))]
    
    return search_area
def search_area_receivers(mesh,receiver_locations,factor=3):
    #Factor determines the search range around the object
    #Search area is always a rectangle, regardless of the shape of the object
    cells = mesh.cell_centers
    cell_width_X = min(mesh.h[0]) #minimum cell width in x-direction
    cell_width_Y = min(mesh.h[1]) #minimum cell width in y-direction
    cell_width_Z = min(mesh.h[2]) #minimum cell width in z-direction
    Left_X = min(receiver_locations[:,0]) #Left X
    Right_X = max(receiver_locations[:,0]) #Right X
    Left_Y = min(receiver_locations[:,1]) #Left Y
    Right_Y = max(receiver_locations[:,1]) #Right Y
    Lower_Z = min(receiver_locations[:,2]) #Lower Z
    Upper_Z = max(receiver_locations[:,2]) #Upper Z
    search_area = cells[(cells[:, 0] > (Left_X - factor*cell_width_X)) & (cells[:, 0] < (Right_X + factor*cell_width_X))
                    & (cells[:, 1] > (Left_Y - factor*cell_width_Y)) & (cells[:, 1] < (Right_Y + factor*cell_width_Y))
                    & (cells[:, 2] > (Lower_Z - factor*cell_width_Z)) & (cells[:, 2] < (Upper_Z + factor*cell_width_Z))]
    
    return search_area
    