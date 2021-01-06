#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:25:20 2021

@author: larslaheij
"""
import emg3d
import numpy as np

src = [0, 0, 0, 0, 0]        # x-dir. source at the origin, 50 m above seafloor
off = np.arange(20, 41)*100  # Offsets
rec = [off, off*0, 0]        # In-line receivers on the seafloor
freq = 1.0                   # Frequency (Hz)

res_background = 1 #Resistivity of the background
res_block = 100 #Resistivity of the block

grid = emg3d.construct_mesh(
        frequency=freq,
        min_width_limits=100,
        properties=res_background,
        center=(0, 0, 0),
        domain=([-500, 4500], [-1000, 1000], [-1200, 200]),
        verb=0,
)

# Background model
model_background = emg3d.Model(grid, property_x=res_background*np.ones(grid.vnC))

# Block model
res = res_background*np.ones(grid.vnC).ravel('F')
xx = (grid.gridCC[:, 0] >= 200) & (grid.gridCC[:, 0] <= 600)
yy = (grid.gridCC[:, 1] >= -500) & (grid.gridCC[:, 1] <= 500)
zz = (grid.gridCC[:, 2] > -1000)*(grid.gridCC[:, 2] < -800)

res[xx*yy*zz] = res_block
model_block = emg3d.Model(grid, property_x=res)

solver_opts = {
        'verb': 2, 'sslsolver': True,
        'semicoarsening': True, 'linerelaxation': True
}

sfield = emg3d.get_source_field(grid, src, freq, strength=0)
efield_backg = emg3d.solve(grid, model_background, sfield, **solver_opts)
efield_block = emg3d.solve(grid, model_block, sfield, **solver_opts)

# Extract responses at receiver locations.
rec_lg_backg = emg3d.get_receiver(grid, efield_backg.fx, tuple(rec))
rec_lg_block = emg3d.get_receiver(grid, efield_block.fx, tuple(rec))