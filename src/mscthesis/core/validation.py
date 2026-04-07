from __future__ import annotations

# implement validation pipeline


""" 
1. Mesh provided brep for a range of resolutions h (factor in (1.0, ...)) [save meshes] 
2. Compute CG1 and CG2 solutions for all resolutions [save solutions, gradients]
3. save as two dataframes 
4. generate 4 plots:
    a.  Show QoI(C_h) for CG1 and CG2 in the same plot [two panel: conc, flux]
    b.  Show discrepancy in J(C_h) and J(grad C_h) for CG1 and CG2 [two panel: conc, flux]
    c . Show convergence of QoI(C_h) towards QoI(C_h_min)_CG2 for CG1 and CG2 [two panel: conc, flux]
    d.  Show C_h similar for CG1 and CG2 in plane, but grad C_h different [4 panel]
5. save plots and exit
"""
