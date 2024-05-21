import numpy as np
import torch
import matplotlib.pyplot as plt

import utils 
import vorticity_model as vort
import update as upd
from configs import parameters

from tqdm import tqdm

torch.set_num_threads(12)

if __name__ == "__main__":
    H = parameters["domain_size"] 
    N = parameters["particles_number"] 
    N_steps =  parameters["steps_number"]
    T = parameters["total_time"] 
    dt = T/N_steps #time increment
    nu = parameters["viscosity_constant"]
    eps = parameters["boundary_cutoff"] 
    domain = parameters["domain"]

    #initialise lattice points for the domain
    lattice_points, h_0 = utils.create_lattice()

    #indicators of hitting the boundary
    if domain == "channel": tau = (lattice_points[:,1] > 0).float()*(lattice_points[:,1] < H).float()
    elif domain == "half-plane": tau = (lattice_points[:,1] > 0).float()

    #initialize the particles at lattice points
    X = torch.clone(lattice_points)

    #print the initial vorticity
    vort.print_vorticity(omega = utils.initial_vorticity)

    #tensors to store the cumulative external vorticity from perturbation and force
    if domain == "channel": 
        cutoff_values_lower, cutoff_values_upper = utils.cutoff(X)
        boundary_vorticity_lower, boundary_vorticity_upper = utils.zeta(X, utils.initial_vorticity)
        ext_vort_pert = nu/(eps**2)*(boundary_vorticity_lower*cutoff_values_lower + boundary_vorticity_upper*cutoff_values_upper)*dt
    elif domain == "half-plane": 
        ext_vort_pert = nu/(eps**2)*utils.zeta(X, utils.initial_vorticity)*utils.cutoff(X)*dt
    ext_vort_force = utils.external_force(X)*dt

    #print the initial stream
    vort.print_stream(omega = utils.initial_vorticity)

    for t in tqdm(range(N_steps)):
        upd.step_update(X, tau, ext_vort_pert, ext_vort_force)
