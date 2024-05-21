"""
In this module, a dictionary with parameters for the simulation is stored.
"""

parameters = {
    "kernel_singularity_cutoff": 0.01, 
    "boundary_cutoff": .3, 
    "domain_size": 6., 
    "particles_number": 40, 
    "steps_number": 100,
    "total_time": 10.0,
    "viscosity_constant": 0.01,
    "domain": "half-plane",
    "initial_vorticity_scalar": 50.,
    "external_vorticity_force_scalar": 0.,
    "epochs_per_iteration": 500,
    "hidden_dimension_for_model": 512,
    "Jacobi_iteration_precision": 1.e-05
    }
