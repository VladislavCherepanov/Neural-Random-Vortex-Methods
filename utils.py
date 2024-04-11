import numpy as np
import torch
import params


def initial_vorticity(x):
    """
    Function to define the initial vorticity.

    params: 
        x: a tensor of arbitrary shape (..., 2) interpreted as tensor of points of shape (...) on the plane.

    return:
        vorticity: a tensor of the shape (...), i.e. as x without the last dimension, of values of the initial vorticity at x.
    """
    vorticity = params.W0*torch.sin(x[...,0])*torch.cos(x[...,1])

    return vorticity

#function for the initial boundary vorticity
#NB: for convenience computed for an array of points in 2d
def theta(vorticity, x):
    projected_points = torch.zeros_like(x)
    projected_points[...,0] = x[...,0]
    return vorticity(projected_points)

#cutoff function for the perturbed vorticity
#NB: comes as the second derivative of 2*(x2-0.5)**3-1.5*(x2-0.5)+0.5
def cutoff(x):
    return 12*(x[...,1]/params.eps - 0.5)*(x[...,1] > 0).float()*(x[...,1] < params.eps).float()

#function for the external force curl
def external_force(x, t):
    return params.G0*torch.ones_like(x[:,0])


def create_lattice(
        N = params.N,
        H = params.H,
        reshaped = True
        ):
    """
    A function to create a custom lattice for the box [-H, H]x[0, H].

    Args:
        N: number of subintervals within the segment [0, H] (default is taken from params.py).
        H: the size of the domain (default is taken from params.py).
        reshaped: if True, returns as a tensor of shape (:, 2).

    Return:
        lattice_points: a tensor of lattice points with shape (:, 2) or (:, :, 2).
        h_0: the meshsize of the lattice.
    """
    lattice_points = torch.empty(2*N+1, N+1, 2)
    lattice_points[:,:,0], lattice_points[:,:,1] = torch.meshgrid(torch.linspace(-H, H, 2*N+1), 
                                                                  torch.linspace(0, H, N+1))
    h_0 = H/N 
    if reshaped == True: return torch.reshape(lattice_points, ((2*N+1)*(N+1), 2)), h_0
    elif reshaped == False: return lattice_points, h_0
