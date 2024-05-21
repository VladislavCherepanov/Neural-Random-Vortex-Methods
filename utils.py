"""
A module storing for convenience some functions used in the other modules.
"""
import torch

from configs import parameters


def initial_vorticity(x):
    """
    A function representing the initial vorticity of the flow.

    params: 
        x: a tensor of arbitrary shape (..., 2) interpreted as a tensor of points of shape (...) on the plane.

    return:
        vorticity: a tensor of the shape (...), i.e. as x without the last dimension, of values of the initial vorticity at x.
    """
    W0 = parameters["initial_vorticity_scalar"]
    vorticity = W0*torch.cos(torch.pi*x[...,1]/3)
    return vorticity


def zeta(x, vorticity, domain = parameters["domain"]):
    """
    A function to compute the boundary vorticity values.

    Args:
        x: a tensor of arbitrary shape (..., 2) interpreted as tensor of points of shape (...) on the plane.
        vorticity: a function representing the vorticity to compute the boundary values from.
        domain: the name of the domain (default is taken from configs).

    Returns:
        boundary_vorticity: if domain is "half-plane", a tensor of shape (...) with values of the boundary vorticity at x;
            if domain is "channel", a tuple of two such tensors for the lower and upper boundaries.
    """
    if domain == "half-plane": 
        projected_points = torch.zeros_like(x)
        projected_points[...,0] = x[...,0]
        boundary_vorticity = vorticity(projected_points).detach()
        return boundary_vorticity
    elif domain == "channel": 
        H = parameters["domain_size"]
        projected_points_lower = torch.zeros_like(x)
        projected_points_lower[...,0] = x[...,0]
        boundary_vorticity_lower = vorticity(projected_points_lower).detach()
        projected_points_upper = H*torch.ones_like(x)
        projected_points_upper[...,0] = x[...,0]
        boundary_vorticity_upper = vorticity(projected_points_upper).detach()
        return boundary_vorticity_lower, boundary_vorticity_upper
    else:
        raise TypeError("Domain is not recognized.")


def cutoff(x, domain = parameters["domain"]):
    """
    The cutoff function for the perturbed vorticity (comes as the second derivative of 2*(x2-0.5)**3-1.5*(x2-0.5)+0.5).

    Args:
        x: a tensor of shape (..., 2) interpreted as tensor of points of shape (...) on the plane.
        domain: the name of the domain (default is taken from configs).

    Returns:
        cutoff_values: if domain is "half-plane", a tensor of shape (...) with values of the cutoff at points from x;
            if domain is "channel", a tuple of two such tensors for the lower and upper boundaries.
    """
    eps = parameters["boundary_cutoff"]
    if domain == "half-plane": 
        cutoff_values = 12*(x[...,1]/eps - 0.5)*(x[...,1] > 0).float()*(x[...,1] < eps).float()
        return cutoff_values
    elif domain == "channel": 
        H = parameters["domain_size"]
        cutoff_values_lower = 12*(x[...,1]/eps - 0.5)*(x[...,1] > 0).float()*(x[...,1] < eps).float() 
        cutoff_values_upper = 12*((H - x[...,1])/eps - 0.5)*(x[...,1] < H).float()*(x[...,1] > H - eps).float() 
        return cutoff_values_lower, cutoff_values_upper
    else:
        raise TypeError("Domain is not recognized.")


def external_force(x):
    """
    The external force curl G(x, t) acting as the external vorticity in the vorticity equation (note: time-independent version realized).

    Args:
        x: a tensor of arbitrary shape (..., 2) interpreted as tensor of points of shape (...) on the plane.
        t: a float number representing the time.

    Returns:
        value: a tensor of shape (...) of values at points from x.
    """
    G0 = parameters["external_vorticity_force_scalar"]
    value = G0*torch.ones_like(x[...,0])
    return value


def create_lattice(
        N = parameters["particles_number"],
        H = parameters["domain_size"],
        reshaped = True
        ):
    """
    A function to create a custom lattice for the box [-H, H]x[0, H].

    Args:
        N: number of subintervals within the segment [0, H] (default is taken from configs).
        H: the size of the domain (default is taken from configs).
        reshaped: if True, returns as a tensor of shape (., 2).

    Return:
        lattice_points: a tensor of lattice points with shape (., 2) or (., ., 2).
        h_0: the meshsize of the lattice.
    """
    lattice_points = torch.empty(2*N+1, N+1, 2)
    lattice_points[:,:,0], lattice_points[:,:,1] = torch.meshgrid(torch.linspace(-H, H, 2*N+1), 
                                                                  torch.linspace(0, H, N+1))
    h_0 = H/N 
    if reshaped == True: return torch.reshape(lattice_points, ((2*N+1)*(N+1), 2)), h_0
    elif reshaped == False: return lattice_points, h_0
