"""
In this module, the velocity values are computed from the vorticity which is realized for two domains: half-plane 
and periodic channel. 
"""
import torch

import utils
from configs import parameters


def BS_kernel(x, y): 
    """
    The Biot-Savart kernel for the half-plane.

    Args: 
        x: a tensor of shape (n, 2) interpreted as n points on the plane.
        y: a tensor of shape (m, 2) interpreted as m points on the plane.

    Returns:
        kernel_values: a tensor of shape (n, m, 2) with the values of the kernel for each pair of points from x and y.
    """
    sig = parameters["kernel_singularity_cutoff"]

    X1 = torch.tensordot(x[:,0], torch.ones_like(y[:,0]), dims=0) 
    X2 = torch.tensordot(x[:,1], torch.ones_like(y[:,1]), dims=0)
    Y1 = torch.tensordot(torch.ones_like(x[:,0]), y[:,0], dims=0)
    Y2 = torch.tensordot(torch.ones_like(x[:,1]), y[:,1], dims=0)

    Sp = (X1-Y1)**2 + (X2+Y2)**2
    Sn = (X1-Y1)**2 + (X2-Y2)**2
    
    Moll_p = (Sp!=0)*(1.0-torch.exp(-Sp**2/sig))/(Sp+(Sp==0))
    Moll_n = (Sn!=0)*(1.0-torch.exp(-Sn**2/sig))/(Sn+(Sn==0))

    kernel_values = torch.zeros((len(x[:,0]), len(y[:,0]), 2))
    
    kernel_values[:,:,0] = ((Y2-X2)*Moll_n-(Y2+X2)*Moll_p)/(2*torch.pi)
    kernel_values[:,:,1] = (-(Y1-X1)*Moll_n+(Y1-X1)*Moll_p)/(2*torch.pi)

    return kernel_values


def Jacobi_upd_step(u_prev, b, h_1, h_2):
    """
    Performs one step of the Jacobi method update for the periodic channel, i.e. the upper and lower boundaries have the zero values
        and the left and right boundaries are periodized.

    Args:
        u_prev: a tensor with previous values of u with shape = (n, m).
        b: a tensor with the values of RHS of the equation with shape = (n, m) as u_prev.
        h_1: spacing in x_1 direction.
        h_2: spacing in x_2 direction.

    Returns:
        u_new: a tensor with new values of u with the same shape as u_prev.
    """
    u_new = torch.empty_like(u_prev)

    #update the values for the interior points
    u_new[1:-1,1:-1] = (((u_prev[1:-1,2:] + u_prev[1:-1,0:-2])*h_2**2 +
                      (u_prev[2:,1:-1] + u_prev[0:-2,1:-1])*h_1**2) / (2*(h_1**2 + h_2**2)) -
                      h_1**2*h_2**2 / (2*(h_1**2 + h_2**2))*b[1:-1,1:-1])

    #periodic boundary condition at x = H
    u_new[-1,1:-1] = (((u_prev[0,1:-1] + u_prev[-2,1:-1])*h_2**2 + (u_prev[-1,2:] + u_prev[-1,0:-2])*h_1**2) /
                   (2*(h_1**2 + h_2**2)) - h_1**2*h_2**2 / (2*(h_1**2 + h_2**2))*b[-1,1:-1])

    #periodic boundary condition at x = 0
    u_new[0,1:-1] = (((u_prev[1,1:-1] + u_prev[-1,1:-1])*h_2**2 + (u_prev[0,2:] + u_prev[0,0:-2])*h_1**2) /
                  (2*(h_1**2 + h_2**2)) - h_1**2*h_2**2 / (2*(h_1**2 + h_2**2))*b[0,1:-1])
        
    #no-slip boundary conditions at y = 0 and y = H
    u_new[:,-1] = torch.zeros_like(u_new[:,-1]) 
    u_new[:,0] = torch.zeros_like(u_new[:,0]) 

    return u_new


def u(x, omega, domain = parameters["domain"], lattice = None):
    """
    The function computing the velocity values at given points for specified domain; NB: for half-plane, use the Biot-Savart law 
        and for the periodic channel, use the Jacobi method.

    Args:
        x: a tensor of shape (., 2) interpreted as n points at which the velocity is to be computed.
        omega: a function representing the vorticity.
        domain: the name of the domain (default is taken from params).
        lattice: a tensor with lattice points to compute the velocity (if None, a custom lattice is created as in utils)
            with shape (., ., 2).

    Returns:
        vel: a tensor of shape (., 2) of values at points from x. 
    """
    if domain == "half-plane":
        #setup the lattice and mesh sizes
        if lattice == None: 
            lattice, h_0 = utils.create_lattice()
            #define meshsizes h_1 and h_2 to be h_0 for uniform notation
            h_1, h_2 = h_0, h_0 
        else: 
            h_1, h_2 = lattice[1,1,0] - lattice[0,0,0], lattice[1,1,1] - lattice[0,0,1]
        
        #compute the values of the vorticity over the lattice
        omega_values = omega(lattice).detach()

        #compute the Biot-Savart kernel values for each point in x and in lattice
        BS_kernel_values = BS_kernel(x, lattice)

        #compute the discretized integral 
        S = torch.tensordot(BS_kernel_values, omega_values, dims=([1],[0]))*h_1*h_2

        return torch.squeeze(S)

    elif domain == "channel":
        #setup the lattice and mesh sizes
        if lattice == None: 
            lattice, h_0 = utils.create_lattice(reshaped = True)
            #define meshsizes h_1 and h_2 to be h_0 for uniform notation
            h_1, h_2 = h_0, h_0 
        else: 
            h_1, h_2 = lattice[1,1,0] - lattice[0,0,0], lattice[1,1,1] - lattice[0,0,1]

        H = parameters["domain_size"]
        N = parameters["particles_number"]

        #tensors to store the values of the RHS, i.e. -curl(omega)
        omega_rhs_term_1, omega_rhs_term_2 = torch.empty_like(lattice[:,0]), torch.empty_like(lattice[:,1])

        #compute the values of the curl of vorticity over the lattice
        for i in range(len(lattice[:,0])):
            x1 = lattice[i,0]
            x2 = lattice[i,1]
            point = torch.tensor([x1,x2], requires_grad = True)
            omega_at_point = omega(point)
            omega_at_point.backward(retain_graph = True)
            omega_rhs_term_1[i] = -point.grad[1]
            omega_rhs_term_2[i] = point.grad[0]

        lattice = lattice.reshape((2*N+1, N+1, 2))
        omega_rhs_term_1, omega_rhs_term_2 = omega_rhs_term_1.reshape((2*N+1, N+1)), omega_rhs_term_2.reshape((2*N+1, N+1))

        #tensors storing components of the velocity 
        u1, u2 = torch.zeros_like(lattice[...,0]), torch.zeros_like(lattice[...,1])
    
        #iterate the method until the difference between the previous step and update is small
        diff = 1.
        while diff > parameters["Jacobi_iteration_precision"]:
            u1_temp = torch.clone(u1)
            u2_temp = torch.clone(u2)
            
            u1, u2 = Jacobi_upd_step(u1_temp,omega_rhs_term_1,h_1,h_2), Jacobi_upd_step(u2_temp,omega_rhs_term_2,h_1,h_2)

            diff = torch.mean((u1-u1_temp)**2+(u2-u2_temp)**2)

        #compute the velocity at points from x by finding their integer coordinates within lattice
        ind1, ind2 = ((x[:,0] + H) / h_1).int(), (x[:,1] / h_2).int()
        S = torch.empty_like(x)
        S[:,0] = (u1[ind1, ind2] + u1[(ind1+1)%(2*N+1), ind2] + u1[ind1, (ind2+1)%(N+1)] + u1[(ind1+1)%(2*N+1), (ind2+1)%(N+1)])/4
        S[:,1] = (u2[ind1, ind2] + u2[(ind1+1)%(2*N+1), ind2] + u2[ind1, (ind2+1)%(N+1)] + u2[(ind1+1)%(2*N+1), (ind2+1)%(N+1)])/4

        return S
        
    else:
        raise TypeError("Domain is not recognized.")
