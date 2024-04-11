"""
In this module, the velocity values are computed from the vorticity which  is realized for two domains: half-plane 
and periodic channel. 
"""
import torch

import utils
import params









def BS_kernel(x, y, sig = params.sig): 
    """
    The Biot-Savart kernel for the half-plane.

    Args: 
        x: a tensor of shape (n, 2) interpreted as n points.
        y: a tensor of shape (m, 2) interpreted as m points.

    Returns:
        kernel_values: a tensor of shape (n, m, 2) with the values of the kernel for each pair of points from x and y.
    """
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


def u(x, omega, domain, lattice = None):
    """
    The function computing the velocity values at given points for specified domain; NB: for half-plane, use the Biot-Savart
    law and for the periodic channel, use the Jacobi method.

    Args:
        x: a tensor of shape (n, 2) interpreted as n points at which the velocity is computed.
        omega: a function representing the vorticity.
        domain: the name of the domain (realized: "half-plane", "channel").
        lattice: a tensor with lattice points to compute the velocity (if None, a custom lattice is created as in utils.py)

    Returns:
        vel: a tensor of values at points from x. 
    """
    if domain == "half-plane":
        #setup the lattice and mesh sizes
        if lattice == None: 
            lattice, h_0 = utils.create_lattice()
            #define meshsizes h_1 and h_2 to be h_0 for uniform notation
            h_1, h_2 = h_0, h_0 
        else: 
            h_1, h_2 = lattice[1,0] - lattice[0,0], lattice[1,1] - lattice[0,1]
        
        #compute the values of the vorticity over the lattice
        omega_values = omega(lattice) 

        #compute the Biot-Savart kernel values for each point in x and in lattice
        BS_kernel_values = BS_kernel(x, lattice)

        #compute the discretized integral 
        S = torch.tensordot(BS_kernel_values, omega_values, dims=([1],[0]))*h_1*h_2

        return torch.squeeze(S)

    elif domain == "channel":
        #setup the lattice and mesh sizes
        if lattice == None: 
            lattice, h_0 = utils.create_lattice(reshaped = False)
            #define meshsizes h_1 and h_2 to be h_0 for uniform notation
            h_1, h_2 = h_0, h_0 
        else: 
            h_1, h_2 = lattice[1,0] - lattice[0,0], lattice[1,1] - lattice[0,1]

        #compute the values of the curl of vorticity over the lattice
        x1 = torch.reshape(lattice[...,0])
        
        omega_values = omega(lattice) 

        #tensors storing components of the velocity 
        u1 = torch.zeros_like(lattice[...,0])
        u2 = torch.zeros_like(lattice[...,1])
    
        while norm > 0.001:
            u1_temp = torch.clone(u1)
            u2_temp = torch.clone(u2)
            
            u1[1:-1,1:-1] = (((u1_temp[1:-1,2:] + u1_temp[1:-1,0:-2])*h_2**2 +
                               (u1_temp[2:,1:-1] + u1_temp[0:-2,1:-1])*h_1**2) / (2*(h_1**2 + h_2**2)) -
                               h_1**2*h_2**2 / (2*(h_1**2 + h_2**2))*omega_values[1:-1,1:-1])
            u2[1:-1,1:-1] = (((u2_temp[1:-1,2:] + u2_temp[1:-1,0:-2])*h_2**2 +
                               (u2_temp[2:,1:-1] + u2_temp[0:-2,1:-1])*h_1**2) / (2*(h_1**2 + h_2**2)) -
                               h_1**2*h_2**2 / (2*(h_1**2 + h_2**2))*omega_values[1:-1,1:-1])

            #periodic boundary condition at x = H
            u1[1:-1,-1] = (((u1_temp[1:-1,0] + u1_temp[1:-1,-2])*h_2**2 + (u1_temp[2:,-1] + u1_temp[0:-2,-1])*h_1**2) /
                            (2*(h_1**2 + h_2**2)) - h_1**2*h_2**2 / (2*(h_1**2 + h_2**2))*omega_values[1:-1, -1])
            u2[1:-1,-1] = (((u2_temp[1:-1,0] + u2_temp[1:-1,-2])*h_2**2 + (u2_temp[2:,-1] + u2_temp[0:-2,-1])*h_1**2) /
                            (2*(h_1**2 + h_2**2)) - h_1**2*h_2**2 / (2*(h_1**2 + h_2**2))*omega_values[1:-1, -1])

            #periodic boundary condition at x = H
            u1[1:-1,0] = (((u1_temp[1:-1,1] + u1_temp[1:-1,-1])*h_2**2 + (u1_temp[2:,0] + u1_temp[0:-2,0])*h_1**2) /
                          (2*(h_1**2 + h_2**2)) - h_1**2*h_2**2 / (2*(h_1**2 + h_2**2))*omega_values[1:-1, 0])
            u2[1:-1,0] = (((u2_temp[1:-1,1] + u2_temp[1:-1,-1])*h_2**2 + (u2_temp[2:,0] + u2_temp[0:-2,0])*h_1**2) /
                          (2*(h_1**2 + h_2**2)) - h_1**2*h_2**2 / (2*(h_1**2 + h_2**2))*omega_values[1:-1, 0])
        
            #no-slip boundary conditions at y = 0 and y = H
            u1[-1,:] = 0. 
            u2[-1,:] = 0. 
            u1[0,:] = 0. 
            u2[0,:] = 0.

            norm = torch.mean((u1-u1_temp)**2+(u2-u2_temp)**2)
    
        return [u1, u2]
        
    else:
        raise TypeError("Domain is not recognized.")
