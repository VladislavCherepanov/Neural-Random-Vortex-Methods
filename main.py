import numpy as np
import torch
import matplotlib.pyplot as plt

import params
import utils 
import vorticity_model as vm

from tqdm import tqdm

torch.set_num_threads(12)

H = params.H #domain size
N = params.N #number of particles
h0 = H/N #mesh size

N_steps = 50 #number of steps
T = 1.0 #total time
dt = T/N_steps #time increment

nu = 0.01 #viscosity constant
eps = params.eps #boundary cutoff parameter


#initialise lattice points for the domain
lattice_points = utils.create_lattice()

#indicators of hitting the boundary
tau = torch.tensor((lattice_points[:,1] > 0).float())

#initialise the particles at lattice points
X = torch.clone(lattice_points)

#compute initial vorticity
omega_0 = utils.initial_vorticity(X)

#print the initial vorticity
fig = plt.figure(figsize = (12,10)) 
ax = plt.axes(projection='3d') 
surf = ax.plot_surface(torch.reshape(lattice_points[:,0], (2*N+1, N+1)).numpy(), 
                       torch.reshape(lattice_points[:,1], (2*N+1, N+1)).numpy(), 
                       torch.reshape(omega_0, (2*N+1, N+1)).numpy(), cmap = plt.cm.coolwarm) 
plt.show() 
plt.close()

#object storing the current vorticity model
omega = None
parameters = None

#tensors to store the cumulative external vorticity from perturbation and force
ext_vort_pert = nu/(eps**2)*utils.theta(utils.initial_vorticity, X)*utils.cutoff(X)*dt
ext_vort_force = utils.external_force(X, 0)*dt



for t in tqdm(range(N_steps)):
    X_temp = torch.clone(X)

    #initial vorticity plus time integral of the external vorticity
    time_integral = omega_0*tau + ext_vort_pert + ext_vort_force

    #compute the vorticity and find its values (at lattice points)
    omega, omega_values = vm.launch(time_integral, X, parameters)

    #values to update the particles positions
    X_upd = u(X, omega_values)*dt + np.sqrt(2*nu)*np.sqrt(dt)*torch.randn_like(X)

    #indicators for particles that just crossed the boundary
    tau_upd = torch.tensor((X_temp[:,1] > 0).float()*(X_temp[:,1] + X_upd[:,1] <= 0).float())

    #update the indicators for particles that hit the boundary
    tau *= (1.0 - tau_upd)

    #update the particles positions and set the crossing particles
    #to the corrsponding point at the boundary 
    X = X_temp + X_upd*torch.tensordot(tau, torch.tensor([1.0, 1.0]), dims=0) - torch.tensordot(X_temp[:,1]/X_upd[:,1], torch.tensor([1.0, 1.0]), dims=0)*X_upd*torch.tensordot(tau_upd, torch.tensor([1.0, 1.0]), dims=0)

    #update the terms with the external vorticity setting them to zero
    #for particles that just crossed the boundary
    ext_vort_pert = ext_vort_pert*(1.0 - tau_upd) + nu/(eps**2)*utils.theta(omega, X)*utils.cutoff(X)*dt
    ext_vort_force = ext_vort_force*(1.0 - tau_upd) + utils.external_force(X, (t+1)*dt)*dt

    #write print function
    vel = u(lattice_points, omega_values)
    a1 = vel[:,0]
    b1 = vel[:,1]
    fig = plt.figure(figsize=(12,12))
    plt.contourf(torch.reshape(lattice_points[:,0], (2*N+1, N+1)).numpy(), torch.reshape(lattice_points[:,1], (2*N+1, N+1)).numpy(), 
                 torch.reshape(omega_values, (2*N+1, N+1)).numpy(),cmap="coolwarm",levels=100)
    plt.colorbar()
    st3=plt.streamplot(torch.reshape(lattice_points[:,0], (2*N+1, N+1)).numpy().T, torch.reshape(lattice_points[:,1], (2*N+1, N+1)).numpy().T, 
                       torch.reshape(a1, (2*N+1, N+1)).numpy().T, torch.reshape(b1, (2*N+1, N+1)).numpy().T,
                       color=np.sqrt(torch.reshape(a1, (2*N+1, N+1)).numpy().T**2+torch.reshape(b1, (2*N+1, N+1)).numpy().T**2),
                       cmap ='viridis',arrowsize=0.8,density=1.5)
    plt.colorbar(st3.lines)
    plt.show()
    plt.close


    plt.scatter(X[:,0].numpy(), X[:,1].numpy())
    plt.show()
    plt.close()
