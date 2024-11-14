"""
In this module, the vorticity model is learned from the current state of the particle system.
"""
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import utils 
import velocity_retrieval as vel
from configs import parameters

torch.set_num_threads(12)

class MLP(nn.Module):
    """
    A basic class with a fully-connected feedforward NN representing the vorticity (the size of the hidden layers is taken from configs).
    """
    def __init__(self, 
                 input_dim = 2, 
                 hidden_dim = parameters["hidden_dimension_for_model"], 
                 output_dim = 1): 
        super(MLP, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), 
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                     nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        return torch.squeeze(self.network(x))

def train(model, 
          optimizer, 
          processes_positions, 
          delay_term, 
          nb_epochs = parameters["epochs_per_iteration"]
          ):
    """
    A training routine for the vorticity model at a particular time.

    Args:
        model: a NN representing the vorticity to be trained.
        optimizer: an optimizer used for the training routine.
        processes_positions: the current positions for the particles.
        delay_term: the term carrying the current state of the system (called \Omega in the paper).
        nb_epochs: a number of epochs for training (default is taken from configs).
    """
    N = parameters["particles_number"]

    #create lattice for model values over the domain [0,1]x[0,1] (as the arguments are scaled)
    lattice_points = torch.empty(2*N+1,N+1,2)
    lattice_points[:,:,0], lattice_points[:,:,1] = torch.meshgrid(torch.linspace(0, 1, 2*N+1), torch.linspace(0, 1, N+1))
    lattice_points = torch.reshape(lattice_points, ((2*N+1)*(N+1), 2))

    training_loss = []

    loss_for_print = np.empty((nb_epochs,))

    for i in tqdm(range(nb_epochs)):

        model_at_lattice = model.forward(lattice_points) 
        model_at_processes = model.forward(processes_positions)

        #compute the two terms in the loss function
        L2_norm = torch.mean(model_at_lattice**2) 
        scalar_product = torch.mean(delay_term*model_at_processes)

        loss = (L2_norm - 2*scalar_product)
        loss_for_print[i] = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss.append(loss.item())

#       uncomment below to report the loss function values
#        if len(training_loss) == 10:
#            print(torch.mean(torch.tensor(training_loss)))
#            training_loss = []


def scale(arg):
    """
    A default scaler for the processes positions and the delay term; note: scaling is performed into the range [0,1] and is not in-place.

    Args:
        arg: a tensor of arbitrary shape.

    Returns:
        scaled_arg: a new tensor with the scaled values.
        total_min: the minimum value of arg.
        total_max: the maximum value of arg.
    """
    total_min = torch.min(arg)
    total_max = torch.max(arg)
    scaled_arg = (arg - total_min) / (total_max - total_min)
    return scaled_arg, total_min, total_max


def descale_model(model, min_x1, max_x1, min_x2, max_x2, min_values, max_values):
    """
    A function to descale the model arguments and values.

    Args:
        model: a trained model representing the vorticity.
        min_x1: minimum value of the first argument.
        max_x1: maximum value of the first argument.
        min_x2: minimum value of the second argument. 
        max_x2: maximum value of the second argument.
        min_values: minimum value of the range.
        max_values: maximum value of the range.

    Returns:
        omega: a function representing vorticity as learned by the model.
    """
    def omega(x):
        scaled_x = torch.empty_like(x)
        scaled_x[...,0] = (x[...,0] - min_x1) / (max_x1 - min_x1)
        scaled_x[...,1] = (x[...,1] - min_x2) / (max_x2 - min_x2)
        return (max_values - min_values)*model.forward(scaled_x) + min_values

    return omega



def print_stream(omega, lattice = None):
    """
    Function printing streamlines of the velocity along with vorticity values.

    Args:
        omega: a function representing the vorticity.
        lattice: lattice points to print the values at (if None, creates a custom lattice from utils).
    """
    if lattice == None: lattice, _ = utils.create_lattice(reshaped = True)
    N = parameters["particles_number"]

    velocity = vel.u(lattice, omega).detach()
    u1, u2 = velocity[:,0], velocity[:,1]
    omega_values = omega(lattice)

    omega_values = torch.reshape(omega_values, (2*N+1, N+1))
    lattice = torch.reshape(lattice, (2*N+1, N+1, 2))
    u1 = torch.reshape(u1, (2*N+1, N+1))
    u2 = torch.reshape(u2, (2*N+1, N+1))

    #create custom coloring levels that ignore too large values of the vorticity
    M=np.percentile(abs(omega_values.detach().numpy()),90)
    st=M/50
    l=np.arange(-M,M,step=st)

    fig = plt.figure(figsize=(12,6))
    plt.contourf(lattice[...,0].numpy().T, lattice[...,1].numpy().T, 
                 omega_values.detach().numpy().T,vmax=M, vmin=-M, extend="both",cmap="coolwarm",levels=l)
    plt.colorbar()
    st3=plt.streamplot(lattice[...,0].numpy().T, lattice[...,1].numpy().T, 
                       u1.numpy().T, u2.numpy().T,
                       color=np.sqrt(u1.numpy().T**2+u2.numpy().T**2),
                       cmap ='viridis',arrowsize=0.8,density=1.5)
    plt.colorbar(st3.lines) 
    DIR = './streamplots'
    num = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    S="./streamplots/outer_flow"+str(num)+".png"
    plt.savefig(S, dpi=300)
    plt.close()

    return


def print_vorticity(omega, lattice = None):
    """
    Function printing a 3d graph for the vorticity.

    Args:
        omega: a function representing the vorticity.
        lattice: lattice points to print the values at (if None, creates a custom lattice from utils).
    """
    if lattice == None: lattice, _ = utils.create_lattice(reshaped = True)
    N = parameters["particles_number"]

    omega_values = omega(lattice)

    omega_values = torch.reshape(omega_values, (2*N+1,N+1))
    lattice = torch.reshape(lattice, (2*N+1, N+1, 2))

    fig = plt.figure(figsize = (12,10)) 
    ax = plt.axes(projection='3d') 
    surf = ax.plot_surface(lattice[...,0].numpy(), 
                           lattice[...,1].numpy(), 
                           omega_values.detach().numpy(), cmap = plt.cm.coolwarm) 
    plt.show() 
    plt.close()

    return


def print_error(omega, t, lattice = None):
    """
    

    Args:
        omega: a function representing the vorticity.
        lattice: lattice points to print the values at (if None, creates a custom lattice from utils).
    """
    if lattice == None: lattice, _ = utils.create_lattice(reshaped = True)
    N = parameters["particles_number"]
    N_steps = parameters["steps_number"]
    T = parameters["total_time"]

    time = t/N_steps*T

    velocity = vel.u(lattice, omega)
    u1, u2 = velocity[:,0], velocity[:,1]
    true_velocity = utils.true_velocity(lattice, time).numpy()
    true_u1, true_u2 = true_velocity[...,0], true_velocity[...,1]
    error = np.sqrt((true_u1-u1)**2+(true_u2-u2)**2)/np.sqrt(np.mean((true_u1)**2+(true_u2)**2))

    error = np.reshape(error, (N+1, N+1))

    #create custom coloring levels that ignore large values of the vorticity
    # M=np.percentile(abs(error.numpy()),90)
    # st=M/50
    # l=np.arange(-M,M,step=st)

    fig = plt.figure(figsize=(12,8))
    plt.contourf(torch.reshape(lattice[...,0], (N+1, N+1)).numpy().T, torch.reshape(lattice[...,1], (N+1, N+1)).numpy().T, 
                 error.T, levels=100)#,vmax=M, vmin=-M, extend="both",cmap="coolwarm",levels=l)
    plt.colorbar()
    DIR = './streamplots'
    num = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    S="./streamplots/outer_flow"+str(num)+".png"
    plt.savefig(S, dpi=300)
    plt.close()

    f = open("errors.txt", "a")
    rel_error = np.sqrt(np.mean((true_u1-u1)**2+(true_u2-u2)**2))/np.sqrt(np.mean((true_u1)**2+(true_u2)**2))
    f.write(str(rel_error)+"\n")

    return


def launch(delay_term, processes_positions):
    """
    A function to setup the model and launch the training procedure for the vorticity; note: printing is done in this function.

    Args:
        delay_term: the term carrying the current state of the system (called \Omega in the paper).
        processes_positions: the current positions for the particles.

    Returns:
        velocity_at_processes: a tensor of values of the velocity at the processes positions.
        zeta_at_processes: values of the boundary vorticity at the processes found from the vorticity model.
    """

    #scaling for the arguments for nn
    scaled_processes_positions = torch.empty_like(processes_positions)
    scaled_processes_positions[:,0], min_x1, max_x1 = scale(processes_positions[:,0])
    scaled_processes_positions[:,1], min_x2, max_x2 = scale(processes_positions[:,1])

    #scaling for the delay term
    scaled_delay_term, min_values, max_values = scale(delay_term)

    #setup model
    model = MLP()

#    saved_model_path = "model.pt"
#    if os.path.exists(saved_model_path) == False: model = MLP()
#    else:
#        model = MLP()
#        prev_model = torch.load(saved_model_path)
#        model.load_state_dict(prev_model['model_state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #train model
    train(model, optimizer, scaled_processes_positions, scaled_delay_term)

#    torch.save({'model_state_dict': model.state_dict()}, saved_model_path)

    omega = descale_model(model, min_x1, max_x1, min_x2, max_x2, min_values, max_values)
    print_vorticity(omega)
    print_stream(omega)
    
    velocity_at_processes = vel.u(processes_positions, omega)
    zeta_at_processes = utils.zeta(processes_positions, omega)

    return velocity_at_processes, zeta_at_processes
