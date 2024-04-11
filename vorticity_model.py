import torch
import numpy as np
import random
from torch import nn
from tqdm import tqdm
import sklearn.preprocessing as proc
import torch.utils.data
import matplotlib.pyplot as plt
import params

N, H = params.N, params.H
torch.set_num_threads(12)

class MLP(nn.Module):

    def __init__(self, input_dim=2, hidden_dim=512, output_dim=1): #hidden_dim = 128
        super(MLP, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), 
                                     nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), 
                                     nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        return torch.squeeze(self.network(x))
    
    
def plot(lattice_points, model_values):
    lattice_points = torch.reshape(lattice_points, (2*N+1,N+1, 2))
    model_values = torch.reshape(model_values, (2*N+1,N+1))
    fig = plt.figure(figsize = (12,10)) 
    ax = plt.axes(projection='3d') 
    surf = ax.plot_surface(lattice_points[:,:,0].numpy(), lattice_points[:,:,1].numpy(), model_values.detach().numpy(), cmap = plt.cm.coolwarm) 
    plt.show() 
    plt.close()

def train(model, optimizer, processes_positions, time_integral, nb_epochs=1_00):

    #create lattice for model values
    lattice_points = torch.empty(2*N+1,N+1,2)
    lattice_points[:,:,0], lattice_points[:,:,1] = torch.meshgrid(torch.linspace(0, 1, 2*N+1), torch.linspace(0, 1, N+1))
    lattice_points = torch.reshape(lattice_points, ((2*N+1)*(N+1), 2))

    training_loss = []

    for _ in tqdm(range(nb_epochs)):

        model_at_lattice = model.forward(lattice_points) 
        model_at_processes = model.forward(processes_positions)

        L2_norm = torch.mean(model_at_lattice**2) 
        scalar_product = torch.mean(time_integral*model_at_processes)

        loss = (L2_norm - 2*scalar_product)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss.append(loss.item())

        if len(training_loss) == 10:
            print(torch.mean(torch.tensor(training_loss)))
            training_loss = []

def scale_arguments(*argv):
    mins = []
    maxs = []
    for arg in argv:
        mins.append(torch.min(arg))
        maxs.append(torch.max(arg))
    total_min = min(mins)
    total_max = max(maxs)
    normalized_data = []
    for arg in argv:
        normalized_data.append((arg - total_min) / (total_max - total_min))
    normalized_data.append(total_min)
    normalized_data.append(total_max)
    return normalized_data

def scale_values(*argv):
    mins = []
    maxs = []
    for arg in argv:
        mins.append(torch.min(arg))
        maxs.append(torch.max(arg))
    total_min = min(mins)
    total_max = max(maxs)
    normalized_data = []
    for arg in argv:
        normalized_data.append((arg - total_min) / (total_max - total_min))
    normalized_data.append(total_min)
    normalized_data.append(total_max)
    return normalized_data

def descale(*argv, total_min, total_max):
    descaled_data = []
    for arg in argv:
        descaled_data.append((total_max - total_min)*arg + total_min)
    if len(descaled_data) == 1: return descaled_data[0]
    else: return descaled_data

#launches vorticity learning -- returns values of vorticity at the lattice
def launch(time_integral, processes_positions):

    #create lattice for model values
    lattice_points = torch.empty(2*N+1,N+1,2)
    lattice_points[:,:,0], lattice_points[:,:,1] = torch.meshgrid(torch.linspace(0, 1, 2*N+1), torch.linspace(0, 1, N+1))
    lattice_points = torch.reshape(lattice_points, ((2*N+1)*(N+1), 2))

    #scaling for the arguments for nn
    processes_positions[:,0], min_arguments, max_arguments = scale_arguments(processes_positions[:,0])
    processes_positions[:,1], min_arguments, max_arguments = scale_arguments(processes_positions[:,1])

    #scaling for the "values"
    time_integral, min_values, max_values = scale_values(time_integral)

    device = 'cpu'

    #setup model
    model = MLP()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #train model
    train(model, optimizer, processes_positions, time_integral)
    
    model_values = model.forward(lattice_points).detach()
    model_values = descale(model_values, total_min=min_values, total_max=max_values)

    plot(lattice_points, model_values.detach())

    return model.forward, model_values.detach(), model.state_dict()
