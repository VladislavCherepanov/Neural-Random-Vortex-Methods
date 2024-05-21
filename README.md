# Neural Networks-based Random Vortex Methods for Modelling Incompressible Flows

A repository with the official implementation code for the paper **"Neural Networks-based Random Vortex Methods for Modelling Incompressible Flows"**. 

![Neural Random Vortex flowchart](./assets/NRV_algo_flowchart.png)

## Requirements

The implementation is done in PyTorch; other packages that were used in the code (mostly for printing the outcomes) are NumPy and Matplotlib. To install the requirements, use the command:
```
pip install -r requirements.txt
```

## Running the code

To run to code, simply run the main file with the command:
```
python main.py
```
The code will show two tqdm progress bars, for the whole simulation and the current step training procedure respectively. 

To change the initial data, change the initial parameters for the algorithm stored in the parameters dictionary in configs.py and the functions initial_vorticity and external_force in utils.py.
