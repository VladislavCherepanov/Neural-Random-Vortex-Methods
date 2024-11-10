import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import P1FEM_periodicConditions


#######################################################################################
#                                   Generate mesh 
#######################################################################################

# choose mesh size
Nx=25
Ny=25
Lx=1.0
Ly=1.0

# Generate a uniform mesh
Points,Elements,Dirichlet,periodicPairs=P1FEM_periodicConditions.uniformTriangulation_periodicPipe(Nx,Ny,Lx,Ly)


################################################
#               First test
################################################

# define true solution (boundary conditions are clearly satisfied)
def SinCos(X,Y):
    return np.cos(2*np.pi*X)*np.sin(2*np.pi*Y)

#define its - Delta SinCos
def LaplacianSinCos(X,Y) :
    return ((2*np.pi)**2) * 2 * np.cos(2*np.pi*X)*np.sin(2*np.pi*Y)

# plot error on larger domain
def SinCosextended(X,Y):
    z=np.cos(2*np.pi*X)*np.sin(2*np.pi*Y)
    z[Y>Ly]=0
    z[Y<0]=0
    return z



# instantiate true solution
f=LaplacianSinCos

# compute approximation u of SinCos by solving the Laplace equation
# with the right hand side given by the Laplacian of SinCos
u=P1FEM_periodicConditions.solveLaplace_periodicPipe(Points,Elements,Dirichlet,periodicPairs,f)


# evaluation of u beyond the domain 

# Generate mesh on the larger domain
x = np.linspace(-Lx, 2*Lx, 3*(Nx+1))
y = np.linspace(-Ly, 2*Ly, 3*(Ny+1))
X, Y = np.meshgrid(x, y)
coordinatesGrid = np.vstack([X.ravel(), Y.ravel()]).T
Z=SinCosextended(coordinatesGrid[:,0], coordinatesGrid[:,1])


fig=plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(coordinatesGrid[:,0], coordinatesGrid[:,1],Z , cmap=cm.jet, linewidth=0)
fig.colorbar(surf)
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))
plt.title('true solution on larger grid')
fig.tight_layout()

##############################################################################################################################
#                                               quadrant average
##############################################################################################################################

interpolationMethod='quadrantAverage'
uApprox_on_Grid=P1FEM_periodicConditions.evalFEM_periodicPipe_unifGrid(coordinatesGrid,u,Nx,Ny,Lx,Ly,interpolationMethod)
# plot U on larger domain
fig=plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(coordinatesGrid[:,0], coordinatesGrid[:,1],uApprox_on_Grid, cmap=cm.jet, linewidth=0)
fig.colorbar(surf)
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))
plt.title('Approximation on larger grid\n'+interpolationMethod)
fig.tight_layout()




fig=plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(coordinatesGrid[:,0], coordinatesGrid[:,1],np.absolute(uApprox_on_Grid-Z) , cmap=cm.jet, linewidth=0)
fig.colorbar(surf)
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))
plt.title('Error on larger grid\n'+interpolationMethod)
fig.tight_layout()


##############################################################################################################################
#                                               convex combination
##############################################################################################################################

interpolationMethod='convexCombination'
uApprox_on_Grid=P1FEM_periodicConditions.evalFEM_periodicPipe_unifGrid(coordinatesGrid,u,Nx,Ny,Lx,Ly,interpolationMethod)
# plot U on larger domain
fig=plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(coordinatesGrid[:,0], coordinatesGrid[:,1],uApprox_on_Grid, cmap=cm.jet, linewidth=0)
fig.colorbar(surf)
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))
plt.title('Approximation on larger grid\n'+interpolationMethod)
fig.tight_layout()



fig=plt.figure(5)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(coordinatesGrid[:,0], coordinatesGrid[:,1],np.absolute(uApprox_on_Grid-Z) , cmap=cm.jet, linewidth=0)
fig.colorbar(surf)
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))
plt.title('Error on larger grid\n'+interpolationMethod)
fig.tight_layout()



##############################################################################################################################
#                                               nearest node combination
##############################################################################################################################

interpolationMethod='nearestNode'
uApprox_on_Grid=P1FEM_periodicConditions.evalFEM_periodicPipe_unifGrid(coordinatesGrid,u,Nx,Ny,Lx,Ly,interpolationMethod)
# plot U on larger domain
fig=plt.figure(6)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(coordinatesGrid[:,0], coordinatesGrid[:,1],uApprox_on_Grid, cmap=cm.jet, linewidth=0)
fig.colorbar(surf)
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))
plt.title('Approximation on larger grid\n'+interpolationMethod)
fig.tight_layout()

fig=plt.figure(7)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(coordinatesGrid[:,0], coordinatesGrid[:,1],np.absolute(uApprox_on_Grid-Z) , cmap=cm.jet, linewidth=0)
fig.colorbar(surf)
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))
plt.title('Error on larger grid\n'+interpolationMethod)
fig.tight_layout()


plt.show()