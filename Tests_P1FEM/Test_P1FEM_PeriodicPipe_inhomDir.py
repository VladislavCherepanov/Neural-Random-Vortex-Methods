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
Nx=10
Ny=10
Lx=1.0
Ly=1.0

# Generate a uniform mesh
Points,Elements,Dirichlet,periodicPairs=P1FEM_periodicConditions.uniformTriangulation_periodicPipe(Nx,Ny,Lx,Ly)



#######################################################################################
#                                 Plot the mesh 
#######################################################################################

plt.figure(1)

# plot points
plt.plot(Points[:,0], Points[:,1], marker='o', color='k', linestyle='none')

# set up list of periodic nodes
PeriodicNodes=[]

for j in range(np.shape(Elements) [0]):
    # extract the nodes of the elements and their periodization
    n1,n2,n3=Elements[j,0],Elements[j,1],Elements[j,2]
    p1,p2,p3=Elements[j,3],Elements[j,4],Elements[j,5]
    
    # plot the edges of the element
    plt.plot( [Points[n1,0],Points[n2,0]], [Points[n1,1],Points[n2,1]], color='b', linestyle='-', linewidth=2)
    plt.plot( [Points[n2,0],Points[n3,0]], [Points[n2,1],Points[n3,1]], color='b', linestyle='-', linewidth=2)
    plt.plot( [Points[n3,0],Points[n1,0]], [Points[n3,1],Points[n1,1]], color='b', linestyle='-', linewidth=2)

    # store the (non trivial) periodization tuples in the list PeriodicNodes
    # and mark the periodization tuples with red-cyanide X's
    if n1!=p1:
        plt.plot(Points[n1,0], Points[n1,1], marker='X', color='r', linestyle='none')
        plt.plot(Points[p1,0], Points[p1,1], marker='X', color='c', linestyle='none')
        PeriodicNodes.append([n1,j])
    if n2!=p2:
        plt.plot(Points[n2,0], Points[n2,1], marker='X', color='r', linestyle='none')
        plt.plot(Points[p2,0], Points[p2,1], marker='X', color='c', linestyle='none')
        PeriodicNodes.append([n2,j])
    if n3!=p3:
        plt.plot(Points[n3,0], Points[n3,1], marker='X', color='r', linestyle='none')
        plt.plot(Points[p3,0], Points[p3,1], marker='X', color='c', linestyle='none')
        PeriodicNodes.append([n3,j])

# mark the midpoint of each Dirichlet edge through green X's
for j in range(np.shape(Dirichlet) [0]):
    p0=(Points[Dirichlet[j,0],0]+Points[Dirichlet[j,1],0])/2
    p1=(Points[Dirichlet[j,0],1]+Points[Dirichlet[j,1],1])/2
    plt.plot(p0, p1, marker='X', color='g', linestyle='none')


plt.title("Generated mesh")



################################################
#               First test: Couette
################################################

# define true solution (boundary conditions are clearly satisfied)
U=3

def Couette(X,Y,h=Ly,Utop=U):

    DirIndicatorTop=np.ones(np.shape(Y))
    DirIndicatorBot=np.ones(np.shape(Y))
    DirIndicatorBot[Y<0]=0
    DirIndicatorTop[Y>=h]=0

    return DirIndicatorTop*DirIndicatorBot*Utop/h*Y + (1-DirIndicatorTop)*DirIndicatorBot*Utop*np.ones(np.shape(Y)[0])

def BoundaryCondCouette(X,Y,h=Ly,Utop=U):
    uD=np.zeros(np.shape(X))
    uD[Y>h/2]=Utop
    return uD

#define its - Delta SinCos
def LaplacianCouette(X,Y) :
    return np.zeros(np.shape(Y))

# instantiate Laplacian and boundary conditions of the true solution
f=LaplacianCouette
uD=BoundaryCondCouette

# compute the true solution on the grid points
trueValsNodes=Couette(Points[:,0],Points[:,1])

# plot true solution
fig=plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(Points[:,0], Points[:,1], trueValsNodes, cmap=cm.jet, linewidth=0)
fig.colorbar(surf)
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))
plt.title("True Couette")
fig.tight_layout()








# compute approximation u of SinCos by solving the Laplace equation
# with the right hand side given by the Laplacian of SinCos
u=P1FEM_periodicConditions.solveLaplace_periodicPipe(Points,Elements,Dirichlet,periodicPairs,f,uD)


# plot approximate solution
fig=plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(Points[:,0], Points[:,1], u, cmap=cm.jet, linewidth=0)
fig.colorbar(surf)
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))
plt.title("approximate solution")
fig.tight_layout()



# Error metrics
firstTriangle= Points[Elements[1,0:3],:]
areaUniform= np.linalg.det(np.vstack(([1,1,1],np.transpose(firstTriangle))))/2
Err=np.linalg.norm(u-trueValsNodes)*np.sqrt(areaUniform)
relErr=np.linalg.norm(u-trueValsNodes)/np.linalg.norm(trueValsNodes)
locErr=np.absolute(u-trueValsNodes)
print("\n\n\n","Elemeent area:",areaUniform,"\nEuclidean error: ",Err,"\nRelative error: ",relErr,"\nMax. loc. error: ",np.max(locErr),"\n")


# plot local error
fig=plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(Points[:,0], Points[:,1],locErr , cmap=cm.jet, linewidth=0)
fig.colorbar(surf)
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))
plt.title("error")
fig.tight_layout()


#########################################################################################################
#                                       plot on larger grid
#########################################################################################################
x = np.linspace(-Lx, 2*Lx, 3*(Nx+1))
y = np.linspace(-Ly, 2*Ly, 3*(Ny+1))
X, Y = np.meshgrid(x, y)
coordinatesGrid = np.vstack([X.ravel(), Y.ravel()]).T


interpolationMethod='convexCombination'
uApprox_on_Grid=P1FEM_periodicConditions.evalFEM_periodicPipe_unifGrid(coordinatesGrid,u,Nx,Ny,Lx,Ly,interpolationMethod,uD)
# plot U on larger domain
fig=plt.figure(5)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(coordinatesGrid[:,0], coordinatesGrid[:,1],uApprox_on_Grid, cmap=cm.jet, linewidth=0)
fig.colorbar(surf)
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))
plt.title('Approximation on larger grid\n'+interpolationMethod)
fig.tight_layout()

plt.show()