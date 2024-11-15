import numpy as np
import scipy.sparse as sparse


def importing_P1FEM_periodicPipe_successfull():
    """
        prints a message that P1FEM_periodicPipe was imported
    """
    print("\n\n\nImporting of P1FEM_periodicPipe successful\n\n\n")


###########################################################################################################################
#                                               mean over Triangulation
########################################################################################################################### 

def meanComputation(coordinates,elements,f):

    fMean = 0
    nE = np.shape(elements) [0]

    for i in range(nE):
        nodesidx = elements[i,0:3]
        nodes=coordinates[nodesidx,:]


        # quadrature of f on the i-th element
        P = np.vstack(([1,1,1],np.transpose(nodes)))
        areaT = np.linalg.det(P)/2
        sT = np.array(nodes).sum(axis=0)/3

        fMean += areaT * f(sT[0],sT[1])

    return fMean






##########################################################################################################################
#                                       Periodic Pipe
##########################################################################################################################




# generate uniform mesh
#
def uniformTriangulation_periodicPipe(Nx,Ny,Lx=1.0,Ly=1.0):
    """
    Creates a uniform triangulation of the periodic Pipe ([0,Lx] mod Lx)X[0,Ly], with
        step size 1/Nx in the first direction (x-direction) and step size 1/(Ny+1) in the second direction.
        I.e. this function produces a triangulation of the rectangle [0,Lx]X[0,Ly] that works with
        periodic boundary conditions on the left and right, and Dirichlet conditions on the top and bottom.

    Args:
        Nx: int, specifies them number of discretization steps in the x-direction
        Ny: int, specifies them number of discretization steps in the y-direction
        Lx: float (optional), determining the (periodic) length of the pipe (standard input: 1.0 )
        Ly: float (optional), determining the height of the pipe (standard input: 1.0 )
    
    Returns:
        coordinates: ndarray of float types, shape ( (Nx+1)*(Ny+1) , 2 )  the row (i,:) stores the coordinates of the i-th gridpoint
        elements:    ndarray of int64, shape ( nE , 6 ), the row (i,:) stores the coordinates of the i-th gridpoint
                        - the first 3 elements of (i,:), i.e.  (i,0:3), contain the vertex numbers before periodization
                        - the last 3 elements of (i,:), i.e.  (i,3:6), contain the vertex numbers after periodization
        Dirichlet:   ndarray of int64, shape ( nD , 4 ), the row (i,:) stores the coordinates of the i-th Dirichlet edge
                        - the first 2 elements of (i,:), i.e.  (i,0:2), contain the vertex numbers of the i-th edge before periodization
                        - the last 2 elements of (i,:), i.e.  (i,2:4), contain the vertex numbers of the i-th edge after periodization
        periodicPairs   a (n,2) int-array, where n is the (i,j) denotes the periodic  correspondence of i an j

    Comments:
        Still not completely optimized, elements and Dirichlet are first stored as lists and then turned into arrays
    """

    # Generate mesh
    x = np.linspace(0, Lx, Nx+1) # Nx+1 because this also creates the boundary nodes before peridization
    y = np.linspace(0, Ly, Ny+1)
    X, Y = np.meshgrid(x, y)
    coordinates = np.vstack([X.ravel(), Y.ravel()]).T


    # Create triangular elements by splitting each square
    elements = []
    Dirichlet = []
    periodicPairs=np.zeros((Ny+1,2), dtype=int)
    k=0
    for i in range(Ny):
        for j in range(Nx):
            # Node indices for square in mesh
            n0 = i * (Nx + 1) + j
            n1 = n0 + 1
            n2 = n0 + (Nx + 1)
            n3 = n2 + 1
            # Node indices after periodization
            p0 = n0
            p1 = n1
            p2 = n2
            p3 = n3
            if j==Nx-1: 
                p1 = i * (Nx + 1)
                p3 = p1 + (Nx + 1)
                periodicPairs[k,0]=p1
                periodicPairs[k,1]=n1
                
                k+=1

            # Two triangles for each square,
            #   before (n0,n1,n2) and after (p0,p1,p2) periodization
            elements.append([n0, n1, n3, p0, p1, p3])  # Lower triangle
            elements.append([n0, n3, n2, p0, p3, p2])  # Upper triangle
            
            # adding lower edge to Dirichlet boundary
            if i==0:
                Dirichlet.append([n0,n1,p0,p1])
            # adding upper edge
            if i==Ny-1:
                Dirichlet.append([n2,n3,p2,p3])

    periodicPairs[k,1] = Ny * (Nx + 1) + Nx
    periodicPairs[k,0] = Ny * (Nx + 1)

    elements=np.array(elements,dtype=int)
    Dirichlet=np.array(Dirichlet,dtype=int)

    return coordinates,elements,Dirichlet,periodicPairs






# solve the Poisson equation
#
def solveLaplace_periodicPipe(coordinates,elements,Dirichlet,periodicPairs,f,uD=None):
    """
    Approximates the solution u of the Poisson equation -Delta u = f on the periodic pipe (with homogenous Dirichlet condition on top and bottom),
        using a finite element discretisation given by [coordinates,elements,Dirichlet,periodicPairs]
    
    params:             
        coordinates:    a (nC,2) float array containing the coordinates of the mesh
        elements:       ndarray of int64, shape ( nE , 6 ), the row (i,:) stores the coordinates of the i-th triangle (before and after periodization)
                            - the first 3 elements of each row, i.e (i,0:3), contain the indices without periodization
                            - the last 3 element of each row, i.e (i,3:6), contain the indices when periodicity is enforced
                        --> see also Returns of the method uniformTriangulation_periodicPipe            
        Dirichlet:      ndarray of int64, shape ( nD , 4 ), the row (i,:) stores the coordinates of the i-th Dirichlet edge (before and after periodization)      
                            - the first 2 elements of each row, i.e (i,0:2), contain the indices without periodization
                            - the last 2 element of each row, i.e (i,2:4), contain the indices when periodicity is enforced
                        --> see also Returns of the method uniformTriangulation_periodicPipe
        periodicPairs:  a (n,2) int-array, where n is the (i,j) denotes the periodic correspondence of i an j
        f:              a function (handle) describing the right hand side of Poisson equation

    Returns:
        x:              a (nC,) float array describing the finite element approximation of the true solution u through its values at the gridpoints (the rows of coordinates)
                            i.e. the i-th component is the value of the finite element approximation Uh at coordinates[i,:]:
                            --> x[i]=Uh(coordinates[i,:])
    """
    
    nC = np.shape(coordinates) [0] # Number of coordinates
    nE = np.shape(elements) [0]    # Number of elements
    x = np.zeros(nC)               # initialization of solution vector
    #periodizationTuple=[]          # list storing the tuples for periodization

    # Assembly of stiffness matrix
    A = sparse.lil_matrix((nC,nC))  # Sparse matrix format

    for i in range(nE):
        # retrieve the coordinates of the nodes of the i-th element
        # Nodes must be ordered counter clockwise!
        nodesidx = elements[i,0:3]
        nodes=coordinates[nodesidx,:]

        #for j in range(3):              # inefficent/wastefull!!! But doesn't increase order
        #    if elements[i,j]!=elements[i,j+3]:
        #        periodizationTuple.append([elements[i,j],elements[i,j+3]])


        P = np.vstack(([1,1,1],np.transpose(nodes)))
        areaT = np.linalg.det(P)/2
        grad = np.linalg.solve(P,np.array([[0,0],[1,0],[0,1]]))

        A[np.ix_(elements[i,3:6], elements[i,3:6])] += areaT * grad @ np.transpose(grad)

    A = A.tocsr()

    # incorporating Dirichlet conditions
    if uD != None:
        for k in np.unique(Dirichlet):
            x[k-1] = uD(coordinates[k-1,0],coordinates[k-1,1])
    
    # Assembly of right hand side
    b = np.zeros(nC)
    b =-A @ x

    for i in range(nE):
        nodesidx = elements[i,0:3]
        nodes=coordinates[nodesidx,:]


        # quadrature of f on the i-th element
        P = np.vstack(([1,1,1],np.transpose(nodes)))
        areaT = np.linalg.det(P)/2
        sT = np.array(nodes).sum(axis=0)/3

        b[elements[i,3:6]] += areaT * f(sT[0],sT[1])/3


    # Computation of P1−FEM approximation
    freenodes=np.setdiff1d(elements[:,3:6] , Dirichlet)
    x[freenodes]=sparse.linalg.spsolve(A[np.ix_(freenodes, freenodes)],b[freenodes])
    #x[Dirichlet]=np.zeros(Dirichlet.shape) # unnecessary in the case of hom Dirichlet

    # enforcing periodicity
    for k in range(periodicPairs.shape[0]):  
        x[periodicPairs[k,1]]=x[periodicPairs[k,0]]

    return x







# evaluate solution
#
def evalFEM_periodicPipe_unifGrid(z,u,Nx,Ny,Lx=1.0,Ly=1.0,interpolation='convexCombination',uD=None):
    """
        Evaluates a function U(z) at an arbitrary point z=(x,y) given by the the finite element discretization (u,coordinates,elements) over a uniform grid
            of the periodic pipe, where u are the coordinates of U in the finite element basis.

        params:
            z               float type array, storing the coordinate of the evaluation points
            u               float type array, storing the coordinates of the function to be evaluated

        
        Returns:
            U               float type array, storing the evluations

    """
    # initiazlize ouput vector
    U=np.zeros(np.shape(z)[0])
    # Compute x index of the quadrant in which z lies (mod is used to enforce periodicity)
    J=np.mod(np.floor((Nx*z[:,0])/Lx),Nx)
    # Compute y index of the quadrant in which z lies
    I=np.floor(Ny*z[:,1]/Ly)
    
    # Compute an indicator function and a value vector to later enforce the Dirichlet conditions beyond the top and the bottom
    DirIndicator=np.ones(np.shape(z)[0])
    DirIndicator[I<0]=0
    DirIndicator[I>Ny-1]=0
    DirVals=np.zeros(np.shape(z)[0])
    if uD != None:
        DirVals=uD(z[:,0],z[:,1])

    # Fix associated grid points for those inputs (by setting them to zero) which lie beyond the Dirichlet boundary for later computations
    I=I*DirIndicator
    # Change datatype of index to int
    Iint=I.astype(int)
    Jint=J.astype(int)
    # Node indices for the quadrant of the mesh in which x lies
    n0 = Iint * (Nx + 1) + Jint
    n1 = n0 + 1
    n2 = n0 + (Nx + 1)
    n3 = n2 + 1
    #compute interpolation
    if interpolation=='convexCombination':      # Compute the bilinear interpolation
        lambdaX=np.remainder((Nx*z[:,0]),Lx)
        lambdaY=np.remainder((Ny*z[:,1]),Ly)
        U=DirIndicator*(u[n0]*(1-lambdaX)*(1-lambdaY)+u[n1]*lambdaX*(1-lambdaY)+u[n2]*(1-lambdaX)*lambdaY+u[n3]*lambdaX*lambdaY) + (1-DirIndicator)*DirVals
    elif interpolation=='quadrantAverage': # Compute U by averaging over the quadrant in which it lies
        U=DirIndicator*(u[n0]+u[n1]+u[n2]+u[n3])/4 + (1-DirIndicator)*DirVals
    #elif interpolation=='weightedMean':
    #    w0= (-z[:,1])
    elif interpolation=='nearestNode':            # in theory this should give the nearest node, but maybe problematic because of conditioning?
        lambdaX=np.round(np.remainder((Nx*z[:,0]),Lx))
        lambdaY=np.round(np.remainder((Ny*z[:,1]),Ly))
        U=DirIndicator*(u[n0]*(1-lambdaX)*(1-lambdaY)+u[n1]*lambdaX*(1-lambdaY)+u[n2]*(1-lambdaX)*lambdaY+u[n3]*lambdaX*lambdaY) + (1-DirIndicator)*DirVals
    
    


    return U
    










####################################################################################################################
#                           Torus
####################################################################################################################



def uniformTriangulation_Torus(Nx,Ny,Lx=1.0,Ly=1.0):
    """
    Creates a uniform triangulation of the periodic Pipe ([0,Lx] mod Lx)X[0,Ly], with
        step size 1/Nx in the first direction (x-direction) and step size 1/(Ny+1) in the second direction.
        I.e. this function produces a triangulation of the rectangle [0,Lx]X[0,Ly] that works with
        periodic boundary conditions on the left and right, and Dirichlet conditions on the top and bottom.

    Args:
        Nx: int, specifies them number of discretization steps in the x-direction
        Ny: int, specifies them number of discretization steps in the y-direction
        Lx: float (optional), determining the (periodic) length of the pipe (standard input: 1.0 )
        Ly: float (optional), determining the height of the pipe (standard input: 1.0 )
    
    Returns:
        coordinates: ndarray of float types, shape ( (Nx+1)*(Ny+1) , 2 )  the row (i,:) stores the coordinates of the i-th gridpoint
        elements:    ndarray of int64, shape ( nE , 6 ), the row (i,:) stores the coordinates of the i-th gridpoint
                        - the first 3 elements of (i,:), i.e.  (i,0:3), contain the vertex numbers before periodization
                        - the last 3 elements of (i,:), i.e.  (i,3:6), contain the vertex numbers after periodization
        periodicPairs   a (n,2) int-array, where n is the (i,j) denotes the periodic  correspondence of i an j

    Comments:
        Still not completely optimized, elements is first stored as a list and then turned into an array
    """

    # Generate mesh
    x = np.linspace(0, Lx, Nx+1) # Nx+1 because this also creates the boundary nodes before peridization
    y = np.linspace(0, Ly, Ny+1)
    X, Y = np.meshgrid(x, y)
    coordinates = np.vstack([X.ravel(), Y.ravel()]).T


    # Create triangular elements by splitting each square
    elements = []
    periodicPairs=np.zeros((Nx+1+Ny,2), dtype=int)
    k=0
    for i in range(Ny):
        for j in range(Nx):
            # Node indices for square in mesh
            n0 = i * (Nx + 1) + j
            n1 = n0 + 1
            n2 = n0 + (Nx + 1)
            n3 = n2 + 1
            # Node indices after periodization
            p0 = n0
            p1 = n1
            p2 = n2
            p3 = n3
            if j==Nx-1 and i == Ny-1: # top right quadrant
                p1 = i * (Nx + 1)
                p2 = j
                p3 = 0
                periodicPairs[k,1]=n2
                periodicPairs[k,0]=p2
                periodicPairs[k+1,1]=n1
                periodicPairs[k+1,0]=p1
                k+=2
            if j==Nx-1 and i<Ny-1: # right side
                p1 = i * (Nx + 1)
                p3 = p1 + (Nx + 1)
                periodicPairs[k,1]=n1
                periodicPairs[k,0]=p1
                k+=1
            if i == Ny-1 and j<Nx-1: # top side
                p2 = j
                p3 = j+1
                periodicPairs[k,1]=n2
                periodicPairs[k,0]=p2
                k+=1
            # Two triangles for each square,
            #   before (n0,n1,n2) and after (p0,p1,p2) periodization
            elements.append([n0, n1, n3, p0, p1, p3])  # Lower triangle
            elements.append([n0, n3, n2, p0, p3, p2])  # Upper triangle


    periodicPairs[k,1] = Ny * (Nx + 1) + Nx
    periodicPairs[k,0] = 0

    elements=np.array(elements,dtype=int)

    return coordinates,elements,periodicPairs









def solveLaplace_Torus(coordinates,elements,periodicPairs,f,umean=0,gridType='uniform',centralization='Off'):
    """
    
    params:
        coordinates:    a Nx2 array containing the coordinates of the mesh
        elements:       a (nE,6)-array, where nE is the number of integers. The i-th row stores the indices of the nodes that make up the i-th triangle
                            the first 3 elements of each row, i.e (i,0:3), contain the indices without periodization
                            the last 3 element of each row, i.e (i,3:6), contain the indices when periodicity is enforced
        periodicPairs:
        f:
        umean           float type scalar, mean valued of the solution u


    """
    
    nC = np.shape(coordinates) [0] # Number of coordinates
    nE = np.shape(elements) [0]    # Number of elements
    x = np.zeros(nC)               # initialization of solution vector
    #periodizationTuple=[]          # list storing the tuples for periodization

    # Assembly of stiffness matrix
    A = sparse.lil_matrix((nC,nC))  # Sparse matrix format

    # initialize approximation of the total area
    totalArea=0

    for i in range(nE):
        # retrieve the coordinates of the nodes of the i-th element
        # Nodes must be ordered counter clockwise!
        nodesidx = elements[i,0:3]
        nodes=coordinates[nodesidx,:]


        P = np.vstack(([1,1,1],np.transpose(nodes)))
        # Compute area of the element areaT and totalArea
        areaT = np.linalg.det(P)/2 
        totalArea += areaT # needed for averaging non uniform grids
        # Compute grads
        grad = np.linalg.solve(P,np.array([[0,0],[1,0],[0,1]]))
        # append stiffness matrix
        A[np.ix_(elements[i,3:6], elements[i,3:6])] += areaT * grad @ np.transpose(grad)

    A = A.tocsr()

    
    # Assembly of right hand side
    b = np.zeros(nC)
    bMean = 0

    for i in range(nE):
        nodesidx = elements[i,0:3]
        nodes=coordinates[nodesidx,:]


        # quadrature of f on the i-th element
        P = np.vstack(([1,1,1],np.transpose(nodes)))
        areaT = np.linalg.det(P)/2
        sT = np.array(nodes).sum(axis=0)/3

        b[elements[i,3:6]] += areaT * f(sT[0],sT[1])/3
        bMean += areaT * f(sT[0],sT[1])/3
        

    if centralization != 'Off':
        print("\nsolveLaplace_Torus:   Centralization of the RHS!\n")
        b=b-bMean*np.ones(nC)
        

    # Computation of freenodes
    freenodes=np.setdiff1d(elements[:,3:6] , np.array([0])) # extract the origin, which will later on be determined by the average
    
    # Computation of P1−FEM approximation 
    x[freenodes]=sparse.linalg.spsolve(A[np.ix_(freenodes, freenodes)],b[freenodes])
    x[0]=0 # temporarily pin the origin to zero, true value computed later through mean

    # enforcing periodicity
    for k in range(periodicPairs.shape[0]):  
        x[periodicPairs[k,1]]=x[periodicPairs[k,0]]
        

    # Compute value at the origin (and its periodic copies) by the average dicrepancy
    
    xMean=0

    if gridType=='uniform':

        xMean=np.average(x[elements[:,0:3]]@np.ones(3)/3)
        meanDiscrepancy=umean-xMean

        x+=meanDiscrepancy
    else:
        #print("\nsolveLaplace_Torus:   mean retrieval with general grid\n")
        for i in range(nE):
            # retrieve the coordinates of the nodes of the i-th element
            # Nodes must be ordered counter clockwise!
            nodesidx = elements[i,0:3]
            nodes=coordinates[nodesidx,:]

            #compute element area
            P = np.vstack(([1,1,1],np.transpose(nodes)))
            areaT = np.linalg.det(P)/2 

            #update mean approximation
            xMean+=(areaT/totalArea)*(x[nodesidx]@np.ones(3)/3)

        meanDiscrepancy=umean-xMean
        x+=meanDiscrepancy


    

    return x




def solveLaplace_Torus_alternateMeanComp(coordinates,elements,periodicPairs,f,umean=0):
    """
    
    params:
        coordinates:    a Nx2 array containing the coordinates of the mesh
        elements:       a (nE,6)-array, where nE is the number of integers. The i-th row stores the indices of the nodes that make up the i-th triangle
                            the first 3 elements of each row, i.e (i,0:3), contain the indices without periodization
                            the last 3 element of each row, i.e (i,3:6), contain the indices when periodicity is enforced
        periodicPairs:
        f:
        umean           float type scalar, mean valued of the solution u


    """
    
    nC = np.shape(coordinates) [0] # Number of coordinates
    nE = np.shape(elements) [0]    # Number of elements
    x = np.zeros(nC)               # initialization of solution vector
    #periodizationTuple=[]          # list storing the tuples for periodization

    # initialize approximation of the total area
    totalArea=0

    # Computation of freenodes
    freenodes=np.unique(elements[:,3:6]) # extract the origin, which will later on be determined by the average

    # Assembly of stiffness matrix
    A = sparse.lil_matrix((nC,nC))  # Sparse matrix format

    for i in range(nE):
        # retrieve the coordinates of the nodes of the i-th element
        # Nodes must be ordered counter clockwise!
        nodesidx = elements[i,0:3]
        nodes=coordinates[nodesidx,:]


        P = np.vstack(([1,1,1],np.transpose(nodes)))
        # Compute area of the element areaT and totalArea
        areaT = np.linalg.det(P)/2 
        totalArea += areaT # needed for averaging non uniform grids
        # Compute grads
        grad = np.linalg.solve(P,np.array([[0,0],[1,0],[0,1]]))
        # append stiffness matrix
        A[np.ix_(elements[i,3:6], elements[i,3:6])] += areaT * grad @ np.transpose(grad)

    for i in range(nC):
        # design top row of the stiffness matrix such that it encodes the mean
        A[0,i] = totalArea/len(freenodes)
    
    #print(nC,len(freenodes))

    A = A.tocsr()

    
    # Assembly of right hand side
    b = np.zeros(nC)
    bMean = 0

    for i in range(nE):
        nodesidx = elements[i,0:3]
        nodes=coordinates[nodesidx,:]


        # quadrature of f on the i-th element
        P = np.vstack(([1,1,1],np.transpose(nodes)))
        areaT = np.linalg.det(P)/2
        sT = np.array(nodes).sum(axis=0)/3

        b[elements[i,3:6]] += areaT * f(sT[0],sT[1])/3
        bMean += areaT * f(sT[0],sT[1])/3
        
    b[0] = umean

    
    
    # Computation of P1−FEM approximation 
    x[freenodes]=sparse.linalg.spsolve(A[np.ix_(freenodes, freenodes)],b[freenodes])
    

    # enforcing periodicity
    for k in range(periodicPairs.shape[0]):  
        x[periodicPairs[k,1]]=x[periodicPairs[k,0]]
         
    xMean=np.average(x[elements[:,0:3]]@np.ones(3)/3)
    meanDiscrepancy=umean-xMean
    x+=meanDiscrepancy
    print(xMean,meanDiscrepancy)

    return x






# evaluate solution
#
def evalFEM_Torus_unifGrid(z,u,Nx,Ny,Lx=1.0,Ly=1.0,interpolation='convexCombination'):
    """
        Evaluates a function U(z) at an arbitrary point z=(x,y) given by the the finite element discretization (u,coordinates,elements) over a uniform grid of the torus, 
            where u are the coordinates of U in the finite element basis.

        params:
            z               float type array, storing the coordinate of the evaluation points
            u               float type array, storing the coordinates of the function to be evaluated

        
        Returns:
            U               float type array, storing the evluations

    """
    # initiazlize ouput vector
    U=np.zeros(np.shape(z)[0])
    # Compute x index of the quadrant in which z lies (mod is used to enforce periodicity)
    J=np.mod(np.floor((Nx*z[:,0])/Lx),Nx)
    # Compute y index of the quadrant in which z lies
    I=np.mod(np.floor(Ny*z[:,1]/Ly),Ny) 
    # Change datatype of index to int
    Iint=I.astype(int)
    Jint=J.astype(int)
    # Node indices for the quadrant of the mesh in which x lies
    n0 = Iint * (Nx + 1) + Jint
    n1 = n0 + 1
    n2 = n0 + (Nx + 1)
    n3 = n2 + 1
    #compute interpolation
    if interpolation=='convexCombination':      # Compute the bilinear interpolation, the standard input
        lambdaX=np.remainder((Nx*z[:,0]),Lx)
        lambdaY=np.remainder((Ny*z[:,1]),Ly)
        U=(u[n0]*(1-lambdaX)*(1-lambdaY)+u[n1]*lambdaX*(1-lambdaY)+u[n2]*(1-lambdaX)*lambdaY+u[n3]*lambdaX*lambdaY)
    elif interpolation=='quadrantAverage': # Compute U by averaging over the quadrant in which it lies
        U=(u[n0]+u[n1]+u[n2]+u[n3])/4
    #elif interpolation=='weightedMean':
    #    w0= (-z[:,1])
    elif interpolation=='nearestNode':            # in theory this should give the nearest node, but maybe problematic because of conditioning?
        lambdaX=np.round(np.remainder((Nx*z[:,0]),Lx))
        lambdaY=np.round(np.remainder((Ny*z[:,1]),Ly))
        U=(u[n0]*(1-lambdaX)*(1-lambdaY)+u[n1]*lambdaX*(1-lambdaY)+u[n2]*(1-lambdaX)*lambdaY+u[n3]*lambdaX*lambdaY)
    
    


    return U
    
   
    






