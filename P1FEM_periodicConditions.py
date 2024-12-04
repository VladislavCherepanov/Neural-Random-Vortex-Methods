import numpy as np
import scipy.sparse as sparse


def importing_P1FEM_periodicPipe_successfull():
    """
        prints a message that P1FEM_periodicPipe was imported
    """
    print("\n\n\nImporting of P1FEM_periodicPipe successful\n\n\n")


###########################################################################################################################
#                                               additional functions
########################################################################################################################### 

# compute mean of a function over a given triangulation
#
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
#                                       classical Neumann + Dirichlet (Dirichlet boundary must be non empty!)
##########################################################################################################################


# solve the Poisson equation with Neumann and Dirichelt conditions
#   --> Dirichlet boundary must be non-trivial, i.e. there is at least one Dirichlet edge in the triangulation
#
def solveLaplace_classical(coordinates,elements,Dirichlet,Neumann,f,uD=None,g=None):
    """
    Approximates the solution u of the Poisson equation -Delta u = f on the periodic pipe (with homogenous Dirichlet condition on top and bottom),
        using a finite element discretisation given by [coordinates,elements,Dirichlet,periodicPairs]
    
    params:             
        coordinates:    a (nC,2) float array containing the coordinates of the mesh
        elements:       ndarray of int64, shape ( nE , 3 ), the row (i,:) stores the coordinates of the i-th triangle            
        Dirichlet:      ndarray of int64, shape ( nD , 2 ), the row (i,:) stores the coordinates of the i-th Dirichlet edge
        Neumann:        ndarray of int64, shape ( nN , 2 ), the row (i,:) stores the coordinates of the i-th Neumann edge
        f:              a function (handle) describing the right hand side of Poisson equation
        uD:             a function (handle) describing the Dirichlet conditions (standard is None, which corresponds to uD=0)
        g:              a function (handle) describing the Neumann conditions (standard is None, which corresponds to g=0)

    Returns:
        x:              a (nC,) float array describing the finite element approximation of the true solution u through its values at the gridpoints (the rows of coordinates)
                            i.e. the i-th component is the value of the finite element approximation Uh at coordinates[i,:]:
                            --> x[i]=Uh(coordinates[i,:])
    """
    
    nC = np.shape(coordinates) [0] # Number of coordinates
    nE = np.shape(elements) [0]    # Number of elements
    nN = np.shape(Neumann) [0]
    x = np.zeros(nC)               # initialization of solution vector

    # (nE,2) array of first vertex of elements and corresponding edge vectors
    c1 = coordinates[elements[:,0],:]
    d21 = coordinates[elements[:,1],:] - c1
    d31 = coordinates[elements[:,2],:] - c1
    # (nE,) array of element areas
    area4 = 2*(d21[:,0]*d31[:,1]-d21[:,1]*d31[:,0])
    # assembly of stiffness matrix
    a = np.sum(d21*d31,1)/area4
    b = np.sum(d31*d31,1)/area4
    c = np.sum(d21*d21,1)/area4
    S=np.transpose(np.array([(-2)*a+b+c, a-b , a-c , a-b , b , -a , a-c , -a , c])) #values for assembly of the stiffness matrix
    I = elements[:,[0,1,2,0,1,2,0,1,2]] # row indices
    J = elements[:,[0,0,0,1,1,1,2,2,2]] # column indices

    A=sparse.csc_matrix((S.flatten('F'), (I.flatten('F'), J.flatten('F'))), shape=(nC, nC))


    # incorporating Dirichlet conditions into right hand side
    if uD != None:
        DirNodes=np.unique(Dirichlet)#-1
        x[DirNodes] = uD(coordinates[DirNodes,0],coordinates[DirNodes,1])
    b = np.zeros(nC)
    b =-A @ x

    # construct the right hand side term comming from the potential f
    for i in range(nE): 
        nodesidx = elements[i,:]
        nodes = coordinates[nodesidx,:]
        # quadrature of f on the i-th element
        P = np.vstack(([1,1,1],np.transpose(nodes)))
        areaT = np.linalg.det(P)/2
        sT = np.array(nodes).sum(axis=0)/3
        b[nodesidx] += areaT * f(sT[0],sT[1])/3

    # construct the right hand side term comming from the Neumann condition g
    if g != None:
        for i in range(nN):
            nodesidx = Neumann[i,:]
            nodes = coordinates[nodesidx,:]
            # midpoint computation
            mE = np.array(nodes).sum(axis=0)/2
            b[nodesidx] += np.linalg.norm(nodes[0,:]-nodes[1,:])* g(mE[0],mE[1])/2 
    


    # Computation of P1−FEM approximation
    freenodes=np.setdiff1d(elements[:,3:6] , Dirichlet)
    x[freenodes]=sparse.linalg.spsolve(A[np.ix_(freenodes, freenodes)],b[freenodes])
    #x[Dirichlet]=np.zeros(Dirichlet.shape) # unnecessary in the case of hom Dirichlet

    # enforcing periodicity
    for k in range(periodicPairs.shape[0]):  
        x[periodicPairs[k,1]]=x[periodicPairs[k,0]]

    return x





##########################################################################################################################
#                                       Periodic Pipe
##########################################################################################################################

# generate uniform mesh
#
def uniformTriangulation_periodicPipe(Nx,Ny,Lx=1.0,Ly=1.0,boxcutout=np.array([])):
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
        boxcutout:  a (2,2) int array marking a box [a(Lx/Nx),b(Lx/Nx)]x[c(Ly/Ny),d(Ly/Ny)] that is cut out from the pipe
                    --> obstacles[0,0]=a,obstacles[0,1]=b and obstacles[1,0]=c,obstacles[1,1]=d
    
    Returns:
        coordinates: ndarray of float types, shape ( (Nx+1)*(Ny+1) , 2 )  the row (i,:) stores the coordinates of the i-th gridpoint
        elements:    ndarray of int64, shape ( nE , 6 ), the row (i,:) stores the coordinates of the i-th gridpoint
                        - the first 3 elements of (i,:), i.e.  (i,0:3), contain the vertex numbers before periodization
                        - the last 3 elements of (i,:), i.e.  (i,3:6), contain the vertex numbers after periodization
        Dirichlet:   ndarray of int64, shape ( nD , 4 ), the row (i,:) stores the coordinates of the i-th Dirichlet edge
                        - the first 2 elements of (i,:), i.e.  (i,0:2), contain the vertex numbers of the i-th edge before periodization
                        - the last 2 elements of (i,:), i.e.  (i,2:4), contain the vertex numbers of the i-th edge after periodization
        periodicPairs   a (n,2) int-array, where n is the number of periodic pairs and (i,j) denotes the periodic  correspondence of i an j
        gridTrans       a (nCu,2) int-array, where nCu is the number

    Comments:
        Still not completely optimized, elements and Dirichlet are first stored as lists and then turned into arrays
    """

    # Generate full mesh (including the box) 
    x = np.linspace(0, Lx, Nx+1) # Nx+1 because this also creates the boundary nodes before peridization
    y = np.linspace(0, Ly, Ny+1)
    X, Y = np.meshgrid(x, y)
    gridpoints = np.vstack([X.ravel(), Y.ravel()]).T
    gridTrans = np.zeros(np.shape(gridpoints),dtype=int)#
    #  readout number of nodes in the mesh (without the inside of the box)
    noCutouts=0
    if np.shape(boxcutout)==(2,2):
        noCutouts=(boxcutout[1,1]-1-boxcutout[1,0])*(boxcutout[0,1]-1-boxcutout[0,0])
    nC=int(np.shape(gridpoints)[0]-noCutouts)
    #set up coordinates vector
    coordinates = np.zeros((nC,2))
    # translate from the mesh with the nodes inside the box to the one without
    # therefore compute the translation matrix gridTrans and fill the coordinate vector (only with those nodes not inside the box)
    iInObstacleBound=False
    jInObstacleBound=False
    k=0
    for i in range(Ny+1):
        for j in range(Nx+1):
            if np.shape(boxcutout)==(2,2):
                if  boxcutout[1,0] < i and i < boxcutout[1,1]:
                    iInObstacleBound = True
                else:
                    iInObstacleBound = False
                if  boxcutout[0,0] < j and j < boxcutout[0,1]:
                    jInObstacleBound = True
                else:
                    jInObstacleBound = False
            # number of the (i,j)-th node on the full grid with nodes inside the box
            n0=i*(Nx+1)+j
            gridTrans[n0,0]=n0
            if jInObstacleBound and iInObstacleBound: # make nodes that are inside the box negative in the translation matrix
                gridTrans[n0,1]=-1
            else: # 
                coordinates[k,:]=gridpoints[n0,:]
                gridTrans[n0,1]=k
                k+=1
    # Create triangular elements by splitting each square
    elements = []
    Dirichlet = []
    periodicPairs=np.zeros((Ny+1,2), dtype=int)
    # define some indices that are used to determine whether a square is fully inside the box or not, and to count the periodic pairs
    iInObstacleBound=False
    jInObstacleBound=False
    i1InObstacleBound=False
    j1InObstacleBound=False
    k=0
    for i in range(Ny):        
        for j in range(Nx):
            # determine whether the square [(i,j),(i+1,j+1)] is fully inside the box <--> no elements in triangulation
            if np.shape(boxcutout)==(2,2):
                if  boxcutout[1,0] <= i and i <= boxcutout[1,1]:
                    iInObstacleBound = True
                else:
                    iInObstacleBound = False
                if  boxcutout[0,0] <= j and j <= boxcutout[0,1]:
                    jInObstacleBound = True
                else:
                    jInObstacleBound = False
                if  boxcutout[1,0] <= i+1 and i+1 <= boxcutout[1,1]:
                    i1InObstacleBound = True
                else:
                    i1InObstacleBound = False
                if  boxcutout[0,0] <= j+1 and j+1 <= boxcutout[0,1]:
                    j1InObstacleBound = True
                else:
                    j1InObstacleBound = False
            if iInObstacleBound and i1InObstacleBound and jInObstacleBound and j1InObstacleBound: 
                continue # break the inner for loop in case that the square is fully contained in the cutout box and thus not part of the triangulation
            # Node indices for square [(i,j),(i+1,j+1)] in mesh
            basenode = i * (Nx + 1) + j
            n0 = gridTrans[basenode,1]
            n1 = gridTrans[basenode + 1,1]
            n2 = gridTrans[basenode + (Nx + 1),1]
            n3 = gridTrans[basenode + (Nx + 1) + 1,1]
            # Node indices after periodization
            p0 = n0
            p1 = n1
            p2 = n2
            p3 = n3      
            # if on the right boundary
            if j==Nx-1:
                #add periodization 
                p1 = gridTrans[i * (Nx + 1),1]
                p3 = gridTrans[(i+1) * (Nx + 1),1]
                periodicPairs[k,0]=p1
                periodicPairs[k,1]=n1
                k+=1
            # Two triangles for each square,
            #   before (n0,n1,n2) and after (p0,p1,p2) periodization
            elements.append([n0, n1, n3, p0, p1, p3])  # Lower triangle
            elements.append([n0, n3, n2, p0, p3, p2])  # Upper triangle
            # adding lower edge to Dirichlet boundary if element is in the first row
            if i==0:
                Dirichlet.append([n0,n1,p0,p1])
            # adding upper edge to Dirichlet boundary if element is in the last row
            if i==Ny-1:
                Dirichlet.append([n2,n3,p2,p3])
            # adding edges to Dirichlet boundary if the belong to the boundary of the cutout box
            if np.shape(boxcutout)==(2,2): 
                # adding left wall of the boxcutout as Dirichlet edges   
                if j==boxcutout[0,0]-1 and iInObstacleBound and i1InObstacleBound:
                    Dirichlet.append([n1,n3,p1,p3])
                # adding right wall of the boxcutout as Dirichlet edges   
                if j==boxcutout[0,1] and iInObstacleBound and i1InObstacleBound:
                    Dirichlet.append([n2,n0,p2,p0])
                # adding bottom wall of the boxcutout as Dirichlet edges
                if i==boxcutout[1,0]-1 and jInObstacleBound and j1InObstacleBound:
                    Dirichlet.append([n3,n2,p3,p2])
                # adding top wall of the boxcutout as Dirichlet edges
                if i==boxcutout[1,1] and jInObstacleBound and j1InObstacleBound:
                    Dirichlet.append([n0,n1,p0,p1])



    periodicPairs[k,1] = gridTrans[Ny * (Nx + 1) + Nx,1]
    periodicPairs[k,0] = gridTrans[Ny * (Nx + 1),1]

    elements=np.array(elements,dtype=int)
    Dirichlet=np.array(Dirichlet,dtype=int)

    return coordinates,elements,Dirichlet,periodicPairs,gridTrans





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

    # (nE,2) array of first vertex of elements and corresponding edge vectors
    c1 = coordinates[elements[:,0],:]
    d21 = coordinates[elements[:,1],:] - c1
    d31 = coordinates[elements[:,2],:] - c1
    # (nE,) array of element areas
    area4 = 2*(d21[:,0]*d31[:,1]-d21[:,1]*d31[:,0])
    # assembly of stiffness matrix
    a = np.sum(d21*d31,1)/area4
    b = np.sum(d31*d31,1)/area4
    c = np.sum(d21*d21,1)/area4
    S=np.transpose(np.array([(-2)*a+b+c, a-b , a-c , a-b , b , -a , a-c , -a , c])) #values for assembly of the stiffness matrix
    I = elements[:,[3,4,5,3,4,5,3,4,5]] # row indices
    J = elements[:,[3,3,3,4,4,4,5,5,5]] # column indices
    A=sparse.csc_matrix((S.flatten('F'), (I.flatten('F'), J.flatten('F'))), shape=(nC, nC))
    # incorporating Dirichlet conditions
    if uD != None:
        DirNodes=np.unique(Dirichlet)#-1
        x[DirNodes] = uD(coordinates[DirNodes,0],coordinates[DirNodes,1])
    # Assembly of the right hand side
    #sT = (c1+d21+d31)/3
    #fsT = (area4/4)*f(sT[:,0],sT[:,1])/3
    #RHS = np.transpose(np.array([fsT,fsT,fsT]))
    #nodesidxPer=elements[:,3:6]
    #b=np.bincount(nodesidxPer.flatten('F'),weights=RHS.flatten('F'),minlength=nC) - (A @ x) # (?) problematic if uD not periodic
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





# solve the Poisson equation through
# the code is only for educational purposes/debugging, as the incremental construction of the stiffness matrix is quite slow
#
def solveLaplace_periodicPipe_slow(coordinates,elements,Dirichlet,periodicPairs,f,uD=None):
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
    # Assembly of stiffness matrix
    A = sparse.lil_matrix((nC,nC))  # Sparse matrix format
    for i in range(nE):
        # retrieve the coordinates of the nodes of the i-th element
        # Nodes must be ordered counter clockwise!
        nodesidx = elements[i,0:3]
        nodes=coordinates[nodesidx,:]
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





# evaluate solution on uniform grids
#
def evalFEM_periodicPipe_unifGrid(z,u,Nx,Ny,Lx=1.0,Ly=1.0,interpolation='convexCombination',uD=None,boxcutout=np.array([]),gridTrans=np.array([])):
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
    # Compute an indicator function to later enforce the Dirichlet conditions beyond the top and the bottom
    DirIndicator=np.ones(np.shape(z)[0])
    DirIndicator[I<0]=0
    DirIndicator[I>Ny-1]=0
    # Compute an indicator function for the complement of the box and update the Dirichlet indicator
    BoxIndicator=np.ones(np.shape(z)[0])
    if np.shape(boxcutout)==(2,2):
        inXbounds=np.logical_and(J>=boxcutout[0,0],J<boxcutout[0,1])
        inYbounds=np.logical_and(I>=boxcutout[1,0],I<boxcutout[1,1])
        BoxIndicator[np.logical_and(inXbounds,inYbounds)]=0
    DirIndicator=BoxIndicator*DirIndicator
    # compute a value vector for the nodes beyond the boundary
    DirVals=np.zeros(np.shape(z)[0])
    if uD != None:
        DirVals=uD(z[:,0],z[:,1])
    # Fix associated grid points for those inputs (by setting them to zero) which lie beyond the Dirichlet boundary for later computations
    I=I*DirIndicator
    # Change datatype of index to int
    Iint=I.astype(int)
    Jint=J.astype(int)
    # Node indices (for full mesh) for the quadrant of the mesh in which x lies
    n0 = Iint * (Nx + 1) + Jint
    n1 = Iint * (Nx + 1) + Jint + 1
    n2 = Iint * (Nx + 1) + Jint + (Nx + 1)
    n3 = Iint * (Nx + 1) + Jint + (Nx + 1) + 1
    if np.shape(boxcutout)==(2,2): # gridtranslation in case of boxcutout
        n0 = gridTrans[Iint * (Nx + 1) + Jint,1]
        n1 = gridTrans[Iint * (Nx + 1) + Jint + 1,1]
        n2 = gridTrans[Iint * (Nx + 1) + Jint + (Nx + 1),1]
        n3 = gridTrans[Iint * (Nx + 1) + Jint + (Nx + 1) + 1,1]
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
    

# evaluate the orthogonal gradient/curl on uniform grids
#
def orthogradFEM_periodicPipe_unifGrid(z,u,Nx,Ny,Lx=1.0,Ly=1.0,interpolation='convexCombination',uD=None,boxcutout=np.array([]),gridTrans=np.array([])):
    """
        Evaluates a function U(z) at an arbitrary point z=(x,y) given by the the finite element discretization (u,coordinates,elements) over a uniform grid
            of the periodic pipe, where u are the coordinates of U in the finite element basis.

        params:
            z               float type array, storing the coordinate of the evaluation points
            u               float type array, storing the coordinates of the function to be evaluated

        
        Returns:
            curlU               float type array, storing the evluations
    """
    # initiazlize ouput vector
    curlU=np.zeros(np.shape(z))
    # Compute x index of the quadrant in which z lies (mod is used to enforce periodicity)
    J=np.mod(np.floor((Nx*z[:,0])/Lx),Nx)
    # Compute y index of the quadrant in which z lies
    I=np.floor(Ny*z[:,1]/Ly)
    # Construct two (nZ,4)-matrix, where nZ is the number of evaluation points,
    # such that the i-th row stores the values of the x/y coordinates of the 4 quadrant nodes
    Zx=np.column_stack((J*Lx/Nx,(J+1)*Lx/Nx,J*Lx/Nx,(J+1)*Lx/Nx))
    Zy=np.column_stack((I*Ly/Ny,I*Ly/Ny,(I+1)*Ly/Ny,(I+1)*Ly/Ny))
    # Compute an indicator function to later enforce the Dirichlet conditions beyond the top and the bottom
    DirIndicator=np.ones(np.shape(z)[0])
    DirIndicator[I<0]=0
    DirIndicator[I>Ny-1]=0
    # Compute an indicator function for the complement of the box and update the Dirichlet indicator
    BoxIndicator=np.ones(np.shape(z)[0])
    if np.shape(boxcutout)==(2,2):
        inXbounds=np.logical_and(J>=boxcutout[0,0],J<boxcutout[0,1])
        inYbounds=np.logical_and(I>=boxcutout[1,0],I<boxcutout[1,1])
        BoxIndicator[np.logical_and(inXbounds,inYbounds)]=0
    DirIndicator=BoxIndicator*DirIndicator
    # Fix associated grid indices for those inputs (by setting them to zero) which lie beyond the Dirichlet boundary for later computations
    I=I*DirIndicator
    # Change datatype of index to int
    Iint=I.astype(int)
    Jint=J.astype(int)
    # Node indices (for full mesh) for the quadrant of the mesh in which x lies
    n0 = Iint * (Nx + 1) + Jint
    n1 = Iint * (Nx + 1) + Jint + 1
    n2 = Iint * (Nx + 1) + Jint + (Nx + 1)
    n3 = Iint * (Nx + 1) + Jint + (Nx + 1) + 1 
    if np.shape(boxcutout)==(2,2): # gridtranslation in case of boxcutout
        n0 = gridTrans[Iint * (Nx + 1) + Jint,1]
        n1 = gridTrans[Iint * (Nx + 1) + Jint + 1,1]
        n2 = gridTrans[Iint * (Nx + 1) + Jint + (Nx + 1),1]
        n3 = gridTrans[Iint * (Nx + 1) + Jint + (Nx + 1) + 1,1]
    # construct a (nZ,4)-matrix, where nZ is the number of evaluation points,
    # such that the i-th row stores the values of u on the points of the surrounding quadrant
    Uvals=np.column_stack((DirIndicator*u[n0],DirIndicator*u[n1],DirIndicator*u[n2],DirIndicator*u[n3]))
    if uD != None:
        DirVals=np.column_stack(( uD(Zx[:,0],Zy[:,0]),uD(Zx[:,1],Zy[:,1]),uD(Zx[:,2],Zy[:,2]),uD(Zx[:,3],Zy[:,3]) ))
        Uvals[DirIndicator<1/2,:]=DirVals[DirIndicator<1/2,:]
    # lower triangle [n0,n1,n3], uper triangle [n0,n3,n2]
    # --> on the triangle T=[z_0,z_2,z_3], one has the identity
    #     2|T| curl U = (z_2-z_0)*(u_1-u_0) - (z_1-z_0)*(u_2-u_0)
    # compute area of quadrant (=2|T|)
    #area = (np.linalg.norm(z1-z0,axis=1) * np.linalg.norm(z2-z0,axis=1))
    #area = (z1[:,0]-z0[:,0])*(z3[:,1]-z0[:,1]) - (z1[:,1]-z0[:,1])*(z3[:,0]-z0[:,0])
    area = Lx/(Nx+1) * Ly/(Ny+1)
    # curl of lower triangle
    curl1 = np.column_stack(( (Zx[:,3]-Zx[:,0])*(Uvals[:,1]-Uvals[:,0])-(Zx[:,1]-Zx[:,0])*(Uvals[:,3]-Uvals[:,0]) , (Zy[:,3]-Zy[:,0])*(Uvals[:,1]-Uvals[:,0])-(Zy[:,1]-Zy[:,0])*(Uvals[:,3]-Uvals[:,0]) ))/area
    curl2 = np.column_stack(( (Zx[:,2]-Zx[:,0])*(Uvals[:,3]-Uvals[:,0])-(Zx[:,3]-Zx[:,0])*(Uvals[:,2]-Uvals[:,0]) , (Zy[:,2]-Zy[:,0])*(Uvals[:,3]-Uvals[:,0])-(Zy[:,3]-Zy[:,0])*(Uvals[:,2]-Uvals[:,0]) ))/area
    #compute interpolation
    if interpolation=='AverageOfBothDiffsInQuadrant':      # Compute the average of both differentialsin quadrant
        curlU=(curl1+curl2)/2
    else:      # Compute the bilinear interpolation, the standard input
        lambdaX=np.remainder((Nx*z[:,0]),Lx)
        lambdaY=np.remainder((Ny*z[:,1]),Ly)
        # Define indicator to indicate the triangle in which z lies, 1 when in lower triangle, 0 when in upper
        upperOrLowerTriangle = np.column_stack(( np.heaviside(lambdaX-lambdaY,0),np.heaviside(lambdaX-lambdaY,0) )) 
        curlU=upperOrLowerTriangle*curl1 + (1-upperOrLowerTriangle)*curl2   
    
    return curlU









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








# 'evaluate' the curl/orthodifferential (−du/dy, du/dx) of the solution at points z
#   --> note that the solution is not globally in C^{1}, thus evaluations only make sense inside of the elements
#
def orthogradFEM_Torus_unifGrid(z,u,coordinates,Nx,Ny,Lx=1.0,Ly=1.0,interpolation='AverageOfBothDiffsInQuadrant'):
    """
        Compute nabla^{ortho}U(z)=(-d_{x_2}U(z),d_{x_1}U(z)) for a function U, given through its coordinates u in the finite element basis

        params:
            z               float type array, storing the coordinate of the evaluation points
            u               float type array, storing the coordinates of the function to be evaluated

        
        Returns:
            U               float type array, storing the evluations of the orthograd

    """
    # initiazlize ouput vector
    U=np.zeros(np.shape(z))
    # Compute x index of the quadrant in which z lies (mod is used to enforce periodicity)
    J=np.mod(np.floor((Nx*z[:,0])/Lx),Nx)
    # Compute y index of the quadrant in which z lies
    I=np.mod(np.floor(Ny*z[:,1]/Ly),Ny) 
    # Change datatype of index to int
    Iint=I.astype(int)
    Jint=J.astype(int)
    # Node indices for the quadrant of the mesh in which z lies
    n0 = Iint * (Nx + 1) + Jint
    n1 = n0 + 1
    n2 = n0 + (Nx + 1)
    n3 = n2 + 1
    # corresponding coordinate vectors
    z0 = coordinates[n0,:]
    z1 = coordinates[n1,:]
    z2 = coordinates[n2,:]
    z3 = coordinates[n3,:]
    # lower triangle [n0,n1,n3], uper triangle [n0,n3,n2]
    # --> on the triangle T=[z_0,z_2,z_3], one has the identity
    #     2|T| curl U = (z_2-z_0)*(u_1-u_0) - (z_1-z_0)*(u_2-u_0)
    # compute area of quadrant (=2|T|)
    #area = (np.linalg.norm(z1-z0,axis=1) * np.linalg.norm(z2-z0,axis=1))
    #area = (z1[:,0]-z0[:,0])*(z3[:,1]-z0[:,1]) - (z1[:,1]-z0[:,1])*(z3[:,0]-z0[:,0])
    area = Lx/(Nx+1) * Ly/(Ny+1)
    # curl of lower triangle
    curl1 = ((z3-z0)*np.transpose(np.array([(u[n1]-u[n0]),(u[n1]-u[n0])])) - (z1-z0)*np.transpose(np.array([(u[n3]-u[n0]),(u[n3]-u[n0])])))/area
    # curl of upper triangle
    curl2 = ((z2-z0)*np.transpose(np.array([(u[n3]-u[n0]),(u[n3]-u[n0])])) - (z3-z0)*np.transpose(np.array([(u[n2]-u[n0]),(u[n2]-u[n0])])))/area
    #compute interpolation
    if interpolation=='DiffInTriangle':      # Compute the bilinear interpolation, the standard input
        lambdaX=np.remainder((Nx*z[:,0]),Lx)
        lambdaY=np.remainder((Ny*z[:,1]),Ly)
        upperOrLowerTriangle = np.transpose(np.array([np.heaviside(lambdaX-lambdaY,0),np.heaviside(lambdaX-lambdaY,0)])) # 1 when in lower, 0 when in upper
        U=upperOrLowerTriangle*curl1 + (1-upperOrLowerTriangle)*curl2   
    elif interpolation=='AverageOfBothDiffsInQuadrant':      # Compute the average of both differentialsin quadrant
        U=(curl1+curl2)/2
    
    return U
    
   



