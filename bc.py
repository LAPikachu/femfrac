import numpy as np

def rigid_body_dofs_2d(nelx: int, nely: int) -> np.ndarray:
    """
    Select three degrees of freedom whose constraint removes the rigid body
    modes (two translations and one rotation) of a regular 2d mesh.

    Without such a constraint the stiffness matrix has three zero eigenvalues
    and the solution of the linear system is not unique: the solver returns an
    arbitrary representative of the family of solutions, which differs from
    the "physical" one by a rigid body motion. For a pure shear test this
    shows up as a skewed displacement field.

    The constrained dofs are taken at interior nodes whenever possible,
    because interior nodes carry no applied load. The constraint does not
    perturb the stress field: the reaction forces at the constrained dofs
    vanish because the constrained solution differs from the unconstrained one
    by a rigid body motion only.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.

    Returns
    -------
    np.ndarray
        indices of three fixed dofs.

    """
    # the node (i,j) has the index i*(nely+1)+j and the dofs 2n (x), 2n+1 (y)
    ic, jc = nelx//2, nely//2
    n_center = ic*(nely+1) + jc
    # the third dof has to be off the horizontal centre line, otherwise the
    # rotation about the centre of the mesh is not constrained
    candidates = []
    for i, j in ((ic+1, jc+1), (ic-1, jc-1), (ic+1, jc-1), (ic-1, jc+1),
                 (ic, jc+1), (ic, jc-1)):
        if 0 <= i <= nelx and 0 <= j <= nely and j != jc:
            candidates.append((i, j))
    # prefer interior nodes, they carry no applied load
    interior = [c for c in candidates
                if 1 <= c[0] <= nelx-1 and 1 <= c[1] <= nely-1]
    i, j = (interior + candidates)[0]
    return np.array([2*n_center, 2*n_center+1, 2*(i*(nely+1)+j)], dtype=int)

def simple_shear_2d(nelx: int, nely: int,
                   ndof: int, 
                   **kwargs: Any
                   ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,None]:
    """

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    ndof : int
        number of degrees of freedom.
 
    Returns
    -------
    u : np.ndarray
        array of zeros for state variable (displacement, temperature) to be
        filled of shape (ndof).
    f : np.ndarray
        array of zeros for state flow variables (forces, flow).
    fixed : np.ndarray
        indices of fixed dofs (nfixed).
    free : np.ndarray
        indices of free dofs (ndofs - nfixed).
    springs : None
        example has no springs.

    """
    if nely%2 !=0:
        raise ValueError("This example works only for nely equal to an even number.")
    # BC's
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # applied force for simple shear
    # f[1:(2*(nely+1)):2] = -1.
    f[-1:-(2*(nely+1)):-2] = 1. # right side
    # fixed dofs
    # the left side is fixed in y-direction, which removes the translation in
    # y-direction only. the two remaining rigid body modes (the translation
    # in x-direction and the rotation) are removed by pinning the x-displacement
    # of two nodes of the left side with different y-coordinates. the reaction
    # forces at the pinned dofs are the (physically meaningful) reactions of
    # the support.
    fixed = np.arange(1, (nely+1)*2, 2) # left side, y-direction
    fixed = np.hstack([fixed, [0, 2*nely]]) # x-direction of two left side nodes
    fixed = np.unique(fixed)
    '''
    fixed = [np.arange(0,(nely+1)*2,2)]
    fixed += [nely+1,ndof-nely-1]
    fixed = np.hstack(fixed)
    '''
    #
    return u,f,fixed,np.setdiff1d(dofs,fixed),None
    

def pure_shear_2d(nelx: int, nely: int,
                   ndof: int, sign = 1, 
                   **kwargs: Any
                   ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,None]:
    """

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    ndof : int
        number of degrees of freedom.
    sign : int
        signe of the shear applied
    Returns
    -------
    u : np.ndarray
        array of zeros for state variable (displacement, temperature) to be
        filled of shape (ndof).
    f : np.ndarray
        array of zeros for state flow variables (forces, flow).
    fixed : np.ndarray
        indices of fixed dofs (nfixed).
    free : np.ndarray
        indices of free dofs (ndofs - nfixed).
    springs : None
        example has no springs.

    """
    if nely%2 !=0:
        raise ValueError("This example works only for nely equal to an even number.")
    # BC's
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # applied force for pure shear
    f[1:(2*(nely+1)):2] = -1.*sign # left side f_y = -1 
    f[-1:-(2*(nely+1)):-2] = 1.*sign # right side f_y = +1
    f[(nelx*2)::(nelx+1)*2] = -1.*sign # bottom side f_x = -1
    f[::(nelx+1)*2] = 1.*sign # top part f_x = +1
    # fixed dofs
    # the three rigid body modes (two translations, one rotation) have to be
    # constrained, otherwise the stiffness matrix is singular and the solution
    # of the linear system is not unique (cf. rigid_body_dofs_2d)
    fixed = rigid_body_dofs_2d(nelx, nely)
    '''
    fixed = [np.arange(1, (nely+1)*2, 2)] # left side
    fixed += [np.arange(nelx*(nely+1)*2 + 1 , nelx*(nely+2)*2 + 2,2)] # right side 
    fixed += [np.arange(0, (nelx + 1)*(nely+1)*2, (nely + 1)*2)] # top 
    fixed += [np.arange(nely*2 , (nelx+1)*(nely+1)*2, (nely + 1)*2)] # bottom 
    fixed = np.hstack(fixed)
    '''
    return u,f,fixed,np.setdiff1d(dofs,fixed),None
    
def tensiletest_2d(nelx: int, nely: int,
                   ndof: int, sign = 1, 
                   **kwargs: Any
                   ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,None]:
    """

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    ndof : int
        number of degrees of freedom.
    sign : int
        signe of the shear applied
    Returns
    -------
    u : np.ndarray
        array of zeros for state variable (displacement, temperature) to be
        filled of shape (ndof).
    f : np.ndarray
        array of zeros for state flow variables (forces, flow).
    fixed : np.ndarray
        indices of fixed dofs (nfixed).
    free : np.ndarray
        indices of free dofs (ndofs - nfixed).
    springs : None
        example has no springs.

    """
    if nely%2 !=0:
        raise ValueError("This example works only for nely equal to an even number.")
    # BC's
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # applied force 
    f[-2:-(2*(nely+2)):-2] = 1.*sign # right side f_y = +1
    # fixed dofs
    fixed = [np.arange(0, (nely+1)*2, 2)] # left side
    fixed += [np.arange(1, (nelx + 1)*(nely+1)*2, (nely + 1)*2)] # top 
    fixed += [np.arange(nely*2 + 1 , (nelx+1)*(nely+1)*2, (nely + 1)*2)] # bottom 
    fixed = np.hstack(fixed)
    '''
    fixed += [np.arange(nelx*(nely+1)*2 + 1 , nelx*(nely+2)*2 + 2,2)] # right side 
    fixed += [np.arange(0, (nelx + 1)*(nely+1)*2, (nely + 1)*2)] # top 
    fixed += [np.arange(nely*2 , (nelx+1)*(nely+1)*2, (nely + 1)*2)] # bottom 
    fixed = np.hstack(fixed)
    '''
    return u,f,fixed,np.setdiff1d(dofs,fixed),None
