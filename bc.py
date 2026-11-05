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

def sparse_rigid_body_dofs_2d(nelx: int, nely: int) -> np.ndarray:
    """
    Select three degrees of freedom whose constraint removes the rigid body
    modes (two translations and one rotation) of a regular 2d mesh, with the
    constrained nodes placed halfway between the centre and opposite corners.

    In contrast to :func:`rigid_body_dofs_2d`, which clusters the constraints
    near the centre of the mesh, this variant puts the fully constrained node
    at the lower-left quarter point and the third dof at the upper-right
    quarter point. Only three dofs are constrained, so the constraints stay
    sparse, but the reaction forces are spread over the mesh instead of
    stiffening one local region. The quarter points keep the constraints away
    from the boundary without putting them back into the centre.

    Interior nodes are preferred because they carry no applied load. The node
    carrying the third dof must not lie on the horizontal line through the
    fully constrained node; otherwise the rotation is not constrained.

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
    # place the nodes halfway between the centre and opposite corners, keeping
    # at least one element between a node and the boundary where possible
    iq1 = min(max(round(nelx/4), 1), max(nelx-1, 1))
    jq1 = min(max(round(nely/4), 1), max(nely-1, 1))
    iq2 = min(max(round(3*nelx/4), 1), max(nelx-1, 1))
    jq2 = min(max(round(3*nely/4), 1), max(nely-1, 1))

    # use the quarter points, falling back to interior and boundary corners
    # for meshes that have no suitable interior nodes
    pairs = (
        ((iq1, jq1), (iq2, jq2)),
        ((iq1, jq2), (iq2, jq1)),
        ((1, 1), (nelx-1, nely-1)),
        ((1, nely-1), (nelx-1, 1)),
        ((0, 0), (nelx, nely)),
        ((0, nely), (nelx, 0)),
    )
    for (i1, j1), (i2, j2) in pairs:
        valid = (0 <= i1 <= nelx and 0 <= j1 <= nely and
                 0 <= i2 <= nelx and 0 <= j2 <= nely)
        # the nodes have to be distinct and off each other's horizontal line
        if valid and (i1, j1) != (i2, j2) and j1 != j2:
            break
    n1 = i1*(nely+1) + j1
    n2 = i2*(nely+1) + j2
    return np.array([2*n1, 2*n1+1, 2*n2], dtype=int)

def force_simple_shear_2d(nelx: int, nely: int,
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
    f[::4] = 1. 
    f[2::4] = -1. 
    # fixed dofs
    # in x-direction and the rotation) are removed by pinning the x-displacement
    # of two nodes of the left side with different y-coordinates. the reaction
    # forces at the pinned dofs are the (physically meaningful) reactions of
    # the support.
    fixed = sparse_rigid_body_dofs_2d(nelx, nely)
    
    return u,f,fixed,np.setdiff1d(dofs,fixed),None
 
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

def force_pure_shear_2d(nelx: int, nely: int,
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
    f[::4] = 1. 
    f[1::4] = 1. 
    f[2::4] = -1. 
    f[3::4] = -1. 
    # fixed dofs
    # in x-direction and the rotation) are removed by pinning the x-displacement
    # of two nodes of the left side with different y-coordinates. the reaction
    # forces at the pinned dofs are the (physically meaningful) reactions of
    # the support.
    fixed = sparse_rigid_body_dofs_2d(nelx, nely)
    
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
    f[1:(2*(nely+1)):2] = -1.*sign #sparse_rigid_body_dofs_2d left side f_y = -1 
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
