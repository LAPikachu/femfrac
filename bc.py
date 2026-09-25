import numpy as np

def sheartest_2d(nelx: int, nely: int,
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
    f[1:(2*(nely+2)):2] = -1.
    f[-1:-(2*(nely+2)):-2] = 1.
    # fixed dofs
    fixed = [np.arange(0,(nely+1)*2,2)]
    fixed += [nely+1,ndof-nely-1]
    fixed = np.hstack(fixed)
    #
    return u,f,fixed,np.setdiff1d(dofs,fixed),None