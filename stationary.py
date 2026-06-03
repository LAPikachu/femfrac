# SPDX-License-Identifier: GPL-3.0-or-later
#
import numpy as np
# set up finite element problem
from topoptlab.fem import create_matrixinds
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
# heatplate example 
# default application case that provides boundary conditions, etc.
from topoptlab.example_bc.heat_conduction import rectangle_2d, heatplate_2d, heatplate_3d
# different elements/physics
from topoptlab.elements.poisson_2d import lk_poisson_2d
from topoptlab.elements.poisson_3d import lk_poisson_3d

from topoptlab.example_bc.lin_elast import tensiletest_2d, tensiletest_3d
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d
from topoptlab.elements.linear_elasticity_3d import lk_linear_elast_3d
# generic functions for solving phys. problem
from topoptlab.fem import assemble_matrix,apply_bc
from topoptlab.solve_linsystem import solve_lin
# output final design to a Paraview readable format
from topoptlab.output_designs import export_vtk
#
from topoptlab.material_interpolation import simp
from topoptlab.fem import interpolate
from topoptlab.elements.bilinear_quadrilateral  import shape_functions_dxi, invjacobian, invjacobian_rectangle

from femfrac.fracture import get_vm_stress, weaken_structure
# debugging
import os
# MAIN DRIVER
def run_tensile_test(nelx, 
                     nely, 
                     nelz=None,
                     xPhys=None, 
                     penal=3, 
                     kmax=1.0, 
                     kmin=1e-9, 
                     lin_solver="cvxopt-cholmod", 
                     preconditioner=None,
                     assembly_mode="lower", 
                     l=1.,
                     file="stationary_heatconduction",
                     thresholds=None,
                     export=True):
    """
    Run a single finite element simulation on a regular grid performing a 
    tensile test pulling along x-direction.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None
        number of elements in z direction. Only important if ndim is 3.
    xPhys : np.ndarray
        physical SIMP density
    penal : float
        penalty exponent for the SIMP method.
    kmax : float
        (maximum) heat conductivity. If xPhys is None, all elements take this 
        value.
    kmin : float
        minimum heat conductivity for the modified SIMP approach.
    solver : str
        solver for linear systems. Check function lin solve for available 
        options.
    preconditioner : str or None
        preconditioner for linear systems. 
    assembly_mode : stshape_functions_dxir
        whether full or only lower triangle of linear system / matrix is 
        created.
    l : float or tuple of length (ndim) or np.ndarray of shape (ndim)
        side lengths of each element
    file : str
        name of output files
    export : bool
        if True, export design as vtk file.

    Returns
    -------
    None.

    """
    #
    if nelz is None:
        ndim = 2
        lk = lk_poisson_2d
        create_edofMat = create_edofMat2d
        bc = heatplate_2d
    else:
        ndim = 3
        lk = lk_poisson_2d
        create_edofMat = create_edofMat3d
        bc = heatplate_3d
    # 
    if isinstance(l,float):
        l = np.array( [l for i in np.arange(ndim)])
    # total number of design variables/elements
    n = np.array([nelx,nely,nelz][:ndim])
    #
    if xPhys is None:
        xPhys = np.ones((np.prod(n).astype(int),1), 
                        dtype=float, order='F')
    # get element stiffness matrix and element of freedom matrix
    KE = lk(k=1.0, 
            l=l)
    # infer nodal degrees of freedom assuming that we have 4/8 nodes in 2/3
    nd_ndof = int(KE.shape[0]/2**ndim)
    # number of degrees of freedom
    ndof = np.prod(n+1)*nd_ndof
    # element degree of freedom matrix plus some helper indices
    edofMat, n1, n2, n3, n4 = create_edofMat(nelx=nelx,
                                             nely=nely,
                                             nelz=nelz,
                                             nnode_dof=nd_ndof)
    # Construct the index pointers for the coo format
    iK,jK = create_matrixinds(edofMat,
                              mode=assembly_mode)
    if assembly_mode == "lower":
        assm_indcs = np.column_stack(np.tril_indices_from(KE))
        assm_indcs = assm_indcs[np.lexsort( (assm_indcs[:,0],assm_indcs[:,1]) )]
    elif assembly_mode == "full":
        pass 
    else:
        raise ValueError("Unknown assembly_mode: ", assembly_mode)
    # BC's and support
    u,f,fixed,free,_ = bc(nelx=nelx,
                          nely=nely,
                          nelz=nelz,
                          ndof=ndof)
    # interpolate material properties
    Kes = kmax*KE[None,:,:]*simp(xPhys=xPhys, 
                                 eps=kmin/kmax, 
                                 penal=penal)[:,:,None]
    # entries of stiffness matrix
    if assembly_mode == "full":
        # this here is more memory efficient than Kes.flatten() as it
        # provides a view onto the original Kes array instead of a copy
        sK = Kes.reshape(np.prod(Kes.shape))
    elif assembly_mode == "lower":
        sK = Kes[:,
                 assm_indcs[:,0],
                 assm_indcs[:,1]].reshape( np.prod(n)*int(KE.shape[-1]/2*(KE.shape[-1]+1)))
    # assemble completely
    KE = assemble_matrix(sK=sK,
                         iK=iK,
                         jK=jK,
                         ndof=ndof,
                         solver=lin_solver,
                         springs=None)
    rhs = f
    # apply boundary conditions to matrix
    KE = apply_bc(K=KE,
                  solver=lin_solver,
                  free=free,fixed=fixed)
    # solve linear system. fact is a factorization and precond a preconditioner
    u[free, :], fact, precond, = solve_lin(K=KE, 
                                           rhs=rhs[free], 
                                           solver=lin_solver,
                                           preconditioner=preconditioner)
    #
    '''
    print(edofMat[0])
    print(len(T))
    print(T[0])
    print(T[0].shape)
    print(T[[[0,1,2], [1,2,3], [5,6,7]],0])    
    print(T[edofMat, 0].shape) 
    delta_N_e = shape_functions_dxi(0,0)
    print(shape_functions_dxi(0,0).shape)
    print(T[edofMat,0][0:2])
    print(shape_functions_dxi(0,0))
    J_e_inv = invjacobian_rectangle(2,2) 
    B_e = delta_N_e @ J_e_inv
    print((T[edofMat,0][...,None]*B_e))
    print((T[edofMat,0][...,None]*B_e).shape)
    print((T[edofMat,0][...,None]*B_e).sum(axis=1).shape)
    print((T[edofMat,0][...,None]*shape_functions_dxi(0,0)).sum(axis=1).shape)
    #print(([[1,1,1,1,1,1]][...,None]*shape_functions_dxi(0,0)).sum(axis=1))
    '''
    sigma_von_mises = get_vm_stress(u, xPhys, edofMat, nelz)
    if export:
        export_vtk(filename=file,
                   nelx=nelx,
                   nely=nely,
                   nelz=nelz,
                   xPhys=xPhys,
                   elem_size=l,
                   u=u,
                   f=f, 
                   stress_vm=sigma_von_mises)
    
    return u, xPhys, edofMat, sigma_von_mises

def get_x_coordinate(i, nelx, nely, nelz):
    if nelz==None:
        return i//nelx
    else:
        pass

def get_y_coordinate(i, nelx, nely, nelz):
    if nelz==None:
        return i%nelx
    else:
        pass

def burn_max_element(sigma_von_mises, xPhys, thresholds):
    # interpolate over element
    #sigma_von_mises = von_mises_stress(T, xPhys, edofMat, nelz)
    ratios = sigma_von_mises/thresholds
    max_element  = np.argmax(ratios)
    xPhys = weaken_structure(max_element, xPhys)
    return xPhys 

def make_xPhys_gradient(xPhys, nelx, nely, nelz):
    pass

def ruler_sequence(levels: int) -> list[int]:
    if levels == 0:
        return [0]
    sub = ruler_sequence(levels - 1)
    return sub + [levels] + sub

def insert_hierarchical_gaps(xPhys, nelx, nely, nelz, steps, delta_c, gap_width):
    if nelz==None:
        n=nelx*nely 
        heights = np.array(ruler_sequence(steps))
        height_factor = (nely//2)/max(heights)
        heights = heights*height_factor
        heights = np.asarray(heights, dtype="int")
        heights = np.repeat(heights, gap_width)
        middle_x = get_x_coordinate(n/2, nelx, nely, nelz)
        for i, c in enumerate(xPhys):
            x_coord = get_x_coordinate(i, nelx, nely, nelz)
            y_coord = get_y_coordinate(i, nelx, nely, nelz)
            delta_middle = middle_x - x_coord

            if x_coord < heights[y_coord]:
                xPhys[i] = delta_c 
        return xPhys, heights
    else:
        raise NotImplementedError("Only 2D implemented")

def insert_random_voids(xPhys, n, nelx, nely, nelz):
    index_array = np.arange(0, nelx*nely)
    index_array = index_array[index_array > nelx//2*nely] #only indices in substrate
    index_array = index_array[index_array % 2 == 0] # only even indices
    index_array = index_array[index_array % 2*nely < nely] # only every second height
    index_array = index_array[index_array < (nelx-1)*nely] # not in upper x-bound
    index_array = index_array[index_array % nely != nely-1] # not in in upper y-bound
    index_array = np.random.choice(index_array, size=n//4)
    for i in index_array:
        xPhys[i] = 0
        xPhys[i+1] = 0
        xPhys[i+nely] = 0
        xPhys[i+nely+1] = 0
    return xPhys

def make_precrack(xPhys:np.arry[float], nelx, nely, nelz, xpos: int, length: int = 10, width: int = 2):
    for i in range(0,width):
        precrack_i = int(nely*xpos + nely*i)
        precrack_e = int(precrack_i + length)
        xPhys[precrack_i:precrack_e] = 0
    return xPhys 

if __name__ == "__main__":
    #
    nelx = 4
    nely = 4
    nelz = None
    steps = 1
    #
    if nelz is None:
        ndim = 2
    else:
        ndim=3
    import sys
    if len(sys.argv)>1: 
        nelx = int(sys.argv[1])
    if len(sys.argv)>2: 
        nely = int(sys.argv[2])
    if len(sys.argv)>3: 
        nelz = int(sys.argv[3])
    # 
    n = np.array([nelx,nely,nelz][:ndim])
    thresholds = np.random.rand(np.prod(n).astype(int),1) *0.1 + 0.9
    xPhys = np.ones((np.prod(n).astype(int),1))
    #xPhys = make_gradient(xPhys, nelx, nely, nelz)
    xPhys, heights = insert_hierarchical_gaps(xPhys, nelx, nely, nelz, 5, 0, 4)
    xPhys = insert_random_voids(xPhys, np.sum(heights),nelx, nely, nelz)
    #xPhys = make_precrack(xPhys, nelx, nely, nelz, xpos=nelx/2, length=20, width=1)
    # get element stiffness matrix and element o
    u, xPhys, edofMat, sigma_von_mises = run_tensile_test(nelx=nelx, nely=nely, nelz=nelz,xPhys=xPhys ,file=f"visualisation/tensile/hierarchical/tensile_test_0", thresholds=thresholds)
    xPhys = burn_max_element(sigma_von_mises, xPhys, thresholds )
    for step in range(1,steps):
        u, xPhys, edofMat, sigma_von_mises =  run_tensile_test(nelx=nelx, nely=nely, nelz=nelz, xPhys=xPhys, file=f"visualisation/tensile/hierarchical/tensile_test_{step}", thresholds=thresholds)
        xPhys = burn_max_element(sigma_von_mises, xPhys, thresholds)
        
