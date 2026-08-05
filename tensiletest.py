# SPDX-License-Identifier: GPL-3.0-or-later
#
import os
import numpy as np
# set up finite element problem
from topoptlab.fem import create_matrixinds
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
# default application case that provides boundary conditions, etc.
from topoptlab.example_bc.lin_elast import tensiletest_2d,tensiletest_3d
# different elements/physics
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d
from topoptlab.elements.linear_elasticity_3d import lk_linear_elast_3d
# generic functions for solving phys. problem
from topoptlab.fem import assemble_matrix,apply_bc
from topoptlab.solve_linsystem import solve_lin
# output final design to a Paraview readable format
from topoptlab.output_designs import export_vtk

from femfrac.fracture import burn_max_element, get_vm_stress, weaken_structure, insert_random_voids, insert_hierarchical_gaps, insert_notch
from femfrac.disorder import get_yield, weibull_redistribute
# MAIN DRIVER
def fem_tensiletest(nelx, nely, nelz=None,
                    xPhys=None, damage=None, f_input=None, penal=3, 
                    Emax=1.0, Emin=1e-9, nu=0.3,
                    KE=None, edofMat=None, ndof=None,
                    lin_solver="cvxopt-cholmod", preconditioner=None,
                    assembly_mode="full", l=1.,
                    file="visualisation/tensile/tensiletest",
                    export=True,
                    thresholds=None):
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
    KE : np.ndarray
        element stiffness matrix
    f_input : np.ndarray
        forces if adjusted by the burn function
    penal : float
        penalty exponent for the SIMP method.
    Emax : float
        (maximum) Young's modulus. If xPhys is None, all elements take this 
        value.
    Emin : float
        minimum Young's modulus for the modified SIMP approach.
    nu : float
        Poissson's ratio.
    solver : str
        solver for linear systems. Check function lin solve for available 
        options.
    preconditioner : str or None
        preconditioner for linear systems. 
    assembly_mode : str
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
    # number of degrees of freedom
    ymax = 1.1
    ymin = ymax*5.
    if nelz is None:
        bc = tensiletest_2d
    else:
        bc = tensiletest_3d
    # BC's and support
    u,f,fixed,free,_ = bc(nelx=nelx,
                          nely=nely,
                          nelz=nelz,
                          ndof=ndof,
                          ymirror=False)
    if f_input is None:
        f = f
    else:
        f = f_input
    # interpolate material properties
    E_scaled = (Emin+(xPhys)** penal*(Emax-Emin))
    yield_scaled = get_yield(damage, ymin, ymax) 
    # entries of stiffness matrix
    rhs = f
    # apply boundary conditions to matrix
    # solve linear system. fact is a factorization and precond a preconditioner
    # assemble completely
    KE = apply_bc(K=KE, solver=lin_solver,
                  free=free,fixed=fixed) 
    u[free, :], fact, precond, = solve_lin(K=KE, 
                                           rhs=rhs[free], 
                                           solver=lin_solver,
                                           preconditioner=preconditioner)
    if isinstance(l,float):
        l = np.array( [l for i in np.arange(ndim)])
    if ndim == 2:
        sigma = f.max()*(nely+1)/l[1]
        print("sigma: ", sigma)
        print("eps_xx theory ",sigma/Emax)
        print("eps_yy theory ",sigma/Emax * nu)
        print("eps_xx measured ",np.abs(u[::2]).max()/(nelx*l[0]) )
        print("eps_yy measured ",np.abs(u[1::2].max())/(nely*l[1]) )
    elif ndim == 3:
        sigma = f.max() * (nely+1)*(nelz+1)/(nely*nelz * l[1]*l[2])
        print("sigma: ",sigma)
        print("eps_xx theory ",sigma/Emax)
        print("eps_yy theory ",sigma/Emax * nu)
        print("eps_zz theory ",sigma/Emax * nu)
        print("eps_xx measured ",np.abs(u[::3]).max()/ (nelx*l[0]) )
        print("eps_yy measured ",np.abs(u[1::3]).max()/ (nely*l[1]) )
        print("eps_zz measured ",np.abs(u[2::3]).max()/ (nelz*l[2]) )
    #
    sigma_vm = get_vm_stress(u, xPhys, edofMat, nelz) 
    print(os.getcwd())
    if export:
        export_vtk(filename=file,
                   nelx=nelx,
                   nely=nely,
                   nelz=nelz,
                   elem_size=l,
                   xPhys=xPhys - damage,
                   u=u,f=f,
                   stress_vm=sigma_vm)
    return u, f, xPhys, edofMat,yield_scaled, sigma_vm 

def get_stiffnessmatrix(nelx, nely, nelz=None,
                        xPhys=None, penal=3, assembly_mode="full", l=1., 
                        Emax=1.0, Emin=1e-9, nu=0.3,):
    if nelz is None:
        ndim = 2
        lk = lk_linear_elast_2d
        create_edofMat = create_edofMat2d
        bc = tensiletest_2d
    else:
        ndim = 3
        lk = lk_linear_elast_3d
        create_edofMat = create_edofMat3d
        bc = tensiletest_3d
    # 
    if isinstance(l,float):
        l = np.array( [l for i in np.arange(ndim)])
    # total number of design variables/elements
    n = np.array([nelx,nely,nelz][:ndim])
    #
    if xPhys is None:
        xPhys = np.ones(np.prod(n).astype(int), 
                        dtype=float, order='F')
    KE = lk(E=1.0, nu=nu, l=l)
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
    E_scaled = (Emin+(xPhys)** penal*(Emax-Emin))
    if assembly_mode == "full":
            sK = (KE.flatten()[:,None]*E_scaled).flatten(order='F')
    #
    lin_solver="cvxopt-cholmod",
    KE = assemble_matrix(sK=sK,
                        iK=iK,
                        jK=jK,
                        ndof=ndof,
                        solver=lin_solver,
                        springs=None)
    return KE, edofMat, ndof, nd_ndof

if __name__ == "__main__":
    #
    nelx = 10
    nely = 10
    nelz = None
    steps=10
    penal=3 
    Emax=1.0
    Emin=1e-9
    nu=0.3    
    l=1.
    #
    import sys
    if len(sys.argv)>1: 
        nelx = int(sys.argv[1])
    if len(sys.argv)>2: 
        nely = int(sys.argv[2])
    if len(sys.argv)>3: 
        nelz = int(sys.argv[3])
    if nelz is None:
        ndim = 2
    else:
        ndim=3
    #
    n = np.array([nelx,nely,nelz][:ndim])
    thresholds = np.random.rand(np.prod(n).astype(int),1) *0.1 + 0.9
    xPhys = np.ones((np.prod(n).astype(int),1))
    xPhys = insert_notch(xPhys=xPhys, length=5)
    #xPhys, heights = insert_hierarchical_gaps(xPhys, nelx, nely, nelz, 5, 0, 4)
    #xPhys = insert_random_voids(xPhys, np.sum(heights),nelx, nely, nelz)
    xPhys = np.squeeze(xPhys, axis=1)
    damage = np.squeeze(np.zeros((np.prod(n).astype(int),1)), axis=1) 
    #thresholds = np.random.rand(np.prod(n).astype(int),1) *0.1 + 0.9
    KE, edofMat, ndof, nd_ndof= get_stiffnessmatrix(nelx, nely, nelz=None, xPhys=xPhys, penal=3, 
                        Emax=Emax, Emin=Emin, nu=nu)
    
    u, f, xPhys, edofMat, yield_scaled, sigma_vm   = fem_tensiletest(nelx=nelx, nely=nely,
                                                                    Emax=Emax, Emin=Emin, penal=penal, l=l, nu=nu,
                                                                    KE=KE, edofMat=edofMat, ndof=ndof,
                                                                    nelz=None, xPhys=xPhys, damage=damage,
                                                                    file="visualisation/tensile/hierarchical/tensile_test_0",
                                                                    thresholds=thresholds)
    f, damage = burn_max_element(f, u, sigma_vm, yield_scaled, edofMat, l,xPhys, damage, penal, Emin, Emax, nu)
   
    for step in range(1,steps):
        u, f, xPhys, edofMat, yield_scaled, sigma_vm  = fem_tensiletest(nelx=nelx, nely=nely,
                                                                       Emax=Emax, Emin=Emin, penal=penal, l=l, nu=nu,
                                                                       KE=KE, edofMat=edofMat, ndof=ndof,
                                                                       nelz=None, xPhys=xPhys, damage=damage,
                                                                       file=f"visualisation/tensile/hierarchical/tensile_test_{step}",
                                                                       thresholds=thresholds)

        f, damage = burn_max_element(f, u, sigma_vm, yield_scaled, edofMat, l,xPhys, damage, penal, Emin, Emax, nu)