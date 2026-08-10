import numpy as np
from topoptlab.elements.stress_measures import von_mises_stress
from topoptlab.elements.bilinear_quadrilateral import bmatrix_rectangle

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

def ruler_sequence(levels: int) -> list[int]:
    if levels == 0:
        return [0]
    sub = ruler_sequence(levels - 1)
    return sub + [levels] + sub

def insert_hierarchical_gaps(xPhys, nelx, nely, nelz, steps, delta_c, gap_width): #turn around gaps! 
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

def insert_notch(length, xPhys):
    middle = int(len(xPhys)//2)
    xPhys[middle:int(middle+length)] = 0
    return xPhys 

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

def get_D_planestress(E, nu):
    return E/(1-nu**2) * np.array([[1., nu, 0.],
                                    [nu, 1., 0.],
                                    [0., 0., (1-nu)/2.]])

def get_element_strain(u, edofMat, index, B_e):
    """Average strain in one element from its nodal displacements."""
    ue = u[edofMat[index], 0]          # (8,) nodal dof values, dof order = edofMat's
    return B_e @ ue                     # (3,) -> [eps_xx, eps_yy, gamma_xy]

def damage_element(index, f, u, edofMat, l, damage, penal, Emin, Emax, nu, t=1., relax=1.0):
    """
    Damage one element by adding an eigenstrain-equivalent nodal force to f.
    K is untouched. relax=1.0 fully relaxes the element's current elastic
    strain (i.e. eps0 = -eps_current); relax<1.0 partially relaxes it.
    """
    B_e = bmatrix_rectangle(xi=np.array([0.]), eta=np.array([0.]),
                             a=l, b=l)[0]      # (3,8)
    E_local = Emin + (1 - damage[index])**penal *(Emax- Emin) # TODO what is this ?!
    D = get_D_planestress(E_local, nu)
    eps_current = get_element_strain(u, edofMat, index, B_e)
    eps0 = -relax * eps_current                       # eigenstrain to apply
    Ae = l * l
    f_eig = t * Ae * (B_e.T @ D @ eps0)                # (8,)
    f_new = f.copy()
    f_new[edofMat[index], 0] += f_eig                  # additive scatter
    # TODO increment plastic strain here
    return f_new

def burn_max_element(f, u, eps, eps_pl, sigma_von_mises, thresholds, edofMat, l, damage, penal, Emin, Emax, nu, t=1., relax=1.0):
    # interpolate over element
    #sigma_von_mises = von_mises_stress(T, xPhys, edofMat, nelz)
    ratios = np.array(sigma_von_mises).reshape(-1) - np.array(thresholds)
    max_element_index  = np.argmax(ratios)
    damage[max_element_index] += relax
    f_new = damage_element(max_element_index, f, u, edofMat, l, damage, penal, Emin, Emax, nu, t=1., relax=1.0)
    return f_new, damage, eps, eps_pl

def get_element_strain(u, edofMat, nelz):
    if nelz == None:  
        from topoptlab.elements.bilinear_quadrilateral  import shape_functions_dxi, invjacobian, invjacobian_rectangle
        delta_N_e = shape_functions_dxi(0,0)
        J_e_inv = invjacobian_rectangle(2,2) 
        B_e = delta_N_e @ J_e_inv
    else:
        from topoptlab.elements.trilinear_hexahedron import shape_functions_dxi, invjacobian, invjacobian_cuboid
        delta_N_e = shape_functions_dxi(0,0,0)
        J_e_inv = invjacobian_cuboid(2,2,2) 
        B_e = delta_N_e @ J_e_inv
    if edofMat.shape[1] == 8: #check if 2d-vector-field
        ux  = u[::2]
        uy  = u[1::2]
        edofMatx = edofMat[:,::2]
        edofMaty = edofMat[:,1::2]
        eps_x  = (ux[(edofMatx/2).astype(int),0][...,None]*B_e).sum(axis=1)
        eps_y = (uy[((edofMaty-1)/2).astype(int),0][...,None]*B_e).sum(axis=1)
        eps = np.column_stack((eps_x, eps_y))
        return eps 

def get_vm_stress(u, xPhys, edofMat, nelz : None):
    C = xPhys.reshape(-1,1)
    eps = get_element_strain(u, edofMat, nelz)
    delta_ux = eps[:,:2]
    delta_uy = eps[:,2:]
    sxx = C * delta_ux[:,0][...,None] # [dux/dx, dux/dy]  
    syy = C * delta_uy[:,1][...,None] # [duy/dx, duy/dy]  
    sxy = C * ((delta_ux[:,1][...,None] + delta_uy[:,0][...,None])/2)
    sigma_vm = von_mises_stress(np.squeeze([sxx, syy, sxy], axis=2).T, ndim=2)
#    elif edofMat.shape[1] == 12: #check if 3d-vector-field
#        raise NotImplementedError("Function not implemented for 3d/ 3d-vector-field.")
#        ux  = u[::3]
#        uy  = u[1::3]
#        uy  = u[2::3]
#        edofMatx = edofMat[:,::3]
#        edofMaty = edofMat[:,1::3]
#        edofMatz = edofMat[:,2::3]
#        delta_ux  = (ux[(edofMatx/3).astype(int),0][...,None]*B_e).sum(axis=1)
#        delta_uy = (uy[((edofMaty-1)/3).astype(int),0][...,None]*B_e).sum(axis=1)
#        delta_uz = (uy[((edofMaty-2)/3).astype(int),0][...,None]*B_e).sum(axis=1)
#        du_dx = delta_ux[:,0]
#        sxx = C * delta_ux[:,0] # [dux/dx, dux/dy]  
#        syy = C * delta_uy[:,1] # [duy/dx, duy/dy]  
#        szz = C * delta_uz[:,1] # [duy/dx, duy/dy]  
#        sxy = C * ((delta_ux[:,1] + delta_uy[:,0])/2)
#        sxz = C * ((delta_ux[:,1] + delta_uz[:,0])/2)
#        syz = C * ((delta_uy[:,1] + delta_uz[:,0])/2)
#        sigma_vm = von_mises_stress(np.array([sxx, syy, sxy]), ndim=2)
#    else:
#        delta_u  = (u[edofMat,0][...,None]*B_e)
#        delta_u  = (u[edofMat,0][...,None]*B_e).sum(axis=1)
#        sigma = C * delta_u 
#        #sigma_von_mises_old = np.sum(np.sqrt(sigma ** 2), axis=1, keepdims=True)
#        sigma_von_mises = np.abs(sigma[:,0] -sigma[:,1]).reshape(-1,1) 
    return sigma_vm

def weaken_structure(index, xPhys):
    xPhys_new = xPhys
    xPhys_new[index] = 10e-9
    return xPhys_new

def increase_load(index, f,edofMat, load=1.):
    indices = edofMat[index]
    f[indices] = load
    f_new = f
    f_new[index] = load 
    return f_new
