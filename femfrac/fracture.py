import numpy as np
from topoptlab.elements.stress_measures import von_mises_stress
from topoptlab.elements.bilinear_quadrilateral import bmatrix_rectangle
from femfrac.disorder import get_yield, weibull_redistribute
from scipy.ndimage import gaussian_filter
from topoptlab.utils import map_eltoimg, map_imgtoel

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

def insert_notch(length, xPhys, nely):
    middle = int(len(xPhys)//2)
    xPhys[middle:int(middle+length)] = 0
    xPhys[middle-nely:int(middle+length-nely)] = 0
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

def get_flow_direction(eps, C , sigma_vm_loc, index): 
   sxx, syy, sxy =  get_stress_tensor(eps[index], index=index)
   delta_eps_xx =(2*sxx - syy)/(2*sigma_vm_loc)
   delta_eps_yy = (2*syy - sxx)/(2*sigma_vm_loc)
   delta_eps_xy = 3 * sxy / sigma_vm_loc  
   direction = np.array([delta_eps_xx, delta_eps_xy, delta_eps_yy, delta_eps_xy]) 
   return direction

def increment_eps_pl(eps, eps_pl, index, E_scaled, sigma_vm, shear_mod = 0.5, d_e_0 = 0.5):
    C = E_scaled.reshape(-1,1)
    flow_dir = get_flow_direction(eps, C, sigma_vm[index], index).squeeze(-1) # [dux/dx, duy/dx, duy/dy, duy/dx]
    # TODO: gaussian to fluctiate flow dir like: new_flow_dir = M(phi).T @ flow_dir M(phic)
    delta_eps_pl = d_e_0 * 2 * sigma_vm[index]/shear_mod
    increment = flow_dir * delta_eps_pl
    print(eps_pl[index])
    print(increment)
    print(eps_pl[index] + increment)
    eps_pl[index] += increment
    return eps_pl, increment 
    

def damage_element(index, f, u, eps, eps_pl, sigma_vm, edofMat, l, xPhys, penal, E_scaled, Emin, Emax, nu, t=1., relax=0.1):
    """
    Damage one element by adding an eigenstrain-equivalent nodal force to f.
    K is untouched. relax=1.0 fully relaxes the element's current elastic
    strain (i.e. eps0 = -eps_current); relax<1.0 partially relaxes it.
    """
    eps_e = eps - eps_pl
    eps_pl, delta_eps_pl = increment_eps_pl(eps_e, eps_pl, index, xPhys, sigma_vm, shear_mod = 0.5, d_e_0 = relax ) 
    B_e = bmatrix_rectangle(xi=np.array([0.]), eta=np.array([0.]),
                            a=l, b=l)[0]      # (3,8)
    eps_rearranged = np.array([delta_eps_pl[0], delta_eps_pl[2], delta_eps_pl[1]]) # arrange to dux/dx, duy/dy, dux/dy
    df_eig  = eps_rearranged @ B_e 
    """
     eps_pl stores [dux/dx, dux/dy, duy/dy, duy/dux]
    """
    """
    E_local = Emin + (1 - damage[index])**penal *(Emax- Emin) # TODO what is this ?!
    D = get_D_planestress(E_local, nu)
    eps_current = get_element_strain(u, edofMat, index, B_e)
    eps0 = -relax * eps_current                       # eigenstrain to apply
    Ae = l * l
    f_eig = t * Ae * (B_e.T @ D @ eps0)                # (8,)
    """
    f_new = f.copy()
    print(edofMat[index])
    for e,i in enumerate(edofMat[index]):
        f_new[i] = df_eig[e]
    f_new.squeeze(-1)
    return f_new, xPhys

def burn_max_element(f, u, eps, eps_pl, sigma_vm, yield_scaled, edofMat,
                     l, xPhys, damage, penal, E_scaled, Emin, Emax, nu,
                     t=1., relax=0.5):
    # interpolate over element
    #sigma_vm = von_mises_stress(T, xPhys, edofMat, nelz)
    ratios = np.array(sigma_vm).reshape(-1) / np.array(yield_scaled).reshape(-1)
    max_element_index  = np.argmax(ratios)
    damage[max_element_index] += relax
    new_sigma_0 = np.maximum(0.0, 1.0 + damage[max_element_index])
    yield_scaled[max_element_index] = weibull_redistribute(new_sigma_0)
    f_new, _ = damage_element(max_element_index, f, u, eps, eps_pl, sigma_vm,
                              edofMat, l, xPhys, penal, E_scaled, Emin, Emax,
                              nu, t=1., relax=relax)
    # f_disp = 0 #TODO: dispers (gaussian?) f 
    return f_new, xPhys, damage, eps, eps_pl, yield_scaled, E_scaled

def get_element_strain(u, edofMat, nelz, l=1.0):
    """
    Strain of every element of a 2d displacement field.

    The strains are returned in the order [du_x/dx, du_x/dy, du_y/dx, du_y/dy],
    i.e. the derivatives with respect to the physical coordinates. The size of
    the elements is needed to transform the derivatives with respect to the
    reference coordinates into physical ones.

    Parameters
    ----------
    u : np.ndarray
        displacement field of shape (ndof, 1).
    edofMat : np.ndarray
        element degree of freedom matrix.
    nelz : int or None
        number of elements in z direction. Only important if ndim is 3.
    l : float or np.ndarray
        side length(s) of the elements.

    Returns
    -------
    np.ndarray
        array of shape (nelem, 4) with the strain components
        [du_x/dx, du_x/dy, du_y/dx, du_y/dy].

    """
    # the size of the elements
    l = np.atleast_1d(np.asarray(l, dtype=float)).ravel()
    if nelz == None:  
        from topoptlab.elements.bilinear_quadrilateral  import shape_functions_dxi, invjacobian, invjacobian_rectangle
        if l.size == 1:
            l = np.repeat(l, 2)
        delta_N_e = shape_functions_dxi(0,0)
        J_e_inv = invjacobian_rectangle(l[0], l[1]) 
        B_e = delta_N_e @ J_e_inv
    else:
        from topoptlab.elements.trilinear_hexahedron import shape_functions_dxi, invjacobian, invjacobian_cuboid
        if l.size == 1:
            l = np.repeat(l, 3)
        delta_N_e = shape_functions_dxi(0,0,0)
        J_e_inv = invjacobian_cuboid(l[0], l[1], l[2]) 
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

def get_stress_tensor(eps, lam=90, my=30, index=None, mode_palain_strain = True):
    # lam = 91.9 & my = 32.3 from https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.185502 
    c_11 = lam + 2*my
    c_12 = lam
    c_66 = my
    if index==None:
        eps_1 = eps[:,0]
        eps_2 = eps[:,2]
        gamma_12 = eps[:,1]
        sxx = c_11 * eps_1[...,None] + c_12 * eps_2[...,None] # [dux/dx, dux/dy]  
        syy = c_12 * eps_1[...,None] + c_11 * eps_2[...,None] # [duy/dx, duy/dy]  
        sxy = c_66 * gamma_12[...,None]/2
        return np.array([sxx, syy, sxy])
    else:
        eps_1 = eps[0]
        eps_2 = eps[2]
        gamma_12 = eps[1]
        sxx = c_11 * eps_1[...,None] + c_12 * eps_2[...,None] # [dux/dx, dux/dy]  
        syy = c_12 * eps_1[...,None] + c_11 * eps_2[...,None] # [duy/dx, duy/dy]  
        sxy = c_66 * gamma_12[...,None]/2
        return np.array([sxx, syy, sxy])

def gaussian_filter_eps(eps, nelx, nely, sigma=1.0, mode="nearest"):
    """Apply an isotropic Gaussian filter to an element-based strain field.

    Parameters
    ----------
    eps : np.ndarray, shape (nelx*nely,) or (nelx*nely, ncomp)
        element field (e.g. strain), one value (or vector of values)
        per element, ordered like the mesh (i = x*nely + y).
    nelx, nely : int
        number of elements in x/y direction (equal for the square grid).
    sigma : float
        Gaussian standard deviation in element units.
    mode : str
        boundary handling passed to scipy.ndimage.gaussian_filter
        (e.g. "nearest", "reflect", "wrap").

    Returns
    -------
    eps_filtered : np.ndarray
        filtered field, same shape as `eps`.
    """
    eps = np.asarray(eps, dtype=float)
    img = map_eltoimg(eps, nelx=nelx, nely=nely)
    img = gaussian_filter(img, sigma=sigma, mode=mode, axes=(0, 1))
    return map_imgtoel(img, nelx=nelx, nely=nely)

def get_vm_stress(u, xPhys, edofMat,nelx, nely, nelz : None, l=1.0):
    C = xPhys.reshape(-1,1)
    eps_unfiltered = get_element_strain(u, edofMat, nelz, l)
    eps = gaussian_filter_eps(eps_unfiltered, nelx, nely, sigma=1.0, mode="nearest")
    sxx, syy, sxy = get_stress_tensor(eps)
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
#        #sigma_vm_old = np.sum(np.sqrt(sigma ** 2), axis=1, keepdims=True)
#        sigma_vm = np.abs(sigma[:,0] -sigma[:,1]).reshape(-1,1) 
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    nx = 10
    ny = nx
    eps = np.zeros((nx*ny,4)) 
    eps[55] = [1., 0, 0.5, 0]
    e_img = map_eltoimg(eps, nx, ny)  
    gauss_eps = gaussian_filter_eps(eps, nx, ny, sigma=1) 
    gauss_img = map_eltoimg(gauss_eps, nx, ny)
    fig, ax = plt.subplots(2,2)
    ax[0,0].imshow(e_img[:,:,0], vmin=0, vmax=1.0)
    ax[0,1].imshow(e_img[:,:,2], vmin=0, vmax=1.0)
    ax[1,0].imshow(gauss_img[:,:,0], vmin=0, vmax=1.0)
    ax[1,1].imshow(gauss_img[:,:,2], vmin=0, vmax=1.0)
    plt.show()

    