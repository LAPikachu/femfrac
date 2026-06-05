import numpy as np
from topoptlab.elements.stress_measures import von_mises_stress

def burn_max_element(sigma_von_mises, xPhys, thresholds):
    # interpolate over element
    #sigma_von_mises = von_mises_stress(T, xPhys, edofMat, nelz)
    ratios = sigma_von_mises/thresholds
    max_element  = np.argmax(ratios)
    xPhys = weaken_structure(max_element, xPhys)
    return xPhys 

def get_vm_stress(u, xPhys, edofMat, nelz : None):
    C = xPhys.reshape(-1,1)
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
        delta_ux  = (ux[(edofMatx/2).astype(int),0][...,None]*B_e).sum(axis=1)
        delta_uy = (uy[((edofMaty-1)/2).astype(int),0][...,None]*B_e).sum(axis=1)
        sxx = C * delta_ux[:,0][...,None] # [dux/dx, dux/dy]  
        syy = C * delta_uy[:,1][...,None] # [duy/dx, duy/dy]  
        sxy = C * ((delta_ux[:,1][...,None] + delta_uy[:,0][...,None])/2)
        sigma_vm = von_mises_stress(np.squeeze([sxx, syy, sxy], axis=2).T, ndim=2)
    elif edofMat.shape[1] == 12: #check if 3d-vector-field
        raise NotImplementedError("Function not implemented for 3d/ 3d-vector-field.")
        '''
        ux  = u[::3]
        uy  = u[1::3]
        uy  = u[2::3]
        edofMatx = edofMat[:,::3]
        edofMaty = edofMat[:,1::3]
        edofMatz = edofMat[:,2::3]
        delta_ux  = (ux[(edofMatx/3).astype(int),0][...,None]*B_e).sum(axis=1)
        delta_uy = (uy[((edofMaty-1)/3).astype(int),0][...,None]*B_e).sum(axis=1)
        delta_uz = (uy[((edofMaty-2)/3).astype(int),0][...,None]*B_e).sum(axis=1)
        du_dx = delta_ux[:,0]
        sxx = C * delta_ux[:,0] # [dux/dx, dux/dy]  
        syy = C * delta_uy[:,1] # [duy/dx, duy/dy]  
        szz = C * delta_uz[:,1] # [duy/dx, duy/dy]  
        sxy = C * ((delta_ux[:,1] + delta_uy[:,0])/2)
        sxz = C * ((delta_ux[:,1] + delta_uz[:,0])/2)
        syz = C * ((delta_uy[:,1] + delta_uz[:,0])/2)
        sigma_vm = von_mises_stress(np.array([sxx, syy, sxy]), ndim=2)
        '''
    else:
        delta_u  = (u[edofMat,0][...,None]*B_e)
        delta_u  = (u[edofMat,0][...,None]*B_e).sum(axis=1)
        sigma = C * delta_u 
        #sigma_von_mises_old = np.sum(np.sqrt(sigma ** 2), axis=1, keepdims=True)
        sigma_von_mises = np.abs(sigma[:,0] -sigma[:,1]).reshape(-1,1) 
    return sigma_vm

def weaken_structure(index, xPhys):
    xPhys_new = xPhys
    xPhys_new[index] = 0
    return xPhys_new
