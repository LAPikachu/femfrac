from topoptlab.elements.trilinear_hexahedron import shape_functions_dxi, invjacobian, invjacobian_cuboid

def von_mises_stress(u, xPhys, edofMat, nelz : None):
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
    delta_T  = (T[edofMat,0][...,None]*B_e).sum(axis=1)
    C = xPhys.reshape(-1,1)
    sigma = C * delta_T 
    sigma_von_mises = np.sum(np.sqrt(sigma ** 2), axis=1, keepdims=True)
    return sigma_von_mises 