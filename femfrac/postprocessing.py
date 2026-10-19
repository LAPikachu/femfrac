from typing import Union, Dict
import numpy as np
from meshio import Mesh

def export_vtk(filename: str, 
               nelx: int, nely: int, 
               xPhys: Union[None,np.ndarray] = None,
               nelz: Union[None,int] = None,
               x: Union[None,np.ndarray] = None,
               u: Union[None,np.ndarray] = None, 
               f: Union[None,np.ndarray] = None,
               nodal_variables: Union[None,Dict[np.ndarray]] = None,
               element_variables: Union[None,Dict[np.ndarray]] = None,
               u_bw: Union[None,np.ndarray] = None,
               f_bw: Union[None,np.ndarray] = None,
               xTilde: Union[None,np.ndarray] = None,
               elem_size: Union[None,float, np.ndarray] = None,
               volfrac: Union[None,float] = None,
               stress_vm: Union[None,float, np.ndarray] = None,
               eps: Union[None,np.ndarray] = None,
               vectors: Union[None,np.ndarray] = None) -> None:
    """
    Export design to a vtk file for visualisation e. g. with Paraview.

    Parameters
    ----------
    filename : str
        filename without ".vtk" ending.
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    xPhys : np.ndarray
        densities used to scale the material properties.
    nelz : int
        number of elements in z direction.
    x : np.ndarray, optional
        interemdiary densities as (usually) returned by the optimizer.
    u : np.ndarray, optional
        field variable (displacements, temperature) which is usually the left 
        hand side of the generic matrix problem Ku=f. The default is None.
    f : np.ndarray, optional
        flow variable (forces, heat sources/sinks) which is usually the right 
        hand side of the generic matrix problem Ku=f
    u_bw : np.ndarray, optional
        field variable of the final black/white design. The default is None.
    f_bw : np.ndarray, optional
        flow variable of the final black/white design. The default is None.
    xTilde : np.ndarray, optional
        interemdiary densities by the density filter. So far they only occur 
        if Haeviside projections are used. The default is None.
    elem_size : np.ndarray, float
        element size.
    volfrac : float, optional
        volume fraction. If not None, then also a thresholded designed is 
        stored. The default is None.
    stress_vm : np.ndarray, optional
        von Mises stress.
    eps : np.ndarray, optional
        element strain field of shape (nelx*nely, 4) (2D), stored in the order
        [eps_xx, eps_xy, eps_yy, eps_yx] where eps_xy == eps_yx by symmetry.
        Written to the vtk as the scalar cell arrays "eps_xx", "eps_yy", "eps_xy"
        (shear taken as the mean of the two duplicate entries) plus a symmetric
        second-order tensor "eps_tensor" in VTK component order
        [xx, xy, xz, yx, yy, yz, zx, zy, zz] for tensor glyphs in Paraview.
        A 1-D array (nelx*nely,) is written as a single scalar "eps".
    vectors : np.ndarray
        vectors [cos(theta)cos(phi); sin(theta)cos(phi); -sin(phi)] used to show the 
        orientation arrow field in Paraview.

    Returns
    -------
    None.

    """
    
    # construct node positions for meshio 
    if nelz is None:
        _x,_y = np.meshgrid(np.linspace(0,nelx,nelx+1),
                            np.linspace(nely,0,nely+1))
        points = np.column_stack((_x.flatten("F"),
                                  _y.flatten("F"))) 
    else:
        #
        xcoords = np.linspace(0,nelx,nelx+1)
        ycoords = np.linspace(nely,0,nely+1)
        zcoords = np.linspace(0,nelz,nelz+1)
        #
        _x = np.tile(np.repeat(xcoords,nely+1),nelz+1).flatten()
        _y = np.tile(ycoords,(nelx+1)*(nelz+1)).flatten()
        _z = np.repeat(zcoords,(nelx+1)*(nely+1))
        #
        points = np.column_stack((_x,_y,_z)) 

    # Apply the physical spacing by element size
    if elem_size is not None:
        ndim = points.shape[1]
        if isinstance(elem_size, float):
            elem_size = np.full(ndim, elem_size, dtype=float)
        else:
            elem_size = np.asarray(elem_size, dtype=float)
        points = points * elem_size[None, :]
    # insert data for nodes
    node_data = {}
    if not u is None:
        # takes care of multiple load cases
        for i in np.arange(u.shape[1]):
            node_data.update({f"u{i}": u[:,i].reshape(points.shape[0],
                                         int(u[:,i].shape[0]/points.shape[0]))})
    if not u_bw is None:
        # takes care of multiple load cases
        for i in np.arange(u_bw.shape[1]):
            node_data.update({f"u_bw{i}": u_bw[:,i].reshape(points.shape[0],
                                         int(u_bw[:,i].shape[0]/points.shape[0]))})
    if not f is None:
        # takes care of multiple load cases
        for i in np.arange(u.shape[1]):
            node_data.update({f"f{i}": f[:,i].reshape(points.shape[0],
                                             int(f[:,i].shape[0]/points.shape[0]))})
    if not f_bw is None:
        # takes care of multiple load cases
        for i in np.arange(f_bw.shape[1]):
            node_data.update({f"f_bw{i}": f_bw[:,i].reshape(points.shape[0],
                                             int(f_bw[:,i].shape[0]/points.shape[0]))})
    if nodal_variables is not None:
        #
        for key in nodal_variables.keys():
            if nodal_variables[key] is None:
                continue
            for i in range(nodal_variables[key].shape[1]):
                node_data.update({f"{key}{i}": nodal_variables[key][:,i].reshape(points.shape[0],
                                             int(nodal_variables[key][:,i].shape[0]/points.shape[0]))})

    # assign node IDs to each cell. 
    if nelz is None:
        elx,ely = np.arange(nelx)[:,None], np.arange(nely)[None,:]
        n1 = ((nely+1)*elx+ely).flatten()
        n2 = ((nely+1)*(elx+1)+ely).flatten()
        idMat = np.column_stack((n1+1, n2+1, n2, n1))
    else:
        elx = np.arange(nelx)[None,:,None]
        ely = np.arange(nely)[None,None,:]
        elz = np.arange(nelz)[:,None,None]
        n1 = ((nelx+1)*(nely+1)*elz + (nely+1)*elx + ely).flatten()
        n2 = ((nelx+1)*(nely+1)*elz + (nely+1)*(elx+1) + ely).flatten()
        n3 = ((nelx+1)*(nely+1)*(elz+1) + (nely+1)*elx + ely).flatten()
        n4 = ((nelx+1)*(nely+1)*(elz+1) + (nely+1)*(elx+1) + ely).flatten()
        idMat = np.column_stack((n1+1,n2+1,n2,n1,
                                 n3+1,n4+1,n4,n3))
    # insert data for elements
    el_data = {}
    if xPhys is not None: 
        el_data.update({"xPhys": [xPhys]})
    if x is not None:
        el_data.update({"x": [x]})
    if xTilde is not None:
        el_data.update({"xTilde": [xTilde]})
    if volfrac is not None and xPhys is not None:
        el_data.update({"xThresh": [threshold(xPhys,volfrac)]})
    if volfrac is not None and \
       element_variables is not None and \
       "xPhys" in element_variables.keys():
        el_data.update({"xThresh": [threshold(element_variables["xPhys"],volfrac)]})
    if stress_vm is not None:
        el_data["stress_vm"] = [np.asarray(stress_vm, dtype=float)]
    if eps is not None:
        eps = np.asarray(eps, dtype=float)
        if eps.ndim == 1:
            el_data["eps"] = [eps]
        elif eps.ndim == 2 and eps.shape[1] == 4:
            # stored as [eps_xx, eps_xy, eps_yy, eps_yx], eps_xy == eps_yx
            eps_xx = eps[:, 0]
            eps_xy = 0.5 * (eps[:, 1] + eps[:, 3])
            eps_yy = eps[:, 2]
            # symmetric 2nd-order tensor in VTK order
            # [xx, xy, xz, yx, yy, yz, zx, zy, zz]
            eps_tensor = np.zeros((eps.shape[0], 9))
            eps_tensor[:, 0] = eps_xx
            eps_tensor[:, 1] = eps_xy
            eps_tensor[:, 3] = eps_xy
            eps_tensor[:, 4] = eps_yy
            el_data["eps_tensor"] = [eps_tensor]
        else:
            raise ValueError(
                f"eps must have shape (n_elem,) or (n_elem, 4), got {eps.shape}")
    if vectors is not None:
        el_data["vectors"] = [np.asarray(vectors, dtype=float)]
    if element_variables is not None:
        #
        for key in element_variables.keys():
            if element_variables[key] is None:
                continue  

            if len(element_variables[key].shape)==2:
                for i in range(element_variables[key].shape[1]): 
                    el_data.update({f"{key}-{i}": [element_variables[key][:,i]]})
            elif len(element_variables[key].shape)==1:
                #
                if element_variables[key].dtype==np.int32:
                    element_variables[key] = element_variables[key].astype(np.int64)

                el_data.update({f"{key}": [element_variables[key]]})
            else:
                raise ValueError
    #
    if nelz is None:
        Mesh(points,
             [("quad", idMat)],
             point_data=node_data,
             cell_data=el_data).write(filename+".vtk")
    else:
        Mesh(points,
             [("hexahedron", idMat)],
             point_data=node_data,
             cell_data=el_data).write(filename+".vtk")
    return

if __name__ == '__main__':
    pass