"""Module includes finite difference operators

Authors: Opal Issan (oissan@ucsd.edu)
Version: Nov 20th, 2023
"""
from scipy.sparse import diags


def ddx_fwd(Nx, dx, periodic=True, order=1):
    """ return the first derivative in x using a forward difference stencil.
    Assuming a uniform mesh in x, i.e. dx = const => return the operator A.
    d/dx = (f_{i+1} - f_{i})/(dx)

    Parameters
    ----------
    Nx: int
        the number of mesh points in x direction.
    dx: float
        the spacing of x, i.e. dx = x_{i+1} - x_{i}
    periodic: bool
        periodic boundary conditions (True/False). If true: Nx = Nx-1
    order: float
        order of accuracy.

    Returns
    -------
    matrix
        d/dx in matrix form
    """
    # if we have periodic boundary conditions in x then Nx is one less.
    if periodic:
        Nx = Nx - 1
    # first order upwind scheme
    if order == 1:
        A = diags([-1, 1], [0, 1], shape=(Nx, Nx)).toarray()
        if periodic:
            A[-1, 0] = 1
        else:
            A[-1, -1] = 1
            A[-1, -2] = -1
        A /= dx
        return A
    # second order upwind scheme
    elif order == 2:
        A = diags([-3 / 2, 2, -1 / 2], [0, 1, 2], shape=(Nx, Nx)).toarray()
        if periodic:
            A[-1, 0] = 2
            A[-1, 1] = -1 / 2
            A[-2, 0] = -1 / 2
        else:
            A[-1, -1] = 1 / 2
            A[-1, -2] = -2
            A[-1, -3] = 3 / 2
            A[-2, -1] = 1 / 2
            A[-2, -2] = -1 / 2
        A /= dx
        return A
    else:
        return None


def ddx_bwd(Nx, dx, periodic=True, order=1):
    """ return the first derivative in x using a forward difference stencil.
    Assuming a uniform mesh in x, i.e. dx = const => return the operator A.
    d/dx = (f_{i+1} - f_{i})/(dx)

    Parameters
    ----------
    Nx: int
        the number of mesh points in x direction.
    dx: float
        the spacing of x, i.e. dx = x_{i+1} - x_{i}
    periodic: bool
        periodic boundary conditions (True/False). If true: Nx = Nx-1
    order: float
        order of accuracy.

    Returns
    -------
    matrix
        d/dx in matrix form
    """
    # if we have periodic boundary conditions in x then Nx is one less.
    if periodic:
        Nx = Nx - 1
    # first order upwind scheme
    if order == 1:
        A = diags([-1, 1], [-1, 0], shape=(Nx, Nx)).toarray()
        if periodic:
            A[0, -1] = -1
        else:
            A[-1, -1] = 1
            A[-1, -2] = -1
        A /= dx
        return A
    else:
        return None


def ddx_central(Nx, dx, periodic=True, order=2):
    """
    return the first-order derivative of in x using a second-order accurate central difference.
    Assuming a uniform mesh in x, i.e. dx = const. => return the operator A.
    d/dx = (f_{i+1} - f_{i-1})/(2*dx)

     Parameters
    ----------
    Nx: int
        number of grid points in x direction.
    dx: float
        the spacing of x, i.e. dx = x_{i+1} - x_{i}
    periodic: bool
        periodic boundary conditions (True/False).

    Returns
    -------
    matrix
        d/dx in operator form
    """
    if periodic:
        Nx = Nx - 1
    if order == 2:
        A = diags([-1, 1], [-1, 1], shape=(Nx, Nx)).toarray()
        if periodic:
            A[0, -1] = -1
            A[-1, 0] = 1
        else:
            A[0, 0] = -3
            A[0, 1] = 4
            A[0, 2] = -1
            A[-1, -1] = 3
            A[-1, -2] = -4
            A[-1, -3] = 1
        A /= (2 * dx)
        return A
    elif order == 4:
        A = diags(diagonals=[1 / 12, -2 / 3, 2 / 3, -1 / 12], offsets=[-2, -1, 1, 2], shape=(Nx, Nx)).toarray()
        if periodic:
            A[0, -2] = 1 / 12
            A[0, -1] = -2 / 3
            A[1, -1] = 1 / 12
            A[-1, 1] = -1 / 12
            A[-1, 0] = 2 / 3
            A[-2, 0] = -1 / 12
        A /= dx
        return A
    elif order == 6:
        A = diags(diagonals=[-1 / 60, 3 / 20, -3 / 4, 3 / 4, -3 / 20, 1 / 60], offsets=[-3, -2, -1, 1, 2, 3],
                  shape=(Nx, Nx)).toarray()
        if periodic:
            A[0, -3] = -1 / 60
            A[1, -2] = -1 / 60
            A[2, -1] = -1 / 60
            A[0, -2] = 3 / 20
            A[1, -1] = 3 / 20
            A[0, -1] = -3 / 4
            A[-1, 2] = 1 / 60
            A[-2, 1] = 1 / 60
            A[-3, 0] = 1 / 60
            A[-1, 1] = -3 / 20
            A[-2, 0] = -3 / 20
            A[-1, 0] = 3 / 4
        A /= dx
        return A
    else:
        return None


def d2dx2_central(Nx, dx=0, periodic=True, order=2):
    """
    return the second derivative in x using a second-order accurate central difference.
    Assuming a uniform mesh discretization in x, i.e. dx = const. => return operator A.
    d2f/dx2 = (f_{i+1} - 2f_{i} + f_{i-1})/(dx^2)

    on the boundaries (for non-periodic):
    (forward)
    d2f/dx2 = 2f_{i} - 5 f_{i+1} + 4 f_{i+2} - f_{i+3}/(dx^3)
    (backward)
    d2f/dx2 = 2f_{i} - 5f_{i-1} + 4 f_{i-2} -f_{i-3}/(dx^3)

     Parameters
    ----------
    Nx: int
        a 2d or 1d array
    dx: float
        the spacing of x, i.e. dx = x_{i+1} - x_{i}
    periodic: bool
        periodic boundary conditions (True/False).

    Returns
    -------
    matrix
        d2/dx2 in operator form
    """
    if periodic:
        Nx = Nx - 1
    # second-order accurate central finite differencing of second-order derivative
    if order == 2:
        A = diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)).toarray()
        if periodic:
            A[0, -1] = 1
            A[-1, 0] = 1
        elif dx != 0:
            # forward
            A[0, 0] = 2 / dx
            A[0, 1] = -5 / dx
            A[0, 2] = 4 / dx
            A[0, 3] = -1 / dx
            # backward
            A[-1, -1] = 2 / dx
            A[-1, -2] = -5 / dx
            A[-1, -3] = 4 / dx
            A[-1, -4] = -1 / dx
        A /= (dx ** 2)
        return A
    # fourth-order accurate central finite differencing of second-order derivative
    elif order == 4:
        A = diags([-1, 16, -30, 16, -1], [-2, -1, 0, 1, 2], shape=(Nx, Nx)).toarray()
        if periodic:
            A[0, -2] = -1
            A[0, -1] = 16
            A[1, -1] = -1
            A[-1, 1] = -1
            A[-1, 0] = 16
            A[-2, 0] = -1
        if dx != 0:
            A /= 12 * (dx ** 2)
        return A
    else:
        return None
