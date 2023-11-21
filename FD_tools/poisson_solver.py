"""module to solve Poisson Equation.

Author: Opal Issan (oissan@ucsd.edu)
Last Update: Nov 20th, 2023
"""
from FD_tools.finite_difference_operators import d2dx2_central, ddx_central
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import gmres
import numpy as np


def gmres_solver(rhs, dx, periodic=True, order="ddx", order_fd=2):
    """Poisson solver using an iterative solver: GMRES

    :param order_fd: order of finite difference approximation
    :param order: str, either first-order derivative "ddx" or second-order derivative "d2dx2"
    :param rhs: array, rhs of the equation (poisson)
    :param periodic: bool: True/False
    :param dx: float, spatial spacing
    :return: E that satisfies d/dx E = rho or d^2/dx^2 phi = rho
    """
    # get finite difference derivative operator
    if order == "ddx":
        # first order derivative matrix
        A = ddx_central(Nx=len(rhs) + 1, dx=dx, periodic=periodic, order=order_fd)
    elif order == "d2dx2":
        # second order derivative matrix
        A = d2dx2_central(Nx=len(rhs) + 1, dx=dx, periodic=periodic, order=order_fd)

    x, _ = gmres(csc_matrix(A), rhs - np.mean(rhs), atol=1e-15, tol=1e-8)
    #print("gmres error ==", np.max(np.abs(A@x - rhs + np.mean(rhs))))
    return x - np.mean(x)


def fft_solver_Ax_b(rhs, dx):
    """Poisson solver using fft of the equations
        A x= b
        fft(A[:, 0]) * fft(x) = fft(b)
        fft(x) = fft(b) / fft(A[:, 0])
        x = ifft(fft(b) / fft(A[:, 0]))

    :param rhs: array, rhs of the equation (poisson)
    :param dx: float, spatial spacing
    :return: E that satisfies d/dx E = rho
    """
    D = ddx_central(Nx=len(rhs) + 1, dx=dx)
    A = np.zeros(len(rhs))
    A[1] = -1 / (2 * dx)
    A[-1] = 1 / (2 * dx)

    rhs_fft = np.fft.fft(rhs - np.mean(rhs))
    A_fft = np.fft.fft(A)
    sol = np.divide(rhs_fft, A_fft, where=A_fft != 0)
    x = np.fft.ifft(sol).real
    L_inf = np.max(np.abs(D @ x - rhs + np.mean(rhs)))
    print("fft error = ", L_inf)
    if L_inf < 1e-10:
        return x - np.mean(x)
    else:
        return gmres_solver(rhs=rhs, dx=dx, periodic=True, order="ddx")


def fft_solver(rhs, L):
    """Poisson solver using fft

        Ax = b
        fft(A[:, 0]) * fft(x) = fft(b)
        fft(x) = fft(b) / fft(A[:, 0])
        x = ifft(fft(b) / fft(A[:, 0]))

    :param rhs: array, rhs of the equation (poisson)
    :param L: float, length of spatial domain
    :return: x that satisfies Ax = rhs
    """
    x = np.fft.fft(rhs - np.mean(rhs))
    x = np.append(x, x[0])
    N = len(x)

    for k in range(-N//2, N//2):
        if not k == 0:
            x[k] /= 1j*2*np.pi*k/L
        else:
            x[k] = 0

    x = np.fft.ifft(x).real
    E = x - np.mean(x)
    return E[:-1]