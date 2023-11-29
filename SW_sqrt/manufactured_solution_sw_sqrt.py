"""Manufactured solution Module SW formulation solved via Vlasov-Poisson

Author: Opal Issan (oissan@ucsd.edu)
Date: November 29th, 2023
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

import numpy as np
from operators.SW_sqrt import linear_1
from FD_tools.finite_difference_operators import ddx_central
from FD_tools.implicit_midpoint import implicit_midpoint_solver


def linear_rhs(state, m, Nv, dx, Nx, u_s, alpha_s, fd_order):
    """ rhs of the linear advection equation

    :param state: ndarray, matrix with spectral coefficient results
    :param m: int, the current spectral index
    :param Nv: int, the number of velocity spectral terms
    :param dx: float, the spatial grid spacing
    :param Nx: int, the number of spatial grid points
    :param u_s: float, the velocity shifting parameter
    :param alpha_s: float, the velocity scaling parameter
    :param fd_order: int, the finite difference order of accuracy
    :return: rhs of linear advection equation
    """
    D = ddx_central(Nx=Nx, dx=dx, periodic=True, order=fd_order)
    term1, term2, term3 = linear_1(state=state, m=m, alpha_s=alpha_s, Nv=Nv, u_s=u_s)
    return -D @ (term1 + term2 + term3)


def rhs(y, t):
    """ rhs of the ode (vector form)

    :param y: current state
    :param t: timestamp
    :return: rhs of the ode (vector form)
    """
    dydt_ = np.zeros(len(y))

    state_e = np.zeros((Nv, Nx - 1))

    for jj in range(Nv):
        state_e[jj, :] = y[jj * (Nx - 1): (jj + 1) * (Nx - 1)]

    for jj in range(Nv):
        dydt_[jj * (Nx - 1): (jj + 1) * (Nx - 1)] = linear_rhs(state=state_e,
                                                               m=jj,
                                                               Nv=Nv,
                                                               alpha_s=alpha_m,
                                                               dx=dx,
                                                               Nx=Nx,
                                                               u_s=u_m,
                                                               fd_order=fd_order)
    return dydt_


if __name__ == '__main__':
    # set up configuration parameters
    # spatial length
    L = 2 * np.pi
    # number of grid points in space
    Nx = 401
    # grid spacing
    dx = L / (Nx - 1)
    # final time step
    T = 1
    # number of time steps
    dt = 1e-3
    # timestamp vector
    t_vec = np.linspace(0, T, int(T / dt) + 1)
    # velocity shifting parameters
    u_m = 1
    alpha_m = 2
    # number of spectral coefficients
    Nv = 100
    # finite difference ordering
    fd_order = 2
    # spatial vector
    x = np.linspace(0, L, Nx)[:-1]
    # initialize states
    states_e = np.zeros((Nv, Nx - 1))
    # initialize the expansion coefficients
    states_e[0, :] = (2 - np.cos(x)) * (np.pi ** (1 / 8))
    # initial condition of the semi-discretized ODE
    y0 = states_e.flatten("C")
    # set up implicit midpoint
    sol_midpoint = implicit_midpoint_solver(t_vec=t_vec, y0=y0, rhs=rhs, nonlinear_solver_type="newton_krylov",
                                            r_tol=1e-8, a_tol=1e-14, max_iter=100)
    # save results
    np.save("../data/SW_sqrt/manufactured/manufactured_solution_u_" + str(Nx) + "_FD_" + str(fd_order) + ".npy", sol_midpoint)
    np.save("../data/SW_sqrt/manufactured/manufactured_solution_t_" + str(Nx) + "_FD_" + str(fd_order) + ".npy", t_vec)
