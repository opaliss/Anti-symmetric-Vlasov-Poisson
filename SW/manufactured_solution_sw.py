"""Manufactured Solution Module SW formulation solved via Vlasov-Poisson

Author: Opal Issan (oissan@ucsd.edu)
Date: November 29th, 2023
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))
import numpy as np
from FD_tools.implicit_midpoint import implicit_midpoint_solver
from SW_sqrt.manufactured_solution_sw_sqrt import linear_rhs


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
    # velocity scaling parameter
    alpha_m = np.sqrt(2)
    # number of spectral coefficients
    Nv = 100
    # finite difference order
    fd_order = 2
    # spatial grid
    x = np.linspace(0, L, Nx)[:-1]
    # initialize states
    states_e = np.zeros((Nv, Nx - 1))
    # initialize the expansion coefficients
    states_e[0, :] = (2 - np.cos(x))**2
    # initial condition of the semi-discretized ODE
    y0 = states_e.flatten("C")
    # set up implicit midpoint
    sol_midpoint = implicit_midpoint_solver(t_vec=t_vec, y0=y0, rhs=rhs, nonlinear_solver_type="newton_krylov",
                                            r_tol=1e-8, a_tol=1e-14, max_iter=100)
    # save results
    np.save("../data/SW/manufactured/manufactured_solution_u_" + str(Nx) + "_FD_" + str(fd_order) + ".npy", sol_midpoint)
    np.save("../data/SW/manufactured/manufactured_solution_t_" + str(Nx) + "_FD_" + str(fd_order) + ".npy", t_vec)
