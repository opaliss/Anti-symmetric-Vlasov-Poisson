"""Nonlinear Landau Damping Module SW formulation solved via Vlasov-Poisson

Author: Opal Issan (oissan@ucsd.edu)
Date: November 24th, 2023
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import numpy as np
from operators.SW_sqrt import RHS, solve_poisson_equation, momentum_drift, energy_drift
from FD_tools.implicit_midpoint import implicit_midpoint_solver


def rhs(y, t):
    dydt_ = np.zeros(len(y))

    # initialize species states
    state_e = np.zeros((Nv, Nx - 1))
    state_i = np.zeros((Nv, Nx - 1))
    # static/background ions
    state_i[0, :] = np.sqrt(np.ones(Nx - 1) / alpha_i)

    # re-arrange the states into matrix form [Nv, Nx]
    for jj in range(Nv):
        state_e[jj, :] = y[jj * (Nx - 1): (jj + 1) * (Nx - 1)]

    # Poisson equation solver
    E = solve_poisson_equation(state_e=state_e,
                               state_i=state_i,
                               alpha_e=alpha_e,
                               alpha_i=alpha_i,
                               dx=dx,
                               solver="gmres",
                               order_fd=2, L=L)

    for jj in range(Nv):
        dydt_[jj * (Nx - 1): (jj + 1) * (Nx - 1)] = RHS(state=state_e,
                                                        m=jj,
                                                        Nv=Nv,
                                                        alpha_s=alpha_e,
                                                        q_s=q_e,
                                                        dx=dx,
                                                        Nx=Nx,
                                                        m_s=m_e,
                                                        E=E,
                                                        u_s=u_e)

    # momentum drift
    dydt_[-2] = momentum_drift(state_e=state_e,
                               state_i=state_i,
                               E=E, Nv=Nv, alpha_e=alpha_e, alpha_i=alpha_i, q_e=q_e, q_i=q_i, dx=dx)

    # energy drift
    dydt_[-1] = energy_drift(state_e=state_e,
                             state_i=state_i,
                             E=E, Nv=Nv, alpha_e=alpha_e,
                             alpha_i=alpha_i, q_e=q_e, q_i=q_i,
                             dx=dx, m_e=m_e, m_i=m_i, Nx=Nx, u_e=u_e, u_i=u_i)
    return dydt_


if __name__ == '__main__':
    # set up configuration parameters
    # number of mesh points in x
    Nx = 101
    # number of spectral expansions
    Nv = 101
    # epsilon displacement in initial electron distribution
    epsilon = 0.5
    # velocity scaling of electron and ion
    alpha_e = np.sqrt(2)
    alpha_i = np.sqrt(2 / 1863)
    # x grid is from 0 to L
    L = 4 * np.pi
    # spacial spacing dx = x[i+1] - x[i]
    dx = L / (Nx - 1)
    # time stepping
    dt = 1e-2
    # final time (non-dimensional)
    T = 45.
    # vector with timestamps
    t_vec = np.linspace(42.1, T, int((T-42.1) / dt) + 1)
    # velocity scaling
    u_e = 0
    u_i = 0
    # mass normalized
    m_e = 1
    m_i = 1836
    # charge normalized
    q_e = -1
    q_i = 1

    # spatial grid
    x = np.linspace(0, L, Nx)

    # initial condition of the first expansion coefficient
    C_0e = np.sqrt((1 + epsilon * np.cos(0.5 * x)) / alpha_e)

    # initialize states (electrons and ions)
    states_e = np.zeros((Nv, Nx - 1))

    # initialize the expansion coefficients
    states_e[0, :] = C_0e[:-1]

    # # initial condition of the semi-discretized ODE
    # y0 = states_e.flatten("C")
    # y0 = np.append(y0, np.zeros(2))
    # 43.72
    # 46.22
    y0 = np.load("../data/SW_sqrt/nonlinear_landau/sol_midpoint_u_" + str(Nv) + "_continued.npy")[:, -1]

    # set up implicit midpoint with newton-krylov solver
    sol_midpoint_u = implicit_midpoint_solver(t_vec=t_vec, y0=y0, rhs=rhs, nonlinear_solver_type="newton_krylov",
                                              r_tol=1e-8, a_tol=1e-8, max_iter=50)

    np.save("../data/SW_sqrt/nonlinear_landau/sol_midpoint_u_" + str(Nv) + "_continued2", sol_midpoint_u)
    np.save("../data/SW_sqrt/nonlinear_landau/sol_midpoint_t_" + str(Nv) + "_continued2", t_vec)
