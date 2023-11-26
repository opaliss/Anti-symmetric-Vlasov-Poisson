"""Two-Stream Instability Module SW formulation solved via Vlasov-Poisson

Author: Opal Issan (oissan@ucsd.edu)
Date: November 24th, 2023
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

import numpy as np
from operators.SW_sqrt import RHS, solve_poisson_equation_two_stream, momentum_drift_two_stream, energy_drift_two_stream
from FD_tools.implicit_midpoint import implicit_midpoint_solver


def rhs(y, t):
    dydt_ = np.zeros(len(y))

    # initialize states
    state_e1 = np.zeros((Nv, Nx - 1))
    state_e2 = np.zeros((Nv, Nx - 1))
    state_i = np.zeros((Nv, Nx - 1))
    # static/background ions
    state_i[0, :] = np.sqrt(np.ones(Nx - 1) / alpha_i)

    # re-arrange states to matrix form [Nv, Nx]
    for jj in range(Nv):
        state_e1[jj, :] = y[jj * (Nx - 1): (jj + 1) * (Nx - 1)]
        state_e2[jj, :] = y[Nv * (Nx - 1) + jj * (Nx - 1): Nv * (Nx - 1) + (jj + 1) * (Nx - 1)]

    # solve Poisson's equation
    E = solve_poisson_equation_two_stream(state_e1=state_e1,
                                          state_e2=state_e2,
                                          state_i=state_i,
                                          alpha_e1=alpha_e1,
                                          alpha_e2=alpha_e2,
                                          alpha_i=alpha_i,
                                          dx=dx,
                                          L=L)

    for jj in range(Nv):
        # electron species 1
        dydt_[jj * (Nx - 1): (jj + 1) * (Nx - 1)] = RHS(state=state_e1,
                                                        m=jj,
                                                        Nv=Nv,
                                                        alpha_s=alpha_e1,
                                                        q_s=q_e1,
                                                        dx=dx,
                                                        Nx=Nx,
                                                        m_s=m_e1,
                                                        E=E,
                                                        u_s=u_e1)
        # electron species 2
        dydt_[Nv * (Nx - 1) + jj * (Nx - 1): Nv * (Nx - 1) + (jj + 1) * (Nx - 1)] = RHS(state=state_e2,
                                                                                        m=jj,
                                                                                        Nv=Nv,
                                                                                        alpha_s=alpha_e2,
                                                                                        q_s=q_e2,
                                                                                        dx=dx,
                                                                                        Nx=Nx,
                                                                                        m_s=m_e2,
                                                                                        E=E,
                                                                                        u_s=u_e2)

        # # momentum drift
        # dydt_[-2] = momentum_drift_two_stream(state_e1=state_e1,
        #                                       state_e2=state_e2,
        #                                       state_i=state_i,
        #                                       E=E, Nv=Nv, alpha_e1=alpha_e1,
        #                                       alpha_e2=alpha_e2, alpha_i=alpha_i,
        #                                       q_e1=q_e1, q_e2=q_e2, q_i=q_i, dx=dx)
        #
        # # energy drift
        # dydt_[-1] = energy_drift_two_stream(state_e1=state_e1,
        #                                     state_e2=state_e2,
        #                                     state_i=state_i,
        #                                     E=E, Nv=Nv,
        #                                     alpha_e1=alpha_e1, alpha_e2=alpha_e2, alpha_i=alpha_i,
        #                                     q_e1=q_e1, q_e2=q_e2, q_i=q_i, dx=dx,
        #                                     m_e1=m_e1, m_e2=m_e2, m_i=m_i, Nx=Nx,
        #                                     u_e1=u_e1, u_e2=u_e2, u_i=u_i)

    return dydt_


if __name__ == '__main__':
    # set up configuration parameters
    # number of mesh points in x
    Nx = 101
    # number of spectral expansions
    Nv = 101
    # epsilon displacement in initial electron distribution
    epsilon = 1e-3
    # velocity scaling of electron and ion
    alpha_e1 = 0.5
    alpha_e2 = 0.5
    alpha_i = np.sqrt(2 / 1863)
    # x grid is from 0 to L
    L = 2 * np.pi
    # spacial spacing dx = x[i+1] - x[i]
    dx = L / (Nx - 1)
    # time stepping
    dt = 1e-2
    # final time
    T = 45
    # timestamp vector
    t_vec = np.linspace(0, T, int(T / dt) + 1)
    # velocity scaling
    u_e1 = 1
    u_e2 = -1
    u_i = 0
    # mass normalized
    m_e1 = 1
    m_e2 = 1
    m_i = 1863
    # charge normalized
    q_e1 = -1
    q_e2 = -1
    q_i = 1

    # spatial grid
    x = np.linspace(0, L, Nx)

    # initial condition of the first expansion coefficient
    C_0e1 = np.sqrt(0.5 * (1 + epsilon * np.cos(x)) / alpha_e1)
    C_0e2 = np.sqrt(0.5 * (1 + epsilon * np.cos(x)) / alpha_e2)

    # initialize states (electrons type 1 and 2)
    states_e1 = np.zeros((Nv, Nx - 1))
    states_e2 = np.zeros((Nv, Nx - 1))

    # initialize the expansion coefficients
    states_e1[0, :] = C_0e1[:-1]
    states_e2[0, :] = C_0e2[:-1]

    # initial condition of the semi-discretized ODE
    y0 = np.zeros((2 * Nv) * (Nx - 1))
    y0[:(Nx - 1) * Nv] = states_e1.flatten("C")
    y0[Nv * (Nx - 1): 2 * Nv * (Nx - 1)] = states_e2.flatten("C")
    y0 = np.append(y0, np.zeros(2))

    # integrate (symplectic integrator: implicit midpoint)
    sol_midpoint_u = implicit_midpoint_solver(t_vec=t_vec, y0=y0, rhs=rhs,
                                              nonlinear_solver_type="newton_krylov",
                                              r_tol=1e-8, a_tol=1e-14, max_iter=100)

    np.save("../data/SW_sqrt/two_stream/sol_midpoint_u_101", sol_midpoint_u)
    np.save("../data/SW_sqrt/two_stream/sol_midpoint_t_101", t_vec)
