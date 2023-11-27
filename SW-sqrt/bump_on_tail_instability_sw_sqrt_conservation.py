"""Bump-on-Tail Instability Conservation Module SW formulation solved via Vlasov-Poisson

Author: Opal Issan (oissan@ucsd.edu)
Date: November 24th, 2023
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

import numpy as np
from operators.SW_sqrt import RHS, solve_poisson_equation_two_stream, momentum_drift_two_stream, energy_drift_two_stream
from FD_tools.implicit_midpoint import implicit_midpoint_solver


def rhs(y, t):
    dydt_ = np.zeros(2)

    # momentum drift
    dydt_[-2] = momentum_drift_two_stream(state_e1=state_e1,
                                          state_e2=state_e2,
                                          state_i=state_i,
                                          E=E, Nv=Nv, alpha_e1=alpha_e1,
                                          alpha_e2=alpha_e2, alpha_i=alpha_i,
                                          q_e1=q_e1, q_e2=q_e2, q_i=q_i, dx=dx)

    # energy drift
    dydt_[-1] = energy_drift_two_stream(state_e1=state_e1,
                                        state_e2=state_e2,
                                        state_i=state_i,
                                        E=E, Nv=Nv,
                                        alpha_e1=alpha_e1, alpha_e2=alpha_e2, alpha_i=alpha_i,
                                        q_e1=q_e1, q_e2=q_e2, q_i=q_i, dx=dx,
                                        m_e1=m_e1, m_e2=m_e2, m_i=m_i, Nx=Nx,
                                        u_e1=u_e1, u_e2=u_e2, u_i=u_i)

    return dydt_


if __name__ == '__main__':
    # set up configuration parameters
    # number of mesh points in x
    Nx = 101
    # number of spectral expansions
    Nv = 101
    # epsilon displacement in initial electron distribution
    epsilon = 0.03
    # velocity scaling of electron and ion
    alpha_e1 = np.sqrt(2)
    alpha_e2 = 1 / np.sqrt(2)
    alpha_i = np.sqrt(2 / 1863)
    # x grid is from 0 to L
    L = 20 * np.pi
    # spacial spacing dx = x[i+1] - x[i]
    dx = L / (Nx - 1)
    # time stepping
    dt = 1e-2
    # final time
    T = 20
    # timestamp vector
    t_vec = np.linspace(0, T, int(T / dt) + 1)
    # velocity scaling
    u_e1 = 0
    u_e2 = 4.5
    u_i = 0
    # mass normalized
    m_e1 = 1
    m_e2 = 1
    m_i = 1863
    # charge normalized
    q_e1 = -1
    q_e2 = -1
    q_i = 1

    # read simulation data
    sol_midpoint_u = np.load("../data/SW_sqrt/bump_on_tail/sol_midpoint_u_" + str(Nv))
    sol_midpoint_t = np.save("../data/SW_sqrt/bump_on_tail/sol_midpoint_t_" + str(Nv))

    # electric field initialization
    E = np.zeros((Nx - 1, len(sol_midpoint_t)))
    # initialize the states for implicit midpoint
    state_e1_midpoint = np.zeros((Nv, Nx - 1, len(sol_midpoint_t)))
    state_e2_midpoint = np.zeros((Nv, Nx - 1, len(sol_midpoint_t)))
    state_i_midpoint = np.zeros((Nv, Nx - 1, len(sol_midpoint_t)))

    for ii in range(len(sol_midpoint_t)):
        for jj in range(0, Nv):
            # unwind the flattening in order to solve the Vlasov-Poisson system
            state_e1_midpoint[jj, :, ii] = sol_midpoint_u[jj * (Nx - 1): (jj + 1) * (Nx - 1), ii]
            state_e2_midpoint[jj, :, ii] = sol_midpoint_u[
                                           Nv * (Nx - 1) + jj * (Nx - 1): Nv * (Nx - 1) + (jj + 1) * (Nx - 1), ii]
            # static/background ions
            state_i_midpoint[0, :, ii] = np.sqrt(np.ones(Nx - 1) / alpha_i)

        E_midpoint[:, ii] = solve_poisson_equation_two_stream(state_e1=state_e1_midpoint[:, :, ii],
                                                              state_e2=state_e2_midpoint[:, :, ii],
                                                              state_i=state_i_midpoint[:, :, ii],
                                                              alpha_e1=alpha_e1,
                                                              alpha_e2=alpha_e2,
                                                              alpha_i=alpha_i,
                                                              dx=dx, L=L)

    # integrate (symplectic integrator: implicit midpoint)
    sol_midpoint_u = implicit_midpoint_solver(t_vec=t_vec, y0=y0, rhs=rhs,
                                              nonlinear_solver_type="newton_krylov",
                                              r_tol=1e-8, a_tol=1e-14, max_iter=100)

    np.save("../data/SW_sqrt/bump_on_tail/sol_midpoint_u_" + str(Nv), sol_midpoint_u)
    np.save("../data/SW_sqrt/bump_on_tail/sol_midpoint_t_" + str(Nv), t_vec)
