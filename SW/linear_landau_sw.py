"""Linear Landau Damping Module SW formulation solved via Vlasov-Poisson

Author: Opal Issan (oissan@ucsd.edu)
Date: November 21st, 2023
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))
import numpy as np
from operators.SW_sqrt import RHS
from FD_tools.finite_difference_operators import ddx_central
from operators.SW import solve_poisson_equation, integral_I0
from FD_tools.implicit_midpoint import implicit_midpoint_solver


def rhs(y, t):
    dydt_ = np.zeros(len(y))

    # initialize state coefficients
    state_e = np.zeros((Nv, Nx - 1))
    state_i = np.zeros((Nv, Nx - 1))
    # static/background ions
    state_i[0, :] = (1 / (np.sqrt(2 * np.sqrt(np.pi)))) * np.ones(Nx - 1) / alpha_i

    # re-arrange states in [Nv, Nx] order
    for jj in range(Nv):
        state_e[jj, :] = y[jj * (Nx - 1): (jj + 1) * (Nx - 1)]

    E = solve_poisson_equation(state_e=state_e,
                               state_i=state_i,
                               alpha_e=alpha_e,
                               alpha_i=alpha_i,
                               dx=dx,
                               Nx=Nx - 1,
                               Nv=Nv,
                               solver="gmres",
                               order_fd=2, L=L)

    for jj in range(Nv):
        dydt_[jj * (Nx - 1): (jj + 1) * (Nx - 1)] = RHS(state=state_e, m=jj, Nv=Nv,
                                                        alpha_s=alpha_e, q_s=q_e, dx=dx, Nx=Nx, m_s=m_e, E=E, u_s=u_e)

    # mass drift (even)
    dydt_[-1] = -dx * (q_e / m_e) * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2) * E.T @ state_e[-1, :] \
                -dx * (q_i / m_i) * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2) * E.T @ state_i[-1, :]

    # momentum drift (odd)
    dydt_[-2] = -dx * (Nv - 1) * integral_I0(n=Nv - 1) * E.T @ (alpha_e * q_e * state_e[-1, :] +
                                                                alpha_i * q_i * state_i[-1, :])
    # momentum (even)
    dydt_[-3] = -dx * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2) * E.T @ (u_e * q_e * state_e[-1, :] +
                                                                             u_i * q_i * state_i[-1, :])

    # energy drift (odd)
    dydt_[-4] = -dx * (Nv - 1) * integral_I0(n=Nv - 1) * E.T @ (u_e * q_e * state_e[-1, :] +
                                                                u_i * q_i * state_i[-1, :])
    # energy drift (even)
    D = ddx_central(Nx=Nx, dx=dx)
    D_pinv = np.linalg.pinv(D)
    dydt_[-5] = -dx * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2) * E.T @ (
                 q_e * ((2 * Nv - 1) * (alpha_e ** 2) + u_e ** 2) * state_e[-1, :]
               + q_i * ((2 * Nv - 1) * (alpha_i ** 2) + u_i ** 2) * state_i[-1, :]
               + q_e**2/m_e * D_pinv @ (E * state_e[-1, :])
               + q_i**2/m_i * D_pinv @ (E * state_i[-1, :]))

    return dydt_


if __name__ == '__main__':
    # set up configuration parameters
    # number of mesh points in x
    Nx = 101
    # number of spectral expansions
    Nv = 101
    # epsilon displacement in initial electron distribution
    epsilon = 1e-2
    # velocity scaling of electron and ion
    alpha_e = 1
    alpha_i = np.sqrt(1 / 1863)
    # x grid is from 0 to L
    L = 2 * np.pi
    # spacial spacing dx = x[i+1] - x[i]
    dx = L / (Nx - 1)
    # time stepping
    dt = 1e-2
    # final time
    T = 10.
    # vector of timestamps
    t_vec = np.linspace(0, T, int(T / dt) + 1)
    # velocity scaling
    u_e = 0
    u_i = 0
    # mass normalized
    m_e = 1
    m_i = 1863
    # charge normalized
    q_e = -1
    q_i = 1

    # spatial grid
    x = np.linspace(0, L, Nx)

    # initial condition of the first expansion coefficient
    C_0e = (1 / (np.sqrt(2 * np.sqrt(np.pi)))) * (1 + epsilon * np.cos(x)) / alpha_e

    # initialize states (electrons and ions)
    states_e = np.zeros((Nv, Nx - 1))

    # initialize the expansion coefficients
    states_e[0, :] = C_0e[:-1]

    # initial condition of the semi-discretized ODE
    y0 = states_e.flatten("C")
    y0 = np.append(y0, np.zeros(5))

    # integrate (symplectic integrator: implicit midpoint)
    sol_midpoint_u = implicit_midpoint_solver(t_vec=t_vec, y0=y0, rhs=rhs,
                                              nonlinear_solver_type="newton_krylov",
                                              r_tol=1e-8, a_tol=1e-14, max_iter=100)

    # save results
    np.save("../data/SW/linear_landau/sol_midpoint_u_101", sol_midpoint_u)
    np.save("../data/SW/linear_landau/sol_midpoint_t_101", t_vec)
