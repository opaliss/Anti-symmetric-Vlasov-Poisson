"""Ion-Acoustic Instability Module SW formulation solved via Vlasov-Poisson

Author: Opal Issan (oissan@ucsd.edu)
Date: December 4th, 2023
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))
import numpy as np
from operators.SW_sqrt import RHS
from operators.SW import solve_poisson_equation, integral_I0
from FD_tools.finite_difference_operators import ddx_central
from FD_tools.implicit_midpoint import implicit_midpoint_solver


def rhs(y, t):
    dydt_ = np.zeros(len(y))

    # initialize species
    state_e = np.zeros((Nv, Nx - 1))
    state_i = np.zeros((Nv, Nx - 1))

    for jj in range(Nv):
        state_e[jj, :] = y[jj * (Nx - 1): (jj + 1) * (Nx - 1)]
        state_i[jj, :] = y[Nv * (Nx - 1) + jj * (Nx - 1): Nv * (Nx - 1) + (jj + 1) * (Nx - 1)]

    E = solve_poisson_equation(state_e=state_e,
                               state_i=state_i,
                               alpha_e=alpha_e,
                               alpha_i=alpha_i,
                               dx=dx,
                               Nx=Nx - 1,
                               Nv=Nv,
                               solver="gmres",
                               order_fd=2,
                               L=L)

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

        dydt_[Nv * (Nx - 1) + jj * (Nx - 1): Nv * (Nx - 1) + (jj + 1) * (Nx - 1)] = RHS(state=state_i,
                                                                                        m=jj,
                                                                                        Nv=Nv,
                                                                                        alpha_s=alpha_i,
                                                                                        q_s=q_i,
                                                                                        dx=dx,
                                                                                        Nx=Nx,
                                                                                        m_s=m_i,
                                                                                        E=E,
                                                                                        u_s=u_i)

    # mass (even)
    dydt_[-1] = -dx * (q_e / m_e) * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2) * E.T @ state_e[-1, :] \
                - dx * (q_i / m_i) * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2) * E.T @ state_i[-1, :]

    # momentum (odd)
    dydt_[-2] = -dx * (Nv - 1) * integral_I0(n=Nv - 1) * E.T @ (alpha_e * q_e * state_e[-1, :] +
                                                                alpha_i * q_i * state_i[-1, :])
    # momentum (even)
    dydt_[-3] = -dx * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2) * E.T @ (u_e * q_e * state_e[-1, :] +
                                                                             u_i * q_i * state_i[-1, :])

    # energy (odd)
    dydt_[-4] = -dx * (Nv - 1) * integral_I0(n=Nv - 1) * E.T @ (u_e * q_e * state_e[-1, :] +
                                                                u_i * q_i * state_i[-1, :])
    # energy (even)
    D = ddx_central(Nx=Nx, dx=dx)
    D_pinv = np.linalg.pinv(D)
    dydt_[-5] = -dx * np.sqrt((Nv - 1) / 2) * integral_I0(n=Nv - 2) * E.T @ (
            q_e * ((2 * Nv - 1) * (alpha_e ** 2) + u_e ** 2) * state_e[-1, :]
            + q_i * ((2 * Nv - 1) * (alpha_i ** 2) + u_i ** 2) * state_i[-1, :]
            + q_e ** 2 / m_e * D_pinv @ (E * state_e[-1, :])
            + q_i ** 2 / m_i * D_pinv @ (E * state_i[-1, :]))

    return dydt_


if __name__ == '__main__':
    # set up configuration parameters
    # number of mesh points in x
    Nx = 101
    # number of spectral expansions
    Nv = 100
    # epsilon displacement in initial electron distribution
    epsilon = 1e-2
    # velocity scaling of electron and ion
    alpha_e = 1
    alpha_i = np.sqrt(1 / 50)
    # x grid is from 0 to L
    L = 10 * np.pi
    # spacial spacing dx = x[i+1] - x[i]
    dx = L / (Nx - 1)
    # time stepping
    dt = 0.2
    # final time (non-dimensional)
    T = 100.
    t_vec = np.linspace(0, T, int(T / dt) + 1)
    # velocity scaling
    u_e = 2
    u_i = 0
    # mass normalized
    m_e = 1
    m_i = 25
    # charge normalized
    q_e = -1
    q_i = 1

    # x direction
    x = np.linspace(0, L, Nx)

    # initialize states (electrons and ions)
    states_e = np.zeros((Nv, Nx-1))
    states_i = np.zeros((Nv, Nx-1))

    # initialize the expansion coefficients
    states_e[0, :] = ((1 / (np.sqrt(2 * np.sqrt(np.pi)))) * (1 + epsilon * np.cos(x / 5)) / alpha_e)[:-1]
    states_i[0, :] = (1 / (np.sqrt(2 * np.sqrt(np.pi)))) / alpha_i * np.ones(Nx)[:-1]

    # initial condition of the semi-discretized ODE
    y0 = np.append(states_e.flatten("C"), states_i.flatten("C"))

    # set up implicit midpoint
    sol_midpoint_u = implicit_midpoint_solver(t_vec=t_vec, y0=y0, rhs=rhs,
                                              nonlinear_solver_type="newton_krylov",
                                              r_tol=1e-8, a_tol=1e-14, max_iter=100)

    # save results
    np.save("data/SW/ion_acoustic/sol_midpoint_u_" + str(Nv), sol_midpoint_u)
    np.save("data/SW/ion_acoustic/sol_midpoint_t_" + str(Nv), t_vec)
