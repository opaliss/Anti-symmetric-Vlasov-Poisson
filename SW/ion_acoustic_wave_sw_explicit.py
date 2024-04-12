"""ion-acoustic wave module SW formulation solved via Vlasov-Poisson

Author: Opal Issan (oissan@ucsd.edu)
Date: December 4th, 2023

Nx=16;Nn=50;nu=1e0;dt=0.01;T=400;Nt=T/dt;
axe=sqrt(2);uxe=0;
axi=sqrt(2)/135;uxi=0;Lx=10; % IAW
mass ratio = hydrogren mass ratio 1863
ions are cold dampens quickly/ probably does not need a lot of basis because its cold
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))
import numpy as np
from operators.SW_sqrt import RHS
from operators.SW import solve_poisson_equation, integral_I0
from FD_tools.finite_difference_operators import ddx_central
import scipy.integrate
from FD_tools.explicit_runge_kutta import runge_kutta_4
from FD_tools.implicit_midpoint import implicit_midpoint_solver


def rhs(t, y):
    dydt_ = np.zeros(len(y))

    # initialize species
    state_e = np.zeros((Nv_e, Nx - 1))
    state_i = np.zeros((Nv_i, Nx - 1))

    for jj in range(Nv_e):
        state_e[jj, :] = y[jj * (Nx - 1): (jj + 1) * (Nx - 1)]
    for jj in range(Nv_i):
        state_i[jj, :] = y[Nv_e * (Nx - 1) + jj * (Nx - 1): Nv_e * (Nx - 1) + (jj + 1) * (Nx - 1)]

    E = solve_poisson_equation(state_e=state_e,
                               state_i=state_i,
                               alpha_e=alpha_e,
                               alpha_i=alpha_i,
                               dx=dx,
                               Nx=Nx - 1,
                               Nv_e=Nv_e,
                               Nv_i=Nv_i,
                               solver="gmres",
                               order_fd=2,
                               L=L)

    for jj in range(Nv_e):
        dydt_[jj * (Nx - 1): (jj + 1) * (Nx - 1)] = RHS(state=state_e,
                                                        m=jj,
                                                        Nv=Nv_e,
                                                        alpha_s=alpha_e,
                                                        q_s=q_e,
                                                        dx=dx,
                                                        Nx=Nx,
                                                        m_s=m_e,
                                                        E=E,
                                                        u_s=u_e)
    for jj in range(Nv_i):
        dydt_[Nv_e * (Nx - 1) + jj * (Nx - 1): Nv_e * (Nx - 1) + (jj + 1) * (Nx - 1)] = RHS(state=state_i,
                                                                                            m=jj,
                                                                                            Nv=Nv_i,
                                                                                            alpha_s=alpha_i,
                                                                                            q_s=q_i,
                                                                                            dx=dx,
                                                                                            Nx=Nx,
                                                                                            m_s=m_i,
                                                                                            E=E,
                                                                                            u_s=u_i)

    # mass (even)
    dydt_[-1] = -dx * (q_e / m_e) * np.sqrt((Nv_e - 1) / 2) * integral_I0(n=Nv_e - 2) * E.T @ state_e[-1, :] \
                - dx * (q_i / m_i) * np.sqrt((Nv_i - 1) / 2) * integral_I0(n=Nv_i - 2) * E.T @ state_i[-1, :]

    # momentum (odd)
    dydt_[-2] = -dx * Nv_e * integral_I0(n=Nv_e - 1) * E.T @ (alpha_e * q_e * state_e[-1, :]) \
                -dx * Nv_i * integral_I0(n=Nv_i - 1) * E.T @ (alpha_i * q_i * state_i[-1, :])
    # momentum (even)
    dydt_[-3] = -dx * np.sqrt((Nv_e - 1) / 2) * integral_I0(n=Nv_e - 2) * E.T @ (u_e * q_e * state_e[-1, :]) \
                -dx * np.sqrt((Nv_i - 1) / 2) * integral_I0(n=Nv_i - 2) * E.T @ (u_i * q_i * state_i[-1, :])

    # energy (odd)
    dydt_[-4] = -dx * Nv_e * integral_I0(n=Nv_e - 1) * E.T @ (u_e * q_e * alpha_e * state_e[-1, :]) \
                -dx * Nv_i  * integral_I0(n=Nv_i - 1) * E.T @ (u_i * q_i * alpha_i * state_i[-1, :])

    # energy (even)
    D = ddx_central(Nx=Nx, dx=dx)
    D_pinv = np.linalg.pinv(D)
    dydt_[-5] = -dx * np.sqrt((Nv_e - 1) / 2) * integral_I0(n=Nv_e - 2) * E.T @ (
            0.5 * q_e * ((2 * Nv_e - 1) * (alpha_e ** 2) + u_e ** 2) * state_e[-1, :]
            + q_e ** 2 / m_e * D_pinv @ (E * state_e[-1, :])) \
            -dx * np.sqrt((Nv_i - 1) / 2) * integral_I0(n=Nv_i - 2) * E.T @ (
            0.5 * q_i * ((2 * Nv_i - 1) * (alpha_i ** 2) + u_i ** 2) * state_i[-1, :]
            + q_i ** 2 / m_i * D_pinv @ (E * state_i[-1, :]))
    print(t)
    return dydt_


if __name__ == '__main__':
    # set up configuration parameters
    # number of mesh points in x
    Nx = 51
    # number of spectral expansions
    Nv_e = 51
    Nv_i = 51
    # epsilon displacement in initial electron distribution
    epsilon = 0.01
    # velocity scaling of electron and ion
    alpha_e = 1
    alpha_i = 1 / 135
    # x grid is from 0 to L
    L = 10
    # spacial spacing dx = x[i+1] - x[i]
    dx = L / (Nx - 1)
    # time stepping
    dt = 0.01
    # final time (non-dimensional)
    T = 600
    t_vec = np.linspace(0, T, int(T/dt) + 1)
    # velocity scaling
    u_e = 0
    u_i = 0
    # mass normalized
    m_e = 1
    m_i = 1836
    # charge normalized
    q_e = -1
    q_i = 1

    # x direction
    x = np.linspace(0, L, Nx)

    # initialize states (electrons and ions)
    states_e = np.zeros((Nv_e, Nx - 1))
    states_i = np.zeros((Nv_i, Nx - 1))

    # initialize the expansion coefficients
    states_e[0, :] = ((1 / (np.sqrt(2 * np.sqrt(np.pi)))) * (1 + epsilon * np.cos(2 * np.pi * x / L)) / alpha_e)[:-1]
    states_i[0, :] = ((1 / (np.sqrt(2 * np.sqrt(np.pi)))) * (1 + epsilon * np.cos(2 * np.pi * x / L)) / alpha_i)[:-1]

    # initial condition of the semi-discretized ODE
    y0 = np.append(states_e.flatten("C"), states_i.flatten("C"))
    y0 = np.append(y0, np.zeros(5))

    # # set up implicit midpoint
    # sol_midpoint_u = implicit_midpoint_solver(t_vec=t_vec, y0=y0, rhs=rhs,
    #                                           nonlinear_solver_type="newton_krylov",
    #                                           r_tol=1e-8, a_tol=1e-14, max_iter=250, inner_maxiter=200)
    #
    # np.save("../data/SW/ion_acoustic/sol_midpoint_u_" + str(Nv_e) + "_dt_" + str(dt) + "_T_" + str(T), sol_midpoint_u)
    # np.save("../data/SW/ion_acoustic/sol_midpoint_t_" + str(Nv_e) + "_dt_" + str(dt) + "_T_" + str(T), t_vec)
    #

    # set up implicit midpoint
    # sol_midpoint_u = scipy.integrate.solve_ivp(fun=rhs, t_span=[0, T], y0=y0, method='RK45')
    sol_midpoint_u = runge_kutta_4(t0=0, y0=y0, tf=T, dt=dt, rhs=rhs)

    np.save("../data/SW/ion_acoustic/sol_midpoint_u_" + str(Nv_e) + "_dt_" + str(dt) + "_T_" + str(T) + "_RK4", sol_midpoint_u)
    np.save("../data/SW/ion_acoustic/sol_midpoint_t_" + str(Nv_e) + "_dt_" + str(dt) + "_T_" + str(T) + "_RK4", t_vec)
