"""Vlasov-Poisson operators for the SW formulation

Author: Opal Issan (oissan@ucsd.edu)
Last Update: November 27th, 2023
"""
import numpy as np
from FD_tools.poisson_solver import gmres_solver, fft_solver, fft_solver_Ax_b


def integral_I0(n):
    """
    :param n: int, the order of the integral
    :return: the integral I0_{n}
    """
    if n < 0:
        return 0
    elif n == 0:
        return np.sqrt(2) * (np.pi ** (1 / 4))
    elif n % 2 == 1:
        return 0
    else:
        term = np.zeros(n+10)
        term[0] = np.sqrt(2) * (np.pi ** (1 / 4))
        for m in range(2, n+10):
            term[m] = np.sqrt((m - 1) / m) * term[m - 2]
        return term[n]


def integral_I1(n, u_s, alpha_s):
    """
    :param n: int, order of the integral
    :param u_s: float, the velocity shifting of species s
    :param alpha_s: float, the velocity scaling of species s
    :return: the integral I1_{n}
    """
    if n % 2 == 0:
        return u_s * integral_I0(n=n)
    else:
        return alpha_s * np.sqrt(2) * np.sqrt(n) * integral_I0(n=n - 1)


def integral_I2(n, u_s, alpha_s):
    """integral I2 in SW formulation

    :param n: int, order of the integral
    :param u_s: float, the velocity shifting of species s
    :param alpha_s: float, the velocity scaling of species s
    :return: the integral I2_{n}
    """
    if n % 2 == 0:
        return (alpha_s ** 2) * (0.5 * np.sqrt((n + 1) * (n + 2)) * integral_I0(n=n + 2) + (
                (2 * n + 1) / 2 + (u_s / alpha_s) ** 2) * integral_I0(n=n) + 0.5 * np.sqrt(n * (n - 1)) *
                                 integral_I0(n=n - 2))
    else:
        return 2 * u_s * integral_I1(n=n, u_s=u_s, alpha_s=alpha_s)


def linear_2(state_e, state_i, alpha_e, alpha_i, Nv_e, Nv_i, Nx, q_e=-1, q_i=1):
    """charge density for single electron and ion species

    :param Nx: int, the number of grid points in space
    :param Nv: int, the number of spectral terms
    :param state_e: ndarray, a matrix of electron coefficients at time t=t*
    :param state_i: ndarray, a matrix of ion coefficients at time t=t*
    :param alpha_e: float, the velocity scaling of electrons
    :param alpha_i: float, the velocity scaling of ions
    :param q_e: float, the normalized charge of electrons
    :param q_i: float, the normalized charge of ions
    :return: L_{2}(t)
    """
    term1 = np.zeros(Nx)
    term2 = np.zeros(Nx)
    for m in range(Nv_e):
        term1 += alpha_e * state_e[m, :] * integral_I0(n=m)
    for m in range(Nv_i):
        term2 += alpha_i * state_i[m, :] * integral_I0(n=m)
    return q_i * term2 + q_e * term1


def linear_2_two_stream(state_e1, state_e2, state_i, alpha_e1, alpha_e2, alpha_i, Nx, Nv_e1, Nv_e2, Nv_i, q_e1=-1, q_e2=-1, q_i=1):
    """charge density for two electron species and single ion species

    :param q_e1: float, the normalized charge of electron species 1
    :param q_e2: float, the normalized charge of electron species 2
    :param q_i: float, the normalized charge of ions
    :param Nv: int, the number of spectral terms in velocity
    :param Nx: int, the number of grid points in space
    :param state_e1: ndarray, a matrix of electron coefficients (species 1) at time t=t*
    :param state_e2: ndarray, a matrix of electron coefficients (species 2) at time t=t*
    :param state_i: ndarray, a matrix of ion states at time t=t*
    :param alpha_e1: float, the velocity scaling of electrons (species 1)
    :param alpha_e2: float, the velocity scaling of electrons (species 2)
    :param alpha_i: float, the velocity scaling of ions
    :return: L_{2}(t)
    """
    term1 = np.zeros(Nx)
    term2 = np.zeros(Nx)
    term3 = np.zeros(Nx)
    for m in range(Nv_e1):
        term1 += alpha_e1 * state_e1[m, :] * integral_I0(n=m)
    for m in range(Nv_e2):
        term2 += alpha_e2 * state_e2[m, :] * integral_I0(n=m)
    for m in range(Nv_i):
        term3 += alpha_i * state_i[m, :] * integral_I0(n=m)
    return q_i * term3 + q_e2 * term2 + q_e1 * term1


def solve_poisson_equation(state_e, state_i, alpha_e, alpha_i, dx, Nx, Nv_e, Nv_i, L, solver="gmres", order_fd=2):
    """solver Poisson equation for single electron and ion species

    :param L: float, spatial domain length
    :param solver: str, solver type for Poisson's equation, default is "gmres"
    :param order_fd: int, finite difference order, default is 2nd order accurate
    :param Nv: int, the number of spectral terms in velocity
    :param Nx: int, the number of grid points in space
    :param state_e: ndarray, a matrix of electron coefficients at time t=t*
    :param state_i: ndarray, a matrix of ion coefficients at time t=t*
    :param alpha_e: float, the velocity scaling of electrons
    :param alpha_i: float, the velocity scaling of ions
    :param dx: float, spatial spacing dx = x_{i+1} - x_{i} (we assume uniform spacing)
    :return: E(t)
    """
    rhs = linear_2(state_e=state_e, state_i=state_i, alpha_e=alpha_e, alpha_i=alpha_i, Nx=Nx, Nv_e=Nv_e, Nv_i=Nv_i)
    if solver == "gmres":
        return gmres_solver(rhs=rhs, dx=dx, periodic=True, order="ddx", order_fd=order_fd)
    elif solver == "fft_solver":
        return fft_solver(rhs=rhs, L=L)
    elif solver == "fft_solver_Ax_b":
        return fft_solver_Ax_b(rhs=rhs, dx=dx)


def solve_poisson_equation_two_stream(state_e1, state_e2, state_i, alpha_e1, alpha_e2, alpha_i, dx, Nx,
                                      Nv_e1, Nv_e2, Nv_i, L, solver="gmres", periodic=True, order_fd=2):
    """solver Poisson equation for two electron species and single ion species

    :param L: float, spatial domain length
    :param order_fd: int, finite difference order, default is 2
    :param periodic: boolean, default is True
    :param solver: str, Poisson's equation solver, default is "gmres"
    :param Nv: int, the number of spectral terms in velocity
    :param Nx: int, the number of grid points in space
    :param state_e1: ndarray, a matrix of electron coefficients at time t=t*
    :param state_e2: ndarray, a matrix of electron coefficients at time t=t*
    :param state_i: ndarray, a matrix of all ion states at time t=t*
    :param alpha_e1: float, the velocity scaling of electrons (species 1)
    :param alpha_e2: float, the velocity scaling of electrons (species 2)
    :param alpha_i: float, the velocity scaling of ions
    :param dx: float, spatial spacing dx = x_{i+1} - x_{i} (we assume uniform spacing)
    :return: E(t)
    """
    rhs = linear_2_two_stream(state_e1=state_e1,
                              state_e2=state_e2,
                              state_i=state_i,
                              alpha_e1=alpha_e1,
                              alpha_e2=alpha_e2,
                              alpha_i=alpha_i,
                              Nx=Nx,
                              Nv_e1=Nv_e1,
                              Nv_e2=Nv_e2,
                              Nv_i=Nv_i)
    if solver == "gmres":
        return gmres_solver(rhs=rhs, dx=dx, periodic=periodic, order="ddx", order_fd=order_fd)
    elif solver == "fft_solver":
        return fft_solver(rhs=rhs, L=L)
    elif solver == "fft_solver_Ax_b":
        return fft_solver_Ax_b(rhs=rhs, dx=dx)


def mass(state, Nv):
    """mass of the particular state

    :param state: ndarray, electron or ion state
    :param Nv: int, number of velocity Hermite spectral terms
    :return: mass for the state
    """
    res = 0
    for m in range(Nv):
        res += integral_I0(n=m) * np.sum(state[m, :])
    return res


def momentum(state, u_s, alpha_s, Nv):
    """momentum of the particular state

    :param state: ndarray, electron or ion state
    :param Nv: int, number of velocity Hermite spectral terms
    :param u_s: float, the velocity shifting parameter of species s
    :param alpha_s: float, the velocity scaling parameter of species s
    :return: momentum for the state
    """
    res = 0
    for m in range(Nv):
        res += integral_I1(n=m, u_s=u_s, alpha_s=alpha_s) * np.sum(state[m, :])
    return res


def energy_k(state, u_s, alpha_s, Nv):
    """kinetic energy of the particular state

    :param state: ndarray, electron or ion state
    :param Nv: int, number of velocity Hermite spectral terms
    :param u_s: float, the velocity shifting parameter of species s
    :param alpha_s: float, the velocity scaling parameter of species s
    :return: kinetic energy for the state
    """
    res = 0
    for m in range(Nv):
        res += integral_I2(n=m, u_s=u_s, alpha_s=alpha_s) * np.sum(state[m, :])
    return res


def total_mass(state, alpha_s, dx, Nv):
    """total mass of single electron and ion setup

    :param state: ndarray, species s state
    :param alpha_s: float, velocity scaling of species s
    :param dx: float, spatial spacing
    :param Nv: int, the number of velocity spectral terms
    :return: total mass of single electron and ion setup
    """
    return mass(state=state, Nv=Nv) * dx * alpha_s


def total_momentum(state, alpha_s, dx, Nv, m_s, u_s):
    """total momentum of single electron and ion setup

    :param state: ndarray, species s state
    :param alpha_s: float, velocity scaling of species s
    :param dx: float, spatial spacing
    :param Nv: int, the number of velocity spectral terms
    :param m_s: float, mass of species s
    :param u_s: float, velocity shifting parameter of species s
    :return: total momentum of single electron and ion setup
    """
    return momentum(state=state, Nv=Nv, alpha_s=alpha_s, u_s=u_s) * dx * alpha_s * m_s


def total_energy_k(state, alpha_s, dx, Nv, m_s, u_s):
    """total kinetic energy of single electron and ion setup

    :param state: ndarray, species s  state
    :param alpha_s: float, velocity scaling of species s
    :param dx: float, spatial spacing
    :param Nv: int, the number of velocity spectral terms
    :param m_s: float, mass of species s
    :param u_s: float, velocity shifting parameter of species s
    :return: total kinetic energy of single electron and ion setup
    """
    return 0.5 * energy_k(state=state, Nv=Nv, alpha_s=alpha_s, u_s=u_s) * dx * alpha_s * m_s
