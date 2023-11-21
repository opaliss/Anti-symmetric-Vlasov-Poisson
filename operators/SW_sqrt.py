"""Vlasov-Poisson operators for the SW-sqrt formulation

Author: Opal Issan (oissan@ucsd.edu)
Last Updated: November 20th, 2023
"""
from FD_tools.poisson_solver import gmres_solver, fft_solver, fft_solver_Ax_b
from FD_tools.finite_difference_operators import ddx_central, ddx_fwd, ddx_bwd
import numpy as np
from scipy.special import hermite, factorial


def linear_1(state, m, alpha_s, Nv, u_s):
    """Linear terms Q_{n}^{s}(t)

    :param state: ndarray, a matrix of coefficients of species s at time t=t*
    :param m: int, the expansion coefficient order
    :param alpha_s: float, the velocity scaling of species s
    :param u_s: float, the velocity shift of species s
    :param Nv: int, the number of spectral expansion terms
    :return: Q_{n}^{s}(t)
    """
    # the first term in the expansion
    if m == 0:
        term1 = 0 * state[m, :]
        term2 = alpha_s * np.sqrt((m + 1) / 2) * state[m + 1, :]
        term3 = u_s * state[m, :]
        return term1, term2, term3

    # the last term in the expansion
    elif m == Nv - 1:
        term1 = alpha_s * np.sqrt(m / 2) * state[m - 1, :]
        term2 = 0 * state[m, :]
        term3 = u_s * state[m, :]
        return term1, term2, term3

    # all other terms
    else:
        term1 = alpha_s * np.sqrt(m / 2) * state[m - 1, :]
        term2 = alpha_s * np.sqrt((m + 1) / 2) * state[m + 1, :]
        term3 = u_s * state[m, :]
        return term1, term2, term3


def psi_ln_sw(xi, n):
    """the Hermite basis function psi_{n}(xi_{s})

    :param xi: float or ndarray, xi^{s} scaled velocity, i.e. xi = (v - u^{s})/alpha^{s}
    :param n: int, order of polynomial
    :return: symmetrically weighted hermite polynomial of degree n evaluated at xi
    """
    # hermite polynomial of degree n
    hermite_function = hermite(n=n)
    # (pi *2^n n!)^{-1/2}
    factor = (np.sqrt(np.pi) * (2 ** n) * factorial(n=n)) ** (-1 / 2)
    return factor * hermite_function(xi) * np.exp(0.5 * (-xi ** 2))


def nonlinear(state, m, alpha_s, q_s, m_s, E, Nv):
    """nonlinear term in the Vlasov equation N^{s}(t)

    :param Nv: int, the number of velocity expansion terms
    :param state: ndarray, a matrix of all states of species s at time t=t*
    :param m: int, the expansion coefficient order
    :param alpha_s: float, the velocity scaling of species s
    :param q_s: float, normalized charge of species s
    :param m_s: float, normalized mass of species s
    :param E: ndarray, electric field
    :return: N^{s}(t)
    """
    if m == 0:
        return (q_s / (m_s * alpha_s)) * E * (np.sqrt((m + 1) / 2) * state[m + 1, :])
    elif m == Nv - 1:
        return (q_s / (m_s * alpha_s)) * E * (-np.sqrt(m / 2) * state[m - 1, :])
    else:
        return (q_s / (m_s * alpha_s)) * E * (np.sqrt((m + 1) / 2) * state[m + 1, :] - np.sqrt(m / 2) * state[m - 1, :])


def linear_2(state_e, state_i, alpha_e, alpha_i, q_e=-1, q_i=1):
    """charge density rho(t)

    :param q_i: float, ion normalized charge (+1)
    :param q_e: float, electron normalized charge (-1)
    :param state_e: ndarray, a matrix of all electron coefficients at time t=t* (length of Nv)
    :param state_i: ndarray, a matrix of all ion coefficients at time t=t*
    :param alpha_e: float, the velocity scaling of electrons
    :param alpha_i: float, the velocity scaling of ions
    :return: rho(t)
    """
    term1 = alpha_e * np.sum(state_e ** 2, axis=0)
    term2 = alpha_i * np.sum(state_i ** 2, axis=0)
    return q_i * term2 + q_e * term1


def linear_2_two_stream(state_e1, state_e2, state_i, alpha_e1, alpha_e2, alpha_i, q_e1=-1, q_e2=-1, q_i=1):
    """charge density rho(t)

    :param q_i: float, ion normalized charge (+1)
    :param q_e1: float, electron normalized charge (-1)
    :param q_e2: float, electron normalized charge (-1)
    :param state_e1: ndarray, a matrix of electron coefficients (species 1) at time t=t*
    :param state_e2: ndarray, a matrix of electron coefficients (species 2) at time t=t*
    :param state_i: ndarray, a matrix of all ion coefficients at time t=t*
    :param alpha_e1: float, the velocity scaling of electrons (species 1)
    :param alpha_e2: float, the velocity scaling of electrons (species 2)
    :param alpha_i: float, the velocity scaling of ions
    :return: rho(t)
    """
    term1 = alpha_e1 * np.sum(state_e1 ** 2, axis=0)
    term2 = alpha_e2 * np.sum(state_e2 ** 2, axis=0)
    term3 = alpha_i * np.sum(state_i ** 2, axis=0)
    return q_i * term3 + q_e2 * term2 + q_e1 * term1


def RHS(state, m, Nv, alpha_s, q_s, dx, Nx, m_s, E, u_s, periodic=True, df_type="central", order_fd=2):
    """the right-hand-side of the Vlasov equation

    :param order_fd: int, finite difference order.
    :param state: ndarray, a matrix of coefficients at time t=t* of species s
    :param m: int, the expansion coefficient order
    :param Nv: int, the number of spectral expansion terms
    :param alpha_s: float, the velocity scaling of species s
    :param q_s: float, normalized charge of species s
    :param Nx: int, number of spatial grid points in x
    :param dx: float, spatial spacing dx = x_{i+1} - x_{i} (we assume uniform spacing)
    :param m_s: float, normalized mass of species s
    :param E: ndarray, electric field at time t=t*
    :param u_s: float, the velocity shift of species s
    :param periodic: boolean, default is True
    :param df_type: str, operator type of derivative of electric potential
    :return: RHS of species s in Vlasov equation
    """
    term1, term2, term3 = linear_1(state=state, m=m, alpha_s=alpha_s, Nv=Nv, u_s=u_s)
    A = ddx_central(Nx=Nx, dx=dx, periodic=periodic, order=order_fd)

    if df_type == "central":
        return - nonlinear(state=state, m=m, alpha_s=alpha_s, q_s=q_s, m_s=m_s, E=E, Nv=Nv) \
               - A @ (term1 + term2 + term3)

    elif df_type == "mixture":
        F = ddx_fwd(Nx=Nx, dx=dx, periodic=periodic, order=1)
        B = ddx_bwd(Nx=Nx, dx=dx, periodic=periodic, order=1)
        return - nonlinear(state=state, m=m, alpha_s=alpha_s, q_s=q_s, m_s=m_s, E=E, Nv=Nv) \
               - F @ term1 - B @ term2 - A @ term3


def solve_poisson_equation(state_e, state_i, alpha_e, alpha_i, dx, L, solver="gmres", order_fd=2):
    """Poisson solver for single electron and ion species

    :param order_fd: int, finite difference order, default is 2nd order accurate
    :param solver: str, Poisson's equation solver, default is "gmres"
    :param L: float, spatial length
    :param state_e: ndarray, a matrix of electron coefficients at time t=t*
    :param state_i: ndarray, a matrix of ion coefficients at time t=t*
    :param alpha_e: float, the velocity scaling of electrons
    :param alpha_i: float, the velocity scaling of ions
    :param dx: float, spatial spacing dx = x_{i+1} - x_{i} (we assume uniform spacing)
    :return: E(t)
    """
    rhs = linear_2(state_e=state_e, state_i=state_i, alpha_e=alpha_e, alpha_i=alpha_i)

    if solver == "gmres":
        return gmres_solver(rhs=rhs, dx=dx, periodic=True, order="ddx", order_fd=order_fd)
    elif solver == "fft_solver":
        return fft_solver(rhs=rhs, L=L)


def solve_poisson_equation_two_stream(state_e1, state_e2, state_i,
                                      alpha_e1, alpha_e2, alpha_i,
                                      dx, L, solver="gmres", order_fd=2, order="ddx", periodic=True):
    """Poisson solver for two electrons and ion species

    :param order: str, solve Poisson "ddx" electric field or "d2dx2" electric potential, default is "ddx"
    :param order_fd: int, finite difference order, default is 2nd order accurate
    :param solver: str, Poisson's equation solver, default is "gmres"
    :param L: float, spatial length
    :param periodic: boolean, default is True
    :param state_e1: ndarray, a matrix of electron coefficients (species 1) at time t=t*
    :param state_e2: ndarray, a matrix of electron coefficients (species 2) at time t=t*
    :param state_i: ndarray, a matrix of ion coefficients at time t=t*
    :param alpha_e1: float, the velocity scaling of electrons (species 1)
    :param alpha_e2: float, the velocity scaling of electrons (species 2)
    :param alpha_i: float, the velocity scaling of ions
    :param dx: float, spatial spacing dx = x_{i+1} - x_{i} (we assume uniform spacing)
    :return: E(t)
    """
    rhs = linear_2_two_stream(state_e1=state_e1, state_e2=state_e2, state_i=state_i,
                              alpha_e1=alpha_e1, alpha_e2=alpha_e2, alpha_i=alpha_i)
    if solver == "gmres":
        return gmres_solver(rhs=rhs, dx=dx, periodic=periodic, order=order, order_fd=order_fd)
    elif solver == "fft_solver_Ax_b":
        return fft_solver_Ax_b(rhs=rhs, dx=dx)
    elif solver == "fft_solver":
        return fft_solver(rhs=rhs, L=L)


def ampere_term(state):
    """quadratic term Ampere' right-hand-side

    :param state: ndarray, coefficient of species s
    :return: quadratic term
    """
    Nv, Nx = np.shape(state)
    res = np.zeros(Nx)
    for m in range(1, Nv):
        res += np.sqrt(m) * state[m, :] * state[m - 1, :]
    return res


def ampere_equation_RHS(state_e, state_i, alpha_e, alpha_i, u_e, u_i, L, dx, q_e=-1, q_i=1):
    """the right-hand-side of the Ampere' equations with a single electron and ion species

    :param q_i: float, ion normalized charge (+1)
    :param q_e: float, electron normalized charge (-1)
    :param L: float, length of spatial dimension
    :param u_i: float, ion velocity shift
    :param u_e: float, electron velocity shift
    :param state_e: ndarray, a matrix of electron coefficients at time t=t*
    :param state_i: ndarray, a matrix of ion coefficients at time t=t*
    :param alpha_e: float, the velocity scaling of electrons
    :param alpha_i: float, the velocity scaling of ions
    :param dx: float, spatial spacing dx = x_{i+1} - x_{i} (we assume uniform spacing)
    :return: d/dt E(t)
    """
    # first term
    term1 = alpha_e * u_e * (-np.sum(state_e ** 2, axis=0) + 1 / L * dx * np.sum(state_e ** 2, axis=(0, 1)))
    term2 = alpha_i * u_i * (-np.sum(state_i ** 2, axis=0) + 1 / L * dx * np.sum(state_i ** 2, axis=(0, 1)))
    # second term
    term3 = np.sqrt(2) * (alpha_e ** 2) * (-ampere_term(state=state_e) + 1 / L * dx * ampere_term(state=state_e))
    term4 = np.sqrt(2) * (alpha_i ** 2) * (-ampere_term(state=state_i) + 1 / L * dx * ampere_term(state=state_i))
    return q_e * (term1 + term3) + q_i * (term2 + term4)


def ampere_equation_RHS_two_stream(state_e1, state_e2, state_i, alpha_e1, alpha_e2, alpha_i,
                                   u_e1, u_e2, u_i, L, dx, q_e1=-1, q_e2=-1, q_i=1):
    """the right-hand-side of the Ampere equation with two electron species and one ion species

    :param q_i: float, ion normalized charge (+1)
    :param q_e1: float, electron (species 1) normalized charge (-1)
    :param q_e2: float, electron (species 2) normalized charge (-1)
    :param L: float, length of spatial dimension
    :param u_i: float, ion velocity shift
    :param u_e1: float, electron (species 1) velocity shift
    :param u_e2: float, electron (species 2) velocity shift
    :param state_e2: ndarray, a matrix of electron (species 2) coefficients at time t=t*
    :param state_e1: ndarray, a matrix of electron (species 1) coefficients at time t=t*
    :param state_i: ndarray, a matrix of ion states at time t=t*
    :param alpha_e1: float, the velocity scaling of electrons (species 1)
    :param alpha_e2: float, the velocity scaling of electrons (species 2)
    :param alpha_i: float, the velocity scaling of ions
    :param dx: float, spatial spacing dx = x_{i+1} - x_{i} (we assume uniform spacing)
    :return: d/dt E(x, t=t*)
    """
    # first term
    term1 = alpha_e1 * u_e1 * (-np.sum(state_e1 ** 2, axis=0) + 1 / L * dx * np.sum(state_e1 ** 2, axis=(0, 1)))
    term2 = alpha_e2 * u_e2 * (-np.sum(state_e2 ** 2, axis=0) + 1 / L * dx * np.sum(state_e2 ** 2, axis=(0, 1)))
    term3 = alpha_i * u_i * (-np.sum(state_i ** 2, axis=0) + 1 / L * dx * np.sum(state_i ** 2, axis=(0, 1)))
    # second term
    term4 = np.sqrt(2) * (alpha_e1 ** 2) * (-ampere_term(state=state_e1) + 1 / L * dx * ampere_term(state=state_e1))
    term5 = np.sqrt(2) * (alpha_e2 ** 2) * (-ampere_term(state=state_e2) + 1 / L * dx * ampere_term(state=state_e2))
    term6 = np.sqrt(2) * (alpha_i ** 2) * (-ampere_term(state=state_i) + 1 / L * dx * ampere_term(state=state_i))
    return q_e1 * (term1 + term4) + q_e2 * (term2 + term5) + q_i * (term3 + term6)
