"""
Authors: Opal Issan (oissan@ucsd.edu)
Version: April 10th, 2024
"""
import numpy as np


def runge_kutta_4(t0, y0, tf, dt, rhs):
    # count number of iterations using step size
    n = int((tf - t0) / dt)
    # initialize solution
    y_sol = np.zeros((len(y0), n + 1))
    y_sol[:, 0] = y0

    for ii in range(n):
        "Apply Runge Kutta Formulas to find next value of y"
        k1 = rhs(t=dt*ii, y=y_sol[:, ii])
        k2 = rhs(t=dt*(ii + 0.5), y=y_sol[:, ii] + 0.5 * dt * k1)
        k3 = rhs(t=dt*(ii + 0.5), y=y_sol[:, ii] + 0.5 * dt * k2)
        k4 = rhs(t=dt*(ii + 1), y=y_sol[:, ii] + dt * k3)

        # Update next value of y
        y_sol[:, ii + 1] = y_sol[:, ii] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_sol