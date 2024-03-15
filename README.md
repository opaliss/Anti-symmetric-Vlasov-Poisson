## Anti-symmetric and Positivity Preserving Formulation of a Spectral Method for Vlasov-Poisson Equations


## Abstract 
We analyze the anti-symmetric properties of spectral discretization for the one-dimensional Vlasov-Poisson equations. The discretization is based on a spectral expansion in velocity with the symmetrically weighted Hermite basis functions, central finite differencing in space, and an implicit Runge Kutta integrator in time. The proposed discretization preserves the anti-symmetric structure of the advection operator in the Vlasov equation, resulting in a stable numerical method. We apply such discretization to two formulations: the canonical Vlasov-Poisson equations and their continuous transformed square-root representation. The latter preserves the positivity of the particle distribution function. We derive analytically the conservation properties of both formulations, including particle number, momentum, and energy, which are verified numerically on the following benchmark problems: manufactured solution, linear and nonlinear Landau damping, two-stream instability, and bump-on-tail instability. 

## Python Dependencies
1. Python >= 3.9.13
2. numpy >= 1.23.3
3. matplotlib >= 3.6.0
4. scipy >= 1.7.1
5. notebook >=6.4.3


## Correspondence
[Opal Issan](https://opaliss.github.io/opalissan/) (Ph.D. student), University of California San Diego. email: oissan@ucsd.edu
