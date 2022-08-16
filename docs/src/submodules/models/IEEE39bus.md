# IEEE39bus

This is a version of the IEEE-39 bus test case as described by 
[Nishikawa, T. et al.](https://iopscience.iop.org/article/10.1088/1367-2630/17/1/015012)
The model, denoted the "effective network", consists of the ten generator buses in the
network with all other buses eliminated by the classical Kron reduction.
The power flow is described in steady state by a fixed point of the nonlinear swing equations

$$\begin{align}
\frac{2H_i}{\omega_\mathrm{R}} \ddot{\delta}_i + \frac{D_i}{\omega_\mathrm{R}} \dot{\delta}_i = A_{i}^\mathrm{EN} - \sum_{j =1, j\neq i}^{n_g} K_{ij}\sin\left(\delta_i - \delta_j -\gamma_{ij}^\mathrm{EN}\right),
\end{align}$$

where we define each of the following:
  * the angular reference frequency (in radians) about which the steady state synchronizes is defined as $\omega_\mathrm{R}$;
  * the angle of rotation of the generator rotor at bus $i$, relative to the frame rotating at the reference frequency, is defined as $\delta_i(t)$;
  * the difference between the reference frequency and the frequency of the rotor at bus $i$ is defined $\dot{\delta}_i(t)$;
  * the rate of acceleration of the difference between the angle of the rotor at bus $i$ and the frame rotating at the reference frequency is defined as $\ddot{\delta}_i(t)$;
  * the values of the inertia and damping at bus $i$ are defined as $H_i$ and $D_i$ respectively;
  * the strength of the dynamical coupling of the buses $i$ and $j$ is defined as $K_{ij}$, while $\gamma_{ij}$ represents the phase shift involved in the coupling of these buses;
  * the active power injected into the network by the generator at bus $i$ is represented by $A^\mathrm{EN}_i$; and
  * the number of generators in the network is defined as $n_g =10$.

This model assumes constant, passive loads at each bus that draws power.
The actual parameters used in the model are defined by files in the
```
DataAssimilationBenchmarks/src/models/IEEE39bus_inputs/
```
directory, taken from the configuration studied by 
[Nishikawa, T. et al.](https://iopscience.iop.org/article/10.1088/1367-2630/17/1/015012),
with details on their interpretation in section 4.1.

The stochastic form in this code loosens the assumption of constant loads in this model by
assuming that, at the time scale of interest, the draw of power fluctuates randomly about
the constant level that defines the steady state. We introduce a Wiener process to 
the above equations of the form  $s W_i(t)$, where $s$ is a parameter in the model
controlling the relative diffusion level.  We assume that the fluctuations in the net
power are uncorrelated across buses and that the diffusion in all buses is proportional to
$s$. 

Making a change of variables $\psi_i =  \dot{\delta}_i$, we recover the system of nonlinear
SDEs,

$$\begin{align}
\dot{\delta}_i = \psi_i,
\end{align}$$
$$\begin{align}
\dot{\psi}_i = \frac{A^\mathrm{EN}_i \omega_\mathrm{R}}{2H_i} - \frac{D_i}{2H_i} \psi_i -
\sum_{j=1,j\neq i}^{n_g} \frac{K_{ij}^\mathrm{EN}\omega_\mathrm{R}}{2H_i} \sin\left(\delta_i - \delta_j -\gamma_{ij}^\mathrm{EN}\right) + \frac{ s \omega_R}{2 H_i} \mathrm{d}W_i(t).
\end{align}$$

The diffusion level $s$ controls the standard deviation of the Gaussian process

$$\begin{align}
\frac{s \omega_R}{2H_i} W_{i,\Delta_t}\doteq \frac{s \omega_R}{2H_i}\left(W_i(\Delta + t) - W_i(t)\right).
\end{align}$$

By definition the standard deviation of $W_{i,\Delta_t}$ is equal to $\sqrt{\Delta}$ so that
for each time-discretization of the Wiener process of step size $\Delta$,
$\frac{s \omega_R}{2 H_i}W_{i,\Delta_t}$ is a mean zero, Gaussian distributed variable
with standard deviation $\frac{s \omega_\mathrm{R}}{2}\sqrt{\Delta}$.  The reference
frequency in North America is 60 Hz and the tolerable deviation from this frequency under
normal operations is approximately $\pm 0.05$ Hz, or of magnitude
approximately $0.08\%$.  In the above model, the
reference frequency is in radians, related to the reference frequency in Hz as
$\omega_\mathrm{R} = 60 \mathrm{Hz} \times 2 \pi \approx 376.99$.  This makes the
tolerable limit of perturbations to the frequency approximately $0.3$ radians under normal
operations.

By definition $\psi_i$ is the $i$-th frequency relative to the reference frequency
$\omega_\mathrm{R}$. One should choose $s$ sufficiently small such that the probability
that the size of a perturbation to the frequency 

$$\begin{align}
\parallel \frac{s \omega_\mathrm{R}}{2 H_i}\mathbf{W}_{\Delta_t} \parallel\geq 0.3
\end{align}$$

is small.  Simulating the model numerically with the four-stage, stochastic Runge-Kutta
algorithm
[`DataAssimilationBenchmarks.DeSolvers.rk4_step!`](@ref)
a step size of $\Delta=0.01$ is recommended, so that the standard deviation of
a perturbation to the $i$-th relative frequency $\psi_i$ at any time step is
$\frac{s \omega_\mathrm{R}}{20 H_i}$. The smallest inertia parameter in the model is
approximately $24.3$, so that three standard deviations of the perturbation
to the frequency is bounded as

$$\begin{align}
\frac{s\omega_\mathrm{R}}{20 \times 24.3} \times 3 \leq 0.03  \Leftrightarrow  s \leq\frac{4.86}{\omega_\mathrm{R}} \approx 0.0129.
\end{align}$$

For $s \leq 0.012$, we bound the standard deviation of each component,
$\frac{s \omega_\mathrm{R}}{2H_i}\sqrt{\Delta}$, of the perturbation vector by $0.01$ so
that over $99.7\%$ of perturbations to the $i$-th frequency have size less than $0.03$.

## Methods

```@autodocs
Modules = [DataAssimilationBenchmarks.IEEE39bus]
```
