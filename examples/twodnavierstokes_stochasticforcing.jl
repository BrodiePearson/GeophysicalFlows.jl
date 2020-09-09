# # 2D forced-dissipative turbulence
#
#md # This example can be run online via [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/generated/twodnavierstokes_stochasticforcing.ipynb).
#md # Also, it can be viewed as a Jupyter notebook via [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/twodnavierstokes_stochasticforcing.ipynb).
#
# A simulation of forced-dissipative two-dimensional turbulence. We solve the
# two-dimensional vorticity equation with stochastic excitation and dissipation in
# the form of linear drag and hyperviscosity. As a demonstration, we compute how
# each of the forcing and dissipation terms contribute to the energy budget.

using FourierFlows, Printf, Plots

using FourierFlows: parsevalsum
using Random: seed!
using FFTW: irfft

import GeophysicalFlows.TwoDNavierStokes
import GeophysicalFlows.TwoDNavierStokes: energy, enstrophy, dissipation, work, drag
import GeophysicalFlows.TwoDNavierStokes: dissipation_ens, work_ens, drag_ens


# ## Choosing a device: CPU or GPU

dev = CPU()    # Device (CPU/GPU)
nothing # hide


# ## Numerical, domain, and simulation parameters
#
# First, we pick some numerical and physical parameters for our model.

 n, L  = 256, 2π             # grid resolution and domain length
 ν, nν = 2e-7, 2             # hyperviscosity coefficient and order
 μ, nμ = 1e-1, 0             # linear drag coefficient
dt, tf = 0.005, 0.2/μ        # timestep and final time
    nt = round(Int, tf/dt)   # total timesteps
    ns = 4                   # how many intermediate times we want to plot
nothing # hide


# ## Forcing

# We force the vorticity equation with stochastic excitation that is delta-correlated
# in time and while spatially homogeneously and isotropically correlated. The forcing
# has a spectrum with power in a ring in wavenumber space of radious $k_f$ and
# width $\delta k_f$, and it injects energy per unit area and per unit time equal
# to $\varepsilon$.

forcing_wavenumber = 14.0    # the central forcing wavenumber for a spectrum that is a ring in wavenumber space
forcing_bandwidth  = 1.5     # the width of the forcing spectrum
ε = 0.001                    # energy input rate by the forcing

gr   = TwoDGrid(dev, n, L)
x, y = gr.x, gr.y

forcing_spectrum = @. exp(-(sqrt(gr.Krsq)-forcing_wavenumber)^2/(2*forcing_bandwidth^2))
forcing_spectrum[ gr.Krsq .< (2π/L*2)^2 ]  .= 0 # make sure that focing has no power for low wavenumbers
forcing_spectrum[ gr.Krsq .> (2π/L*20)^2 ] .= 0 # make sure that focing has no power for high wavenumbers
ε0 = parsevalsum(forcing_spectrum.*gr.invKrsq/2.0, gr)/(gr.Lx*gr.Ly)
forcing_spectrum .= ε/ε0 * forcing_spectrum # normalize forcing to inject energy ε

seed!(1234)
nothing # hide

# Next we construct function `calcF!` that computes a forcing realization every timestep
function calcF!(Fh, sol, t, cl, v, p, g)
  ξ = ArrayType(dev)(exp.(2π*im*rand(eltype(gr), size(sol)))/sqrt(cl.dt))
  ξ[1, 1] = 0
  @. Fh = ξ*sqrt(forcing_spectrum)
  nothing
end
nothing # hide


# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments. The
# `stepper` keyword defines the time-stepper to be used.
prob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, nμ=nμ, dt=dt, stepper="ETDRK4",
                                calcF=calcF!, stochastic=true)
nothing # hide

# Define some shortcuts for convenience.
sol, cl, v, p, g = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
nothing # hide


# First let's see how a forcing realization looks like.
calcF!(v.Fh, sol, 0.0, cl, v, p, g)

heatmap(x, y, irfft(v.Fh, g.nx),
     aspectratio = 1,
               c = :balance,
            clim = (-200, 200),
           xlims = (-L/2, L/2),
           ylims = (-L/2, L/2),
          xticks = -3:3,
          yticks = -3:3,
          xlabel = "x",
          ylabel = "y",
           title = "a forcing realization",
      framestyle = :box)


# ## Setting initial conditions

# Our initial condition is simply fluid at rest.
TwoDNavierStokes.set_zeta!(prob, zeros(g.nx, g.ny))


# ## Diagnostics

# Create Diagnostics; the diagnostics are aimed to probe the energy budget.
E = Diagnostic(energy,      prob, nsteps=nt) # energy
R = Diagnostic(drag,        prob, nsteps=nt) # dissipation by drag
D = Diagnostic(dissipation, prob, nsteps=nt) # dissipation by hyperviscosity
W = Diagnostic(work,        prob, nsteps=nt) # work input by forcing
Z = Diagnostic(enstrophy,      prob, nsteps=nt) # energy
RZ = Diagnostic(drag_ens,        prob, nsteps=nt) # dissipation by drag
DZ = Diagnostic(dissipation_ens, prob, nsteps=nt) # dissipation by hyperviscosity
WZ = Diagnostic(work_ens,        prob, nsteps=nt) # work input by forcing
diags = [E, D, W, R, Z, DZ, WZ, RZ] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hide


# ## Visualizing the simulation

# We define a function that plots the vorticity field and the evolution of
# the diagnostics: energy and all terms involved in the energy budget. Last
# we confirm whether the energy budget is accurate, i.e., $\mathrm{d}E/\mathrm{d}t = W - R - D$.

function computetendencies_and_makeplot(prob, diags)
  TwoDNavierStokes.updatevars!(prob)
  E, D, W, R, Z, DZ, WZ, RZ = diags

  clocktime = round(μ*cl.t, digits=2)

  i₀ = 1
  dEdt_numerical = (E[(i₀+1):E.i] - E[i₀:E.i-1])/cl.dt #numerical first-order approximation of energy tendency
  dZdt_numerical = (Z[(i₀+1):Z.i] - Z[i₀:Z.i-1])/cl.dt #numerical first-order approximation of enstrophy tendency
  ii = (i₀):E.i-1
  ii2 = (i₀+1):E.i

  t = E.t[ii]
  dEdt_computed = W[ii2] - D[ii] - R[ii]        # Stratonovich interpretation
  dZdt_computed = WZ[ii2] - DZ[ii] - RZ[ii]

  residual_E = dEdt_computed - dEdt_numerical
  residual_Z = dZdt_computed - dZdt_numerical

  εZ = ε*(forcing_wavenumber^2)  # Estimate of mean enstrophy injection rate

  l = @layout grid(2,3)

  pzeta = heatmap(x, y, v.zeta,
            aspectratio = 1,
            legend = false,
                 c = :viridis,
              clim = (-25, 25),
             xlims = (-L/2, L/2),
             ylims = (-L/2, L/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "μt",
            ylabel = "y",
             title = "∇²ψ(x, y, t="*@sprintf("%.2f", cl.t)*")",
        framestyle = :box)

  pζ = plot(pzeta, size = (400, 400))

  p1 = plot(μ*t, [W[ii2] ε.+0*t -D[ii] -R[ii]],
             label = ["work, W" "ensemble mean work, <W>" "dissipation, D" "drag, D=-2μE"],
         linestyle = [:solid :dash :solid :solid],
         linewidth = 2,
             alpha = 0.8,
            xlabel = "μt",
            ylabel = "energy sources and sinks")

  p2 = plot(μ*t, [dEdt_computed[ii], dEdt_numerical],
           label = ["computed W-D" "numerical dE/dt"],
       linestyle = [:solid :dashdotdot],
       linewidth = 2,
           alpha = 0.8,
          xlabel = "μt",
          ylabel = "dE/dt")

  p3 = plot(μ*t, residual_E,
           label = "residual dE/dt = computed - numerical",
       linewidth = 2,
           alpha = 0.7,
          xlabel = "μt")

  p4 = plot(μ*t, [WZ[ii2] εZ.+0*t -DZ[ii] -RZ[ii]],
           label = ["Enstrophy work, WZ" "mean enstrophy work, <WZ>" "enstrophy dissipation, DZ" "enstrophy drag, D=-2μZ"],
       linestyle = [:solid :dash :solid :solid],
       linewidth = 2,
           alpha = 0.8,
          xlabel = "μt",
          ylabel = "enstrophy sources and sinks")


  p5 = plot(μ*t, [dZdt_computed[ii], dZdt_numerical],
         label = ["computed WZ-DZ" "numerical dZ/dt"],
     linestyle = [:solid :dashdotdot],
     linewidth = 2,
         alpha = 0.8,
        xlabel = "μt",
        ylabel = "dZ/dt")

  p6 = plot(μ*t, residual_Z,
         label = "residual dZ/dt = computed - numerical",
     linewidth = 2,
         alpha = 0.7,
        xlabel = "μt")


  pbudgets = plot(p1, p2, p3, p4, p5, p6, layout=l, size = (1300, 900))

  return pζ, pbudgets
end
nothing # hide




# ## Time-stepping the `Problem` forward

# Finally, we time-step the `Problem` forward in time.

startwalltime = time()
for i = 1:ns
  stepforward!(prob, diags, round(Int, nt/ns))
  TwoDNavierStokes.updatevars!(prob)
  cfl = cl.dt*maximum([maximum(v.u)/g.dx, maximum(v.v)/g.dy])

  log = @sprintf("step: %04d, t: %.1f, cfl: %.3f, walltime: %.2f min", cl.step, cl.t,
        cfl, (time()-startwalltime)/60)

  println(log)
end


# ## Plot
# And now let's see what we got. We plot the output.

pζ, pbudgets = computetendencies_and_makeplot(prob, diags)
