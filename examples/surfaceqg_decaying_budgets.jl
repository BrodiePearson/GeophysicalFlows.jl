# # Decaying Surface QG turbulence
#
# A simulation of decaying surface quasi-geostrophic turbulence.
# The dynamics include an initial stochastic excitation and small-scale
# hyper-viscous dissipation.

using FourierFlows, Plots, Statistics, Printf, Random

using FourierFlows: parsevalsum
using FFTW: irfft, rfft
using Statistics: mean
using Random: seed!

import GeophysicalFlows.SurfaceQG
import GeophysicalFlows.SurfaceQG: kinetic_energy, buoyancy_variance, buoyancy_dissipation
import GeophysicalFlows.SurfaceQG: kinetic_energy_advection, buoyancy_advection


# ## Choosing a device: CPU or GPU

dev = CPU()    # Device (CPU/GPU)ENV["GRDIR"]=""
nothing # hide


# ## Numerical parameters and time-stepping parameters

     nx = 512            # 2D resolution = nx^2
stepper = "FilteredRK4"  # timestepper
     dt = 0.005          # timestep
     tf = 10             # length of time for simulation
 nsteps = tf/dt           # total number of time-steps
 nsubs  = round(Int, nsteps/tf)         # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)
nothing # hide


# ## Physical parameters

  L = 2π        # domain size
 nν = 4
  ν = 1e-19     # 1e-19
nothing # hide


# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments. Not providing
# a viscosity coefficient ν leads to the module's default value: ν=0. In this
# example numerical instability due to accumulation of enstrophy in high wavenumbers
# is taken care with the `FilteredTimestepper` we picked.
prob = SurfaceQG.Problem(dev; nx=nx, Lx=L, dt=dt, stepper=stepper,
                            ν=ν, nν=nν, stochastic=true)
nothing # hide

# Let's define some shortcuts.
sol, cl, vs, pr, gr = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = gr.x, gr.y
nothing # hide


# ## Setting initial conditions
#
# We initialize the buoyancy equation with stochastic excitation that is delta-correlated
# in time, and homogeneously and isotropically correlated in space. The forcing
# has a spectrum with power in a ring in wavenumber space of radius kᵖ and
# width δkᵖ, and it injects energy per unit area and per unit time equalto ϵ.

gr  = TwoDGrid(nx, L)

init_b = exp.(-((repeat(gr.x', nx)').^2 + (4*repeat(gr.y', nx)).^2))

seed!(1234) # reset of the random number generator for reproducibility
nothing # hide

# Our initial condition is simply fluid at rest.
SurfaceQG.set_b!(prob, init_b)


# ## Diagnostics

# Create Diagnostic -- `energy` and `enstrophy` are functions imported at the top.
b² = Diagnostic(buoyancy_variance, prob; nsteps=nsteps)
KE = Diagnostic(kinetic_energy, prob; nsteps=nsteps)
Dᵇ = Diagnostic(buoyancy_dissipation, prob; nsteps=nsteps)
Aᵇ = Diagnostic(buoyancy_advection, prob; nsteps=nsteps)
Aᵏ = Diagnostic(kinetic_energy_advection, prob; nsteps=nsteps)
diags = [b², KE, Dᵇ, Aᵇ, Aᵏ] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hidenothing # hide


# ## Output

# We choose folder for outputing `.jld2` files and snapshots (`.png` files).
# Define base filename so saved data can be distinguished from other runs
base_filename = string("SurfaceQG_decaying_n_", nx, "_visc_", round(ν, sigdigits=1), "_order_", 2*nν)
# We choose folder for outputing `.jld2` files and snapshots (`.png` files).
datapath = "./"
plotpath = "./"

dataname = joinpath(datapath, base_filename)
plotname = joinpath(plotpath, base_filename)
nothing # hide

# Do some basic file management,
if !isdir(plotpath); mkdir(plotpath); end
if !isdir(datapath); mkdir(datapath); end
nothing # hide

# and then create Output.
get_sol(prob) = sol # extracts the Fourier-transformed solution
get_u(prob) = irfft(im*gr.l.*sqrt.(gr.invKrsq).*sol, gr.nx)
out = Output(prob, dataname, (:sol, get_sol), (:u, get_u))
nothing # hide


# ## Visualizing the simulation

# We define a function that plots the buoyancy field and the time evolution of
# kinetic energy and buoyancy variance.

function computetendencies_and_makeplot(prob, diags)
  SurfaceQG.updatevars!(prob)
  b², KE, Dᵇ, Aᵇ, Aᵏ = diags

  clocktime = round(cl.t, digits=2)

  i₀ = 1
  dKEdt_numerical = (KE[(i₀+1):KE.i] - KE[i₀:KE.i-1])/cl.dt #numerical first-order approximation of energy tendency
  db²dt_numerical = (b²[(i₀+1):b².i] - b²[i₀:b².i-1])/cl.dt #numerical first-order approximation of enstrophy tendency
  ii = (i₀):KE.i-1
  ii2 = (i₀+1):KE.i

  t = KE.t[ii]   # Dissipation rate of kinetic energy is half that of b²
  dKEdt_computed = - 0.5*Dᵇ[ii] #+ Aᵏ[ii]        # Stratonovich interpretation
  db²dt_computed = - Dᵇ[ii] #+ Aᵇ[ii]

  residual_KE = dKEdt_computed - dKEdt_numerical
  residual_b² = db²dt_computed - db²dt_numerical

  pbuoy = heatmap(x, y, vs.b,
            aspectratio = 1,
            legend = false,
                 c = :viridis,
#              clim = (-25, 25),
             xlims = (-L/2, L/2),
             ylims = (-L/2, L/2),
            xticks = -3:3,
            yticks = -3:3,
            xlabel = "t",
            ylabel = "y",
             title = "bˢ(x, y, t="*@sprintf("%.2f", cl.t)*")",
        framestyle = :box)

  pb = plot(pbuoy, size = (400, 400))

  l = @layout grid(2,3)

  p1 = plot(t, [-0.5*Dᵇ[ii] Aᵏ[ii]],
             label = ["dissipation, Dᵏ" "Advection, Aᵏ"],
         linestyle = [:solid :solid],
         linewidth = 2,
             alpha = 0.8,
            xlabel = "t",
            ylabel = "kinetic energy sinks")

  p2 = plot(t, [dKEdt_computed[ii], dKEdt_numerical],
           label = ["computed dKE/dt" "numerical dKE/dt"],
       linestyle = [:solid :dashdotdot],
       linewidth = 2,
           alpha = 0.8,
          xlabel = "t",
          ylabel = "dKE/dt")

  p3 = plot(t, residual_KE,
           label = "residual dKE/dt = computed - numerical",
       linewidth = 2,
           alpha = 0.7,
          xlabel = "t")

  p4 = plot(t, [-Dᵇ[ii] Aᵇ[ii]],
           label = ["buoyancy dissipation, Dᵇ" "buoyancy advection, Aᵇ"],
       linestyle = [:solid :solid],
       linewidth = 2,
           alpha = 0.8,
          xlabel = "t",
          ylabel = "buoyancy variance sinks")


  p5 = plot(t, [db²dt_computed[ii], db²dt_numerical],
         label = ["computed db²/dt" "numerical db²/dt"],
     linestyle = [:solid :dashdotdot],
     linewidth = 2,
         alpha = 0.8,
        xlabel = "t",
        ylabel = "db²/dt")

  p6 = plot(t, residual_b²,
         label = "residual db²/dt = computed - numerical",
     linewidth = 2,
         alpha = 0.7,
        xlabel = "t")


  pbudgets = plot(p1, p2, p3, p4, p5, p6, layout=l, size = (1300, 900))

  return pb, pbudgets
end
nothing # hide


# ## Time-stepping the `Problem` forward and create animation by updating plot

startwalltime = time()
for i = 1:Int(nsteps/nsubs)
  stepforward!(prob, diags, nsubs)
  SurfaceQG.updatevars!(prob)
  cfl = cl.dt*maximum([maximum(vs.u)/gr.dx, maximum(vs.v)/gr.dy])

  log = @sprintf("step: %04d, t: %.1f, cfl: %.3f, walltime: %.2f min", cl.step, cl.t,
        cfl, (time()-startwalltime)/60)

  println(log)

  log = @sprintf("buoyancy variance diagnostics - b²: %.2e, Diss: %.2e, b² Adv: %.2e, KE Adv: %.2e",
            b².data[b².i], Dᵇ.data[Dᵇ.i], Aᵇ.data[Aᵇ.i], Aᵏ.data[Aᵏ.i])

      println(log)
end

pb, pbudgets = computetendencies_and_makeplot(prob, diags)

pbudgets
png("SQG_Budgets_small_viscosity")

# Last we save the output.
saveoutput(out)
