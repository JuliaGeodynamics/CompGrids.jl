using CompGrids
const PS_MPI = true
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using MPI
    
@static if PS_MPI    # using MPI-parallel setup
    using ImplicitGlobalGrid

    @init_backend(ParallelStencil, Threads, true, Float64);
    @init_parallel_stencil(Threads, Float64, 3);

else                # no MPI
    
    @init_parallel_stencil(Threads, Float64, 3);
    @init_backend(ParallelStencil, Threads, false, Float64);
end


@parallel function diffusion3D_step!(T2, T, Ci, lam, dt, dx, dy, dz)
    @inn(T2) = @inn(T) + dt*(lam*@inn(Ci)*(@d2_xi(T)/dx^2 + @d2_yi(T)/dy^2 + @d2_zi(T)/dz^2));
    return
end

function diffusion3D()
    
# Physics
lam        = 1.0;                                        # Thermal conductivity
cp_min     = 1.0;                                        # Minimal heat capacity

# Numerics
#opts=Dict(:dimy=>1,:dimx=>4)
@time grid       = RegularRectilinearCollocatedGrid(size=(64, 64, 64),  extent=(10.,10.,10.), fields=(T=0, Ci=1, T2=0))

Δ,L        = grid.Δ, grid.L                             # spacing & global grid size
nt         = 10;                                        # Number of time steps
if mpirank==0
    @show grid
end

# Set initial conditions on local portion of grid (heat capacity and temperature with two Gaussian anomalies each)
x,y,z      = grid.Face[1],grid.Face[2],grid.Face[3]
T,T2,Ci    = grid.fields[:T], grid.fields[:T2], grid.fields[:Ci]    # for convenience, create a link to the actual arrays
#@time T = @zeros(grid.Nl)
#T2 = @zeros(grid.Nl)
#Ci = @zeros(grid.Nl)

Ci .= 1.0./( cp_min .+ Data.Array([5*exp(-(( x[ix]-L[1]/1.5))^2-((y[iy]-L[2]/2))^2-((z[iz]-L[3]/1.5))^2) +
                                   5*exp(-(( x[ix]-L[1]/3.0))^2-((y[iy]-L[2]/2))^2-((z[iz]-L[3]/1.5))^2) for ix=1:grid.Nl[1], iy=1:grid.Nl[2], iz=1:grid.Nl[3]]) )
T  .= Data.Array([100*exp(-((x[ix]-L[1]/2)/2)^2-((y[iy]-L[2]/2)/2)^2-((z[iz]-L[3]/3.0)/2)^2) +
                   50*exp(-((x[ix]-L[1]/2)/2)^2-((y[iy]-L[2]/2)/2)^2-((z[iz]-L[3]/1.5)/2)^2) for ix=1:grid.Nl[1], iy=1:grid.Nl[2], iz=1:grid.Nl[3]])
#T2.= grid.fields[:T];                                               # Assign also T2 to get correct boundary conditions.


# Time loop
dt = minimum(Δ.^2)*cp_min/lam/8.1;                 # Time step for the 3D Heat diffusion
for it = 1:nt
    @parallel diffusion3D_step!(T2, T, Ci, lam, dt, Δ[1], Δ[2], Δ[3]);
    T, T2 = T2, T;
    
    if (backend.mpi) update_halo!(T, T2) end
    if (mpirank==0) println(it) end
end

if (backend.mpi) finalize_global_grid(finalize_MPI=false) end

end

diffusion3D()