module CompGrids

using ParallelStencil
using ImplicitGlobalGrid
using MPI
using OffsetArrays

@init_parallel_stencil(Threads, Float64, 3);

export backend,
       BackendPETSc,  BackendParallelStencil, BackendNone,
       
       #
       initialize_backend, initialize_grid,
       
       # Boundary topology

       # Grids
       RegularRectilinearCollocatedGrid, Bounded, Ghosted, Periodic



# Declare the backend we are employing:
include("Backends.jl")
include("PETSc_backend.jl")
include("ParallelStencil_backend.jl")
include("Grids.jl")

# Declare the backend we are employing:


# Define different grid types





end # module
