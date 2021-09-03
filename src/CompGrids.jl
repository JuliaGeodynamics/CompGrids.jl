module CompGrids


export backend,
       BackendPETSc,  BackendParallelStencil, BackendNone,
       
       #
       initialize_backend, initialize_grid,
       
       # Boundary topology

       # Grids
       RegularRectilinearCollocatedGrid, Bounded, Flat2D, Ghosted, Periodic



# Declare the backend we are employing:
include("Backends.jl")
include("PETSc_backend.jl")
include("ParallelStencil_backend.jl")
include("Grids.jl")

# Declare the backend we are employing:

# Define different grid types





end # module
