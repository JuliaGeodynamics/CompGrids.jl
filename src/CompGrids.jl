module CompGrids


export backend,
       BackendPETSc,  BackendParallelStencil, BackendNone,
       initialize_backend, initialize_grid,
       RegularRectilinearCollocatedGrid, Bounded, Flat2D, Ghosted, Periodic


# Declare the backend we are employing:
include("Backends.jl")
include("PETSc_backend.jl")
include("Grids.jl")

# Declare the backend we are employing:

# Define different grid types





end # module
