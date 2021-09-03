# routines related to the PETSc backend

export initialize_backend, petsc_data

"""
    Stores key PETSc information with the `grid` structure
"""
struct petsc_data
    petsclib
    da
end

"""
    start_backend(backend{BackendPETSc}; Scalar=Float64, dim=1)

Starts the PETSc backend for the `Scalar`, which loads PETSc (and, if requested, MPI)
"""
function initialize_backend(b::backend{BackendPETSc}; dim=1, Scalar=Float64)
    
    if !isdefined(Main, :PETSc)
        error("PETSc is not loaded; ensure it is loaded first with: using PETSc")
    else
        @eval using PETSc
    end
    if (b.mpi==true && !isdefined(Main, :MPI))
        error("MPI is not loaded; ensure it is loaded first with: using MPI")
    else
        @eval using MPI
        
    end
    if b.mpi==true
        # Set our MPI communicator
        comm = MPI.COMM_WORLD
    else
        comm = nothing
    end

    # get the PETSc lib with our chosen `PetscScalar` type
    petsclib = PETSc.getlib(; PetscScalar = Scalar)

    # Initialize PETSc
    PETSc.initialize(petsclib)

    return petsclib, comm
end

# Internal function that translates the CompGrids topology to PETSc nomenclature
function bcs_translate(bound)
    bound_petsc = PETSc.LibPETSc.DM_BOUNDARY_NONE
    if bound == Ghosted
        bound_petsc = PETSc.LibPETSc.DM_BOUNDARY_GHOSTED
    elseif bound == Periodic
        bound_petsc = PETSc.DM_BOUNDARY_PERIODIC
    end
    return bound_petsc
end