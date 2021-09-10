# routines related to the PETSc backend

export initialize_backend, petsc_data

"""
    Stores key PETSc information with the `grid` structure
"""
struct petsc_data
    petsclib
    da
end
petsc_data() = petsc_data(nothing,nothing)

"""
    check_backend(backend{BackendPETSc}; Scalar=Float64, dim=1)

checks PETSc backend for the `Scalar`, which loads PETSc (and, if requested, MPI)
"""
function check_backend(b::Backend{BackendPETSc}; dim=1, Scalar=Float64)
    
    if !isdefined(Main, :PETSc)
        error("PETSc is not loaded; ensure it is loaded first with: using PETSc")
    end
    if (b.mpi==true && !isdefined(Main, :MPI))
        error("MPI is not loaded; ensure it is loaded first with: using MPI")
    end
    @eval using PETSc
  
    # get the PETSc lib with our chosen `PetscScalar` type
    petsclib = PETSc.getlib(; PetscScalar = Scalar)

    # Initialize PETSc
    PETSc.initialize(petsclib)

    return petsclib
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



function globalfromlocalsize(size, localsize, opts, stencilwidth, b::Backend{BackendPETSc, FT}) where FT

    if !b.mpi & !isempty(localsize)
        size = localsize   # not using MPI
    elseif b.mpi & !isempty(localsize)
        error("The PETSc backend currently doesn't allow specifying the local grid size")
    else
        size=size
    end

    return size
end