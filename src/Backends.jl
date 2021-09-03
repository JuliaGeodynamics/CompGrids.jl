"""
    This implements the backends
"""

import Base: show

export  backend,
        # Possibiities 
        BackendPETSc,  BackendParallelStencil, BackendNone

"""
    AbstractBackend{type} 

Abstract supertype that specifies the backend of the simulation (PETSc.jl, ParallelStencil, None) as well as how 
the simulation is being setup (MPI or not)
"""
abstract type AbstractBackend{type} end

"""
    AbstractTypeBackend

Abstract supertype that determines the types of the backends that are implemented
"""
abstract type AbstractTypeBackend end

"""
    BackendPETSc

Use PETSc as a backend 
"""
abstract type BackendPETSc <: AbstractTypeBackend end

"""
    BackendParallelStencil

Use ParallelStencil as a backend 
"""
abstract type BackendParallelStencil <: AbstractTypeBackend end


"""
    BackendNone

No backend; julia native only. Note that this is not compatible with MPI
"""
abstract type BackendNone <: AbstractTypeBackend end




"""
    backend(type::Symbol, arch::Symbol, mpi::Bool) 

Structure that defines the backend used for the computional grid  

- `type`: `:ParallelStencil`, `:PETSc` (requires loading the corresponding packages), or `:Julia` (julia native grid)
- `arch`: `:CPU`,`:GPU`,`:CUDA` (requires `CUDA` to be loaded first)
- `mpi`: Boolean that indicates if we employ MPI 


"""
struct backend{T} <: AbstractBackend{T}
    type :: Symbol
    arch :: Symbol
    mpi  :: Bool
end

"""
    backend(; type::Symbol=:PETSc, arch::Symbol=:CPU, mpi::Bool=true) 

Specify the backend we are using as well as whether the simulation is done on an MPI parallel machine or not

- `type`: `:ParallelStencil`, `:PETSc` (requires loading the corresponding packages), or `:Julia` (julia native)
- `arch`: `:CPU`,`:GPU`,`:CUDA` 
- `mpi`: Boolean that indicates if we employ MPI 

"""
function backend(; arch::Symbol=:CPU, type::Symbol=:PETSc, mpi::Bool=true) 
    
    # Determine the main backend we are using (PETSc/ParallelStencil or None)
    if type == :PETSc
        backend_type = BackendPETSc;
    elseif type == :ParallelStencil
        backend_type = BackendParallelStencil;
    else
        backend_type = BackendNone;
    end

    return backend{backend_type}(type, arch, mpi)
end


# Printing 
function show(io::IO, b::backend)
    if b.mpi
        println(io, "Backend: $(b.type) ($(b.arch) | MPI)")
    else
        println(io, "Backend: $(b.type) ($(b.arch) | no MPI)")
    end
end


