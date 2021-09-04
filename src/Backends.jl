"""
    This implements the backends
"""

import Base: show

export  backend, Backend,
        # Possibiities 
        BackendPETSc,  BackendParallelStencil, BackendNone, @init_backend,
        
        # MPI, flobal variables
        comm, mpirank, mpisize

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
    Backend(type::Symbol, arch::Symbol, mpi::Bool) 

Structure that defines the backend used for the computional grid  

- `type`: `:ParallelStencil`, `:PETSc` (requires loading the corresponding packages), or `:Julia` (julia native grid)
- `arch`: `:CPU`,`:GPU`,`:CUDA` (requires `CUDA` to be loaded first)
- `mpi`: Boolean that indicates if we employ MPI 


"""
struct Backend{T} <: AbstractBackend{T}
    type   :: Symbol
    arch   :: Symbol
    mpi    :: Bool
    Scalar :: DataType 
end

"""
    Backend(; type::Symbol=:PETSc, arch::Symbol=:CPU, mpi::Bool=true) 

Specify the backend we are using as well as whether the simulation is done on an MPI parallel machine or not

- `type`: `:ParallelStencil`, `:PETSc` (requires loading the corresponding packages), or `:Julia` (julia native)
- `arch`: `:CPU`,`:GPU`,`:CUDA` 
- `mpi`: Boolean that indicates if we employ MPI or not. Note that PETSc always requires MPI 
- `dim`: Number of dimensions (only relevant for `ParallelStencil`)

"""
function Backend(; arch::Symbol=:CPU, type::Symbol=:PETSc, mpi::Bool=true)
    global backend

    @show type, typeof(type)

    # Determine the main backend we are using (PETSc/ParallelStencil or None)
    if type == :PETSc
        backend_type = BackendPETSc;
    elseif type == :ParallelStencil
        backend_type = BackendParallelStencil;
    else
        backend_type = BackendNone;
    end

    return Backend{backend_type}(type, arch, mpi)
end


# Printing 
function show(io::IO, b::Backend)
    if b.mpi
        println(io, "Backend{$(b.Scalar)}: $(b.type) ($(b.arch) | MPI)")
    else
        println(io, "Backend{$(b.Scalar)}: $(b.type) ($(b.arch) | no MPI)")
    end
end





"""
    @init_backend(type, arch, mpi=false, Scalar=Float64) 

Specify the backend we are using as well as whether the simulation is done on an MPI parallel machine or not

- `type`: `:ParallelStencil`, `:PETSc` (requires loading the corresponding packages), or `:Julia` (julia native)
- `arch`: `:CPU`,`:GPU`,`:CUDA` 
- `mpi`: Boolean that indicates if we employ MPI or not. Note that PETSc always requires MPI 
- `Scalar`: Type of Scalar

"""
macro init_backend(type, arch, mpi=false, Scalar=Float64)
    global backend, comm, mpirank, mpisize

    #backend = Backend(arch=arch, type=type, mpi=mpi); 

    mpirank = 0
    mpisize = 1
    comm    = 0

    if mpi 
        # Initialize MPI & make 
        if isdefined(Main, :MPI)
            comm = MPI.COMM_WORLD
            if !MPI.Initialized()
                MPI.Init()
            end
            mpirank = MPI.Comm_rank(MPI.COMM_WORLD)
            mpisize = MPI.Comm_size(MPI.COMM_WORLD)
        else
            error("Load MPI package first with using MPI")
        end
    end

    #
    if type == :PETSc
        backend_type = BackendPETSc;
        type_sym = :PETSc
    elseif type == :ParallelStencil
        backend_type = BackendParallelStencil;
        type_sym = :ParallelStencil
    else
        backend_type = BackendNone;
        type_sym = :Julia
    end

    if arch == :CUDA
        arch_sym = :CUDA
    elseif arch == :Threads
        arch_sym = :Threads
    elseif arch == :Base
        arch_sym = :CPU
    else
        arch_sym = :CPU
    end

    backend = Backend{backend_type}(type_sym, arch_sym, mpi, Scalar)

end