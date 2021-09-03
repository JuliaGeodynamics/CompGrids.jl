# routines related to the ParallelStencil backend

export initialize_backend


"""
    start_backend(backend{BackendParallelStencil}; Scalar=Float64, dim=1)

Starts the `ParallelStencil` backend for the `Scalar`, which loads ParallelStencil (and, if requested, MPI) for the correct number of dimensions
"""
function initialize_backend(b::backend{BackendParallelStencil}; dim=1, Scalar=Float64)
    # Load MPI if needed
    #if (b.mpi==true && !isdefined(Main, :MPI))
    #    error("MPI is not loaded; ensure it is loaded first with: using MPI")
    #else
    #    @eval using MPI 
    #end
    if (b.mpi==true && !isdefined(Main, :ImplicitGlobalGrid))
        error("ImplicitGlobalGrid is not loaded; ensure it is loaded first with: using ImplicitGlobalGrid")
    else
        @eval using ImplicitGlobalGrid 
    end

    # Load ParallelStencil 
    if !isdefined(Main, :ParallelStencil)
        error("ParallelStencil is not loaded; ensure it is loaded first with: using ParallelStencil")
    else
        @eval using ParallelStencil
        
        if dim==1
            @eval using ParallelStencil.FiniteDifferences1D
        elseif dim==2
            @eval using ParallelStencil.FiniteDifferences2D
        elseif dim==3
            @eval using ParallelStencil.FiniteDifferences3D
        end

    end

    # Load GPU or CPU backend
    if (b.arch==:CPU)
        if dim==1
            @eval @init_parallel_stencil(Threads, Float64, 1);
        elseif dim==2
            @eval @init_parallel_stencil(Threads, Float64, 2);
        elseif dim==3
            @eval @init_parallel_stencil(Threads, Float64, 3);
        end
    elseif ((b.arch==:GPU) || (b.arch==:CUDA)) 
        # Currently, ParallelStencil, only supports CUDA
        if !isdefined(Main, :CUDA)
            error("CUDA is not loaded; ensure it is loaded first with: using CUDA")
        end
        if dim==1
            @eval @init_parallel_stencil(CUDA, Float64, 1);
        elseif dim==2
            @eval @init_parallel_stencil(CUDA, Float64, 2);
        elseif dim==3
            @eval @init_parallel_stencil(CUDA, Float64, 3);
        end
    end



    return nothing
end
