# routines related to the ParallelStencil backend

export initialize_backend


"""
    start_backend(Backend{BackendParallelStencil}; Scalar=Float64, dim=1)

Starts the `ParallelStencil` backend for the `Scalar`, which loads ParallelStencil (and, if requested, MPI) for the correct number of dimensions
"""
function check_backend(b::Backend{BackendParallelStencil}; dim=1, Scalar=Float64)
    # Check whether MPI is loaded
    if (b.mpi==true && !isdefined(Main, :ImplicitGlobalGrid))
        error("ImplicitGlobalGrid is not loaded; ensure it is loaded first with: using ImplicitGlobalGrid")
    end
    if (b.mpi==true && !isdefined(Main, :MPI))
        error("MPI is not loaded; ensure it is loaded first with: using MPI")
    end

    # Check whether ParallelStencil is loaded
    if !isdefined(Main, :ParallelStencil)
        error("ParallelStencil is not loaded; ensure it is loaded first with: using ParallelStencil")
    end

    # Load GPU or CPU backend
    if ((b.arch==:GPU) || (b.arch==:CUDA)) 
        # Currently, ParallelStencil, only supports CUDA
        if !isdefined(Main, :CUDA)
            error("CUDA is not loaded; ensure it is loaded first with: using CUDA")
        end
    end


    return nothing
end
