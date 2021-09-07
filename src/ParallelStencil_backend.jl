# routines related to the ParallelStencil backend

export initialize_backend!


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


function initialize_backend!(grid, b::Backend{BackendParallelStencil}, stencilwidth, opts, fields::NamedTuple)

    N_vec = ones(Int,3);
    D = grid.dim

    Nl,dims = localgridsize(backend.mpi, grid.N, opts)      # local size using MPI (w/out halo)
    N_vec[1:D] = collect(Nl)
    
    initmpi=false
    if !MPI.Initialized()
        initmpi=true
    end

    # Halo
    overlap  = zeros(Int64,3)
    overlap[1:D] .= 2*stencilwidth
    Nl_halo  = N_vec .+ overlap

    # Periodic?
    period  = zeros(Int64,3)
    period[findall(grid.topology.==Periodic)] .= 1

    IGG_data = init_global_grid(Nl_halo[1], Nl_halo[2], Nl_halo[3], 
                                                            init_MPI=initmpi, 
                                                            quiet=false, 
                                                            overlapx=overlap[1], overlapy=overlap[2], overlapz=overlap[3],
                                                            periodx=period[1], periody=period[2], periodz=period[3];
                                                            opts...);
    
    IGG_data = ImplicitGlobalGrid.get_global_grid()     
    grid.IGG = IGG_data;                    # store

    # Store local grid dimensions
    grid.Nl = (Nl_halo[1:D]...,)            # local with halo
    grid.Ng = (IGG_data.nxyz_g[1:D]...,)    # global + halo

    # Compute local indices of grid in global grid (not including the halo)
    ind_s = IGG_data.coords.*(IGG_data.nxyz .- IGG_data.overlaps) 
    ind_e = ind_s[1:D] .+ Nl;
    ind_local = ();
    for i=1:D
        ind_local = (ind_local..., ind_s[i]+1:ind_e[i])
    end
    grid.ind_local = ind_local

    # Update the 1D coordinate vectors 
    local_coordinates!(grid, grid.backend)

    return nothing

end

"""
    initialize_fields!(grid, b::Backend{BackendParallelStencil}, fields::NamedTuple)

This initializes fields to a grid structure, in case the `ParallelStencil` is employed. 
The names of the fields are a NamedTuple, which specifies the initial value (constant) value of the arrays, such as `fields=(T=0, P=1, Ci=2)`.

Note: typically, this routine is not called directly but as part of setting up the grid

# Example

```
julia> @init_backend(ParallelStencil, Threads, false, Float64);
julia> initialize_fields!(grid, b::Backend{BackendParallelStencil}, fields::NTuple)
```

"""
function initialize_fields!(grid, b::Backend{BackendParallelStencil, FT}, fields::NamedTuple) where FT

    names = keys(fields)
    for ifield=1:length(fields)

        # Created named tuple for new field. We use @eval here, to avoid 
        name = names[ifield]
        value = fields[ifield]
        
        # Create a string with the expression to be evaluated (as calling @ones directly induces a compilation error)
        str = "new_field = @ones$(grid.Nl)*$value"      
        eval(Meta.parse(str))                       # evaluate string
        
        # Add to Tuple        
        grid.fields = add_field(grid.fields,name,new_field)
    end
end
