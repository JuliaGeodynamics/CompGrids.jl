
"""
    RegularRectilinearCollocatedGrid{FT, TX, TY, TZ, R} <: AbstractRectilinearGrid{FT, TX, TY, TZ}

A rectilinear grid with collocated points and constant grid spacings `Δx`, `Δy`, and `Δz` between cell centers
and cell faces, elements of type `FT`, topology `{TX, TY, TZ}`, and coordinate ranges
of type `R`.
"""
# NOTE: we can later add info here about the local & global extent of the grid, the halo, the neigboring processors and the type of calculation (CPU/MPI-CPU/GPU)
mutable struct RegularRectilinearCollocatedGrid{FT, D, B} <: AbstractRectilinearCollocatedGrid{FT, D, B}
    # number of dimensions
    dim :: Int

    # Number of global grid points in (x,y,z) [is specified by user]
    N  :: NTuple{D,Int}
    
    # Actual number of global grid points in (x,y,z) (includes halo if using IGG)
    Ng :: NTuple{D,Int}

    # Local number of local grid points in (x,y,z) on everty processor (w/oout halo)
    Nl :: NTuple{D,Int}

    # Topology
    topology :: NTuple{D,DataType}
    
    # Stencilwidth
    stencilwidth :: Int

    # Domain size
    L  :: NTuple{D,FT}
    
    # Grid spacing 
    Δ  :: NTuple{D,FT}
    
    # Range of (local) coordinates at the centers of the cells. 
    Center  :: NTuple{D,StepRangeLen}
    
    # Range of (local) coordinates  (+halo in case of IGG) at the faces of the cells.
    Face  :: NTuple{D,StepRangeLen}

    # Range of (global) coordinates at the centers of the cells.
    Center_global  :: NTuple{D,StepRangeLen}
    
    # Range of (global) coordinates at the faces of the cells.
    Face_global  :: NTuple{D,StepRangeLen}
    
    # Indices of local portion of grid in x/y/z direction (w/out halo)
    ind_local :: NTuple{D,UnitRange{Int}}

    # Corners
    corners :: NamedTuple
    
    # Corners with ghost points
    ghostcorners :: NamedTuple

    # Backend employed
    backend :: B
    
    # store PETSc specific info (if used)
    PETSc   :: petsc_data 

    # store data in case we employ ImplicitGlobalGrid
    IGG                  

    # fields
    fields

end

"""
        RegularRectilinearCollocatedGrid(;
                            size=nothing,
                            local_size=nothing,
                            x = nothing, y = nothing, z = nothing,
                            topology= (Bounded, Bounded, Bounded),    
                             extent = nothing,
                             fields = NamedTuple(),
                        stencilwidth= 1, 
                         stenciltype= :Star,
                                opts= (),
                        )

Creates a `RegularRectilinearCollocatedGrid` with `size = (Nx, Ny, Nz)` grid cells. 
All variables are collocated, that is, placed at the same points

Keyword arguments
=================

- `size` : A tuple prescribing the number of grid points all directions.
                     `size` is a 3-tuple for 3D models, a 2-tuple for 2D models, and a 1-tuple for 1D.
- `local_size` : A tuple prescribing the number of grid points all directions.
                     `size` is a 3-tuple for 3D models, a 2-tuple for 2D models, and a 1-tuple for 1D.

- `extent`: A tuple prescribing the physical extent of the grid.
            The origin for three-dimensional domains is the lower left corner `(-Lx/2, -Ly/2, -Lz)`.

- `x`, `y`, and `z`: Each of `x, y, z` are 2-tuples that specify the end points of the domain
                     in their respect directions. 

- `topology`: A tuple prescribing the topology of the grid in the different directions. 
              This is also used to set the type of the `DMDA` boundary conditions in case of using the `PETSc` backend. 
              Examples are `Bounded`, `Ghost`, `Periodic`

- `fields`: A NamedTuple with the names of fields that are initialized using the correct backend usinng a specifiable, constant, value (e.g., `fields=(T=0, P=11.1)`)                  

*Note*: 
- _Either_ `extent`, or all of `x`, `y`, and `z` must be specified.
- _Either_ `size`, or `local_size` (the dimensions per processor; preferred by IGG) must be specified.


Optional PETSc arguments
========================

- `stencilwidth`: the width of the stencil                     

- `stenciltype`: the type of the stencil (`:Star` or `:Box`)                    

- `opts`: additional PETSc options                    

Optional ImplicitGlobalGrid arguments
=====================================

- `stencilwidth`=1: the width of the stencil (`overlapx`=`overlapy`=`overlapz`= `stencilwidth+1`)  

- `opts::Dict`: a dictionary that contains additional options that can be passed to IGG, such as `opts=Dict(:dimy=>2,:dimx=>1)`

The physical extent of the domain can be specified via `x`, `y`, and `z` keyword arguments
indicating the left and right endpoints of each dimensions, e.g. `x=(0, 10)` or via
the `extent` argument, e.g. `extent=(Lx, Ly, Lz)` which specifies the extent of each dimension
in which case -Lx/2 ≤ x ≤ Lx/2, -Ly/2 ≤ y ≤ Ly/2, and -Lz ≤ z ≤ 0.

Constants are stored using floating point values of type `FT`, which is the type specified in the backend structure

Grid properties
===============

- `(Nx, Ny, Nz)::Int`: Number of physical points in the (x, y, z)-direction

- `(Lx, Ly, Lz)::FT`: Physical extent of the grid in the (x, y, z)-direction

- `(Δx, Δy, Δz)::FT`: Center width in the (x, y, z)-direction

- `(xC, yC, zC)`: (x, y, z) coordinates of cell centers.

- `(xF, yF, zF)`: (x, y, z) coordinates of cell faces.

Examples
========

* A default grid with Float64 type:

```jldoctest
julia> using CompGrids, ParallelStencil
julia> @init_backend(ParallelStencil, Threads, false, Float64);
julia> grid = RegularRectilinearCollocatedGrid(size=(32, 32, 32), extent=(1, 2, 3))
RegularRectilinearCollocatedGrid{Float64, 3, Backend{BackendParallelStencil}}
        Backend: ParallelStencil ( Threads ) 
       gridtype: Collocated 
         domain: x ∈ [-0.5, 0.5], y ∈ [-1.0, 1.0], z ∈ [-3.0, 0.0]
       topology: (Bounded, Bounded, Bounded)
     resolution: (32, 32, 32)
 grid spacing Δ: (0.03125, 0.0625, 0.09375)
```


"""
function RegularRectilinearCollocatedGrid(;
                                    size=(),
                                    local_size=(),
                                     x = nothing, y = nothing, z = nothing,
                               topology= (Bounded, Bounded, Bounded),    
                                extent = nothing,
                                fields = NamedTuple(),
                           stencilwidth= 1, 
                            stenciltype= :Star,
                                   opts= ()
                              )
    FT = backend.Scalar;        # Scalar type
    
    # Checks
    if !isempty(local_size) & !isempty(size)
        error("You need to specify either size or local_size, but you cannot specify both at the same time!")
    end
    if isempty(local_size) & isempty(size)
        error("You need to specify either size or local_size")
    end
    
    # Create Tuples if needed
    if typeof(size)==Int64 
        size = (size,) 
    end
    if typeof(extent)==FT  
        extent = (extent,) 
    end
    
    size = globalfromlocalsize(size, local_size, opts, stencilwidth, topology, backend)   # Compute global size from local size


    dim =   length(size)                                  # dimensions of the grid [1-3]
    L, X₁ = validate_regular_grid_domain(FT, extent, x, y, z)

    # Unpacking
    N = size
    Δ = L ./ (N .- FT(1))
    
    # Face-node limits in x, y, z
    XF₋ = @. X₁ 
    XF₊ = @. XF₋ + L              

    # Center-node limits in x, y, z
    XC₋ = @. XF₋ + Δ / 2
    XC₊ = @. XC₋ + L - Δ

    # Generate 1D coordinate arrays of faces & centers in all directionbs
    Face=Center=()
    for idim=1:dim
        Face   = (Face...,   range(XF₋[idim], XF₊[idim]; length = N[idim]  ))
        Center = (Center..., range(XC₋[idim], XC₊[idim]; length = N[idim]-1))
    end

    # Local coordinate index arrays (updated for parallel simulations)
    ind_local = ()
    for idim=1:dim
        ind_local = (ind_local..., 1:N[idim])
    end

    # Initialize corners (to iterate over local portion of grid; note that these values are overwritten in parallel cases)
    corners = (lower = CartesianIndex((ones(Int64,dim)...,)), upper=CartesianIndex(N), size=N );        
    ghostcorners = (lower = CartesianIndex((ones(Int64,dim)...,)), upper=CartesianIndex(N), size=N );
    
    # Initialise backend and create grid structure
    grid = RegularRectilinearCollocatedGrid{FT, dim, typeof(backend)}(
          dim, N, N, N,                                 # dimensions, and sizes of grid
          topology[1:dim],                              # boundary conditions
          stencilwidth,                                 # size of halo (in parallel)
          L, Δ,                                         # domain size and (regular) spacing 
          Center, Face, Center, Face, ind_local,        # data related to the 1D local/global coordinate grids
          corners, ghostcorners,                        # corners of local grid
          backend, petsc_data(), nothing,               # data related to backend
          NamedTuple())                                 # fields              

    # Initialize grid - this will reset some of the default parameters above     
    initialize_grid!(grid, 
                    stencilwidth= stencilwidth, 
                    stenciltype = stenciltype,
                            opts= opts,
                          fields= fields)

    return grid
end


function validate_regular_grid_domain(FT, extent, x, y, z)

    # Find domain endpoints or domain extent, depending on user input:
    if !isnothing(extent) # the user has specified an extent
        dim   = length(extent)
        x,y,z = nothing, nothing, nothing

        L = extent;

        # Default domain:
        X₁ = -1 .* collect(L) ./ 2
        if dim>2
            X₁[3] = -L[3]
        end
        X₁ = (X₁...,)

    else # isnothing(extent) === true implies that user has not specified a length
        
        if isnothing(y)
            L = (x[2] - x[1],)
            X₁= (x[1], )
        elseif isnothing(z)
            L = (x[2] - x[1], y[2] - y[1])
            X₁= (x[1], y[1])
        else
            L = (x[2] - x[1], y[2] - y[1], z[2] - z[1])
            X₁= (x[1], y[1], z[1])
        end
        
    end

    return FT.(L), FT.(X₁)
end



function domain_string(grid)
    
    xₗ, xᵣ = grid.Face_global[1][1], grid.Face_global[1][end]
    if grid.dim>1
        yₗ, yᵣ = grid.Face_global[2][1], grid.Face_global[2][end]
    end
    if grid.dim>2
        zₗ, zᵣ = grid.Face_global[3][1], grid.Face_global[3][end]
    end
    if grid.dim==1
        return "x ∈ [$xₗ, $xᵣ]"
    elseif grid.dim==2
        return "x ∈ [$xₗ, $xᵣ], y ∈ [$yₗ, $yᵣ]"
    elseif grid.dim==3
        return "x ∈ [$xₗ, $xᵣ], y ∈ [$yₗ, $yᵣ], z ∈ [$zₗ, $zᵣ]"
    end
end




function resolution_string(g)
    if g.backend.mpi
        if g.backend.type==:ParallelStencil
            str = "$(g.N) (global, no halo)
                 $(g.Nl) (local  + halo)
                 $(g.Ng) (global + halo)"
        else
            str = "$(g.N) (global, no halo)
                 $(g.Nl) (local,  no halo)"
        end

    else
        str="$(g.N)"
    end
    return str
end

# view grid object
function show(io::IO, g::RegularRectilinearCollocatedGrid{FT, DIM, B}) where {FT, DIM, B}
    
    if g.backend.mpi
        mpi_type = "| MPI "
    else
        mpi_type = ""
    end

    print(io, "RegularRectilinearCollocatedGrid{$FT, $DIM, $B}\n",
              "        Backend: $(g.backend.type) ( $(g.backend.arch) $mpi_type) \n",
              "       gridtype: Collocated \n",
              "         domain: $(domain_string(g)) \n",
              "       topology: ", g.topology, '\n',
              "     resolution: $(resolution_string(g))\n",
              " grid spacing Δ: ", g.Δ, "\n",
              "         fields: $(keys(g.fields)) \n" )
end





"""
    initialize_grid(grid::RegularRectilinearCollocatedGrid{FT, 2, backend{BackendPETSc}})

Initializes a `DMDA` object using the `PETSc` backend. Note that this `DMDA` has one degree of freedom,
and is used to store information. If you want to base a solver on this, 
"""
function initialize_grid!(grid::RegularRectilinearCollocatedGrid{FT, D, Backend{BackendPETSc,FT}};
        stencilwidth=1, stenciltype=:Star, opts=(), fields::NamedTuple=NamedTuple()) where {FT, D}

    # initialize backend
    petsclib = check_backend(grid.backend, Scalar=backend.Scalar);
    
    # Transform local boundaries to PETSc boundary conditions 
    bcs = Vector{PETSc.LibPETSc.DMBoundaryType}(undef, D)
    for idim=1:D
        bcs[idim] = bcs_translate(grid.topology[idim]);
    end

    # Stencil type
    if stenciltype==:Star
        stenciltype=PETSc.DMDA_STENCIL_STAR;
    else
        stenciltype=PETSc.DMDA_STENCIL_BOX;
    end    
    
    # Create the DMDA
    da = PETSc.DMDA(
        petsclib,
        comm,
        (bcs...,),               # boundary conditions
        grid.N,                  # Global grid size
        1,                       # Number of DOF per node [set to 1 as this DMDA is just to store vectors with data]
        stencilwidth,            # Stencil width
        stenciltype;             # Stencil type
        opts...,
    )

    # Set regular coordinates
    c_start = -1*ones(D)
    c_end = ones(D)
    for idim=1:D
        c_start[idim] = grid.Face[idim][1];
        c_end[idim] = grid.Face[idim][end];
    end
    PETSc.setuniformcoordinates!(da, (c_start...,), (c_end...,))

    # Determine number of local grid points
    corners = PETSc.getcorners(da)
    grid.Nl = corners.size[1:D]         # local with no halo
    grid.Ng = PETSc.getinfo(da).global_size[1:D]     # global size 
     
    # Add local coordinates of Faces (can likely be done more elegantly)
    coord = PETSc.getlocalcoordinatearray(da)   # local coordinate array
    Face_local=Center_local=()
    for iDim=1:D
        Face = coord[iDim,corners.lower]:grid.Δ[iDim]:coord[iDim,corners.upper]
        Center = (Face[2:end] .+ Face[1:end-1])./2.0
        
        Face_local = (Face_local..., Face)
        Center_local = (Center_local..., Center)
    end
    grid.Face = Face_local
    grid.Center = Center_local

    # initialize fields - they are initialized as global vectors with 1 dof
    names = keys(fields)
    val   = values(fields)
    for ifield=1:length(fields)
        # Create global vector
        grid.fields = add_field(grid.fields, names[ifield], PETSc.DMGlobalVec(da) )

        # Set constant value to global vector
        fill!(grid.fields[ifield], val[ifield])

    end

    # Corners of grid
    grid.corners = PETSc.getcorners(da)
    grid.ghostcorners = PETSc.getghostcorners(da)

    # Store info in a PETSc data object
    grid.PETSc = petsc_data(petsclib, da)

    return nothing
end


"""
    initialize_grid(grid::RegularRectilinearCollocatedGrid{FT, 2, backend{BackendParallelStencil}})

Initializes a grid when we use `ParallelStencil` (and potentially `ImplicitGlobalGrid`) as a Backend
"""
function initialize_grid!(grid::RegularRectilinearCollocatedGrid{FT, D, Backend{BackendParallelStencil,FT}};
        stencilwidth=1, stenciltype=:Star, opts=nothing, fields::NamedTuple=NamedTuple()) where {FT, D}

    # initialize backend
    check_backend(grid.backend);
    
    # retrieve global & local grid dimensions
    if grid.backend.mpi 
        initialize_backend!(grid, grid.backend, stencilwidth, opts, fields)
    end

    # Initialize fields
    initialize_fields!(grid, grid.backend, fields)

end

"""
    local_coordinates(b::Backend{BackendParallelStencil}, grid)

Computes local coordinate vectors, including the halo points 
"""
function local_coordinates!(grid::RegularRectilinearCollocatedGrid{FT, D, Backend{BackendParallelStencil,FT}}, b::Backend{BackendParallelStencil}) where {FT, D}

    Face_local=Center_local=()
    for idim=1:grid.dim
        ind_nohalo = grid.ind_local[idim];
        
        # As we have a constant grid-spacing this is easy
        ind_s, ind_e = ind_nohalo[1], ind_nohalo[end]
        stencilwidth = grid.stencilwidth
        
        range_local = -stencilwidth:(ind_e-ind_s)+stencilwidth;

        Face = range_local*grid.Δ[idim] .+ grid.Face[idim][ind_s]
        Center = (Face[2:end] .+ Face[1:end-1])./2.0
        
        Face_local = (Face_local..., Face)
        Center_local = (Center_local..., Center)
    end
  
    grid.Face   = Face_local;
    grid.Center = Center_local;

    return nothing

end