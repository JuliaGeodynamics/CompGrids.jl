"""
    RegularRectilinearCollocatedGrid{FT, TX, TY, TZ, R} <: AbstractRectilinearGrid{FT, TX, TY, TZ}

A rectilinear grid with collocated points and constant grid spacings `Δx`, `Δy`, and `Δz` between cell centers
and cell faces, elements of type `FT`, topology `{TX, TY, TZ}`, and coordinate ranges
of type `R`.
"""
# NOTE: we can later add info here about the local & global extent of the grid, the halo, the neigboring processors and the type of calculation (CPU/MPI-CPU/GPU)
mutable struct RegularRectilinearCollocatedGrid{FT, DIM, B} <: AbstractRectilinearCollocatedGrid{FT, DIM, B}
    # number of dimensions
    dim :: Int

    # Number of grid points in (x,y,z).
    N  :: Tuple
    
    # Topology
    topology :: Tuple
    
    # Domain size
    L  :: Tuple
    
    # Grid spacing 
    Δ  :: Tuple
    
    # Range of (global) coordinates at the centers of the cells.
    Center  :: Tuple
    
    # Range of (global) coordinates at the faces of the cells.
    Face  :: Tuple

    # Backend employed
    backend :: B
    
    # store PS or PETSc specific info
    petsc 

    # Parallel layout

    # Parallel Neighbors 

end

"""
        RegularRectilinearCollocatedGrid(FT=Float64;
                            size,
                            x = nothing, y = nothing, z = nothing,
                            topology= (Bounded, Bounded, Bounded),    
                             extent = nothing,
                             backend= backend()
                                 dof= 1,
                        stencilwidth= 1, 
                         stenciltype= :Star,
                          petsc_opts= (),
                        )

Creates a `RegularRectilinearCollocatedGrid` with `size = (Nx, Ny, Nz)` grid cells. 
All variables are collocated, that is, placed at the same points

Keyword arguments
=================

- `size` (required): A tuple prescribing the number of grid points all directions.
                     `size` is a 3-tuple for 3D models, a 2-tuple for 2D models, and a 1-tuple for 1D.

- `extent`: A tuple prescribing the physical extent of the grid.
            The origin for three-dimensional domains is the lower left corner `(-Lx/2, -Ly/2, -Lz)`.

- `x`, `y`, and `z`: Each of `x, y, z` are 2-tuples that specify the end points of the domain
                     in their respect directions. 

- `topology`: A tuple prescribing the topology of the grid in the different directions. 
              This is also used to set the type of the `DMDA` boundary conditions in case of using the `PETSc` backend. 
              Examples are `Bounded`, `Ghost`, `Periodic`

- `backend`: the backend used for the grid indicating whether we use ParallelStencil, PETSc or native Julia                  

*Note*: _Either_ `extent`, or all of `x`, `y`, and `z` must be specified.

Optional PETSc arguments
========================

- `dof`: the degree of freedoms per node                   

- `stencilwidth`: the width of the stencil                     

- `stenciltype`: the type of the stencil (`:Star` or `:Box`)                    

- `petsc_opts`: additional PETSc options                    


The physical extent of the domain can be specified via `x`, `y`, and `z` keyword arguments
indicating the left and right endpoints of each dimensions, e.g. `x=(0, 10)` or via
the `extent` argument, e.g. `extent=(Lx, Ly, Lz)` which specifies the extent of each dimension
in which case -Lx/2 ≤ x ≤ Lx/2, -Ly/2 ≤ y ≤ Ly/2, and -Lz ≤ z ≤ 0.

Constants are stored using floating point values of type `FT`. By default this is `Float64`.
Make sure to specify the desired `FT` if not using `Float64`.

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
julia> using CompGrids

julia> grid = RegularRectilinearGrid(size=(32, 32, 32), extent=(1, 2, 3))
RegularRectilinearGrid{Float64, Bounded, Bounded, Bounded}
                   domain: x ∈ [0.0, 1.0], y ∈ [0.0, 2.0], z ∈ [-3.0, 0.0]
                 topology: (Bounded, Bounded, Bounded)
  resolution (Nx, Ny, Nz): (32, 32, 32)
grid spacing (Δx, Δy, Δz): (0.03125, 0.0625, 0.09375)
```


"""
function RegularRectilinearCollocatedGrid(FT=Float64;
                                    size,
                                     x = nothing, y = nothing, z = nothing,
                               topology= (Bounded, Bounded, Bounded),    
                                extent = nothing,
                                backend= backend(),
                                    dof= 1,
                           stencilwidth= 1, 
                           stenciltype=  :Star,
                            petsc_opts=  (),
                              )

    dim =   length(size)                    # dimensions of the grid [1-3]
    L, X₁ = validate_regular_grid_domain(FT, extent, x, y, z)

    # Unpacking
    N = size
    Δ = L ./ N
    
    # Face-node limits in x, y, z
    XF₋ = @. X₁ 
    XF₊ = @. XF₋ + L              

    # Center-node limits in x, y, z
    XC₋ = @. XF₋ + Δ / 2
    XC₊ = @. XC₋ + L - Δ

    # Generate 1D coordinate arrays
    xF = range(XF₋[1], XF₊[1]; length = N[1]+1)
    xC = range(XC₋[1], XC₊[1]; length = N[1])
    if dim>1
        yF = range(XF₋[2], XF₊[2]; length = N[2]+1)
        yC = range(XC₋[2], XC₊[2]; length = N[2])
    elseif dim>3
        zF = range(XF₋[3], XF₊[3]; length = N[3]+1)
        zC = range(XC₋[3], XC₊[3]; length = N[3])
    end
    if dim==1
        Face = (xF,)
        Center = (xC,)
    elseif dim==2
        Face = (xF,yF)
        Center = (xC,yC)
    elseif dim==3
        Face = (xF,yF,zF)
        Center = (xC,yC,zC)
    end

    # Initialise backend and create grid structure
    grid = RegularRectilinearCollocatedGrid{FT, dim, typeof(backend)}(
          dim, N, topology[1:dim], L, Δ, Center, Face, backend, nothing)

    # Construct grid      
    initialize_grid!(grid, 
                             dof= dof, 
                    stencilwidth= stencilwidth, 
                    stenciltype = stenciltype,
                            opts= petsc_opts)

    return grid
end


function validate_regular_grid_domain(FT, extent, x, y, z)

    # Find domain endpoints or domain extent, depending on user input:
    if !isnothing(extent) # the user has specified an extent
        dim   = length(extent)
        x,y,z = nothing, nothing, nothing

        L = extent;

        # Default domain:
        X₁ = -1 .* L ./ 2
        if dim>2
            X₁[3] = -L[3]
        end

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
    
    xₗ, xᵣ = grid.Face[1][1], grid.Face[1][end]
    if grid.dim>1
        yₗ, yᵣ = grid.Face[2][1], grid.Face[2][end]
    elseif grid.dim>2
        zₗ, zᵣ = grid.Face[3][1], grid.Face[3][end]
    end
    if grid.dim==1
        return "x ∈ [$xₗ, $xᵣ]"
    elseif grid.dim==2
        return "x ∈ [$xₗ, $xᵣ], y ∈ [$yₗ, $yᵣ]"
    elseif grid.dim==3
        return "x ∈ [$xₗ, $xᵣ], y ∈ [$yₗ, $yᵣ], z ∈ [$zₗ, $zᵣ]"
    end
end

function show(io::IO, g::RegularRectilinearCollocatedGrid{FT, DIM, B}) where {FT, DIM, B}
    
    if g.backend.mpi
        mpi_type = "| MPI"
    else
        mpi_type = ""
    end

    print(io, "RegularRectilinearCollocatedGrid{$FT, $DIM, $B}\n",
              "        Backend: $(g.backend.type) ($(g.backend.arch) $mpi_type) \n",
              "       gridtype: Collocated \n",
              "         domain: $(domain_string(g))\n",
              "       topology: ", g.topology, '\n',
              "     resolution: ", g.N, '\n',
              " grid spacing Δ: ", g.Δ)
end





"""
    initialize_grid(grid::RegularRectilinearCollocatedGrid{FT, 2, backend{BackendPETSc}})

Initializes a `DMDA` object using the PETSc backend.
"""
function initialize_grid!(grid::RegularRectilinearCollocatedGrid{FT, DIM, backend{BackendPETSc}};
        dof=1, stencilwidth=1, stenciltype=:Star, opts=()) where {FT, DIM}

    # initialize backend
    petsclib, comm = initialize_backend(grid.backend, Scalar=FT);
    
    # make PETSc available
    @eval using PETSc
    
    # Transform local boundaries to PETSc boundary conditions 
    bcs_vec = Vector{PETSc.LibPETSc.DMBoundaryType}(undef,DIM)
    for idim=1:DIM
        bcs_vec[idim] = bcs_translate(grid.topology[idim]);
    end
    bcs = (bcs_vec...,)

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
        bcs,                     # boundary conditions
        grid.N,                  # Global grid size
        dof,                     # Number of DOF per node
        stencilwidth,            # Stencil width
        stenciltype;             # Stencil type
        opts...,
    )

    # Set coordinates
    c_start = -1*ones(DIM)
    c_end = ones(DIM)
    for idim=1:DIM
        c_start[idim] = grid.Face[idim][1];
        c_end[idim] = grid.Face[idim][end];
    end
    PETSc.setuniformcoordinates!(da, (c_start...,), (c_end...,))

    # Store info in a PETSc data object
    grid.petsc = petsc_data(petsclib, da)

    return nothing
end