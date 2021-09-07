"""
    This implements the Grids infrastructure
"""

# Grid infrastructure. Note that much of this is taken from Oceananigans

#####
##### Place-holder functions
#####


export
    Center, Face,
    AbstractTopology, Periodic, Bounded, Ghosted, topology,
    AbstractGrid,
    AbstractRectilinearGrid, RegularRectilinearCollocatedGrid,
    initialize_grid!, add_field

import Base: size, length, eltype, show, convert

#####
##### Abstract types
#####

"""
    AbstractField{X, Y, Z, A, G}

Abstract supertype for fields located at `(X, Y, Z)` with data stored in a container
of type `A`. The field is defined on a grid `G`.
"""
abstract type AbstractField{X, Y, Z, A, G} end

"""
    Center

A type describing the location at the center of a grid cell.
"""
struct Center end

"""
	xFace

A type describing the location at the x-face of a grid cell.
"""
struct xFace end


"""
	yFace

A type describing the location at the y-face of a grid cell.
"""
struct yFace end


"""
	zFace

A type describing the location at the z-face of a grid cell.
"""
struct zFace end


"""
	Vertex

A type describing the location at the vertex of a grid cell.
"""
struct Vertex end

"""
	Edge

A type describing the location at the edge of a grid cell.
"""
struct Edge end

"""
    AbstractTopology

Abstract supertype for grid topologies.
"""
abstract type AbstractTopology end

"""
    Periodic

Grid topology for periodic dimensions.
"""
struct Periodic <: AbstractTopology end

"""
    Bounded

Grid topology for bounded dimensions. These could be wall-bounded dimensions
or dimensions
"""
struct Bounded <: AbstractTopology end

"""
    Ghosted

Grid topology for bounded dimensions where we set boundary conditions using ghost points. These could be wall-bounded dimensions
or dimensions
"""
struct Ghosted <: AbstractTopology end

"""
    AbstractGrid{FT, TX, TY, TZ}

Abstract supertype for grids with elements of type `FT` and topology `{TX, TY, TZ}`.
"""
abstract type AbstractGrid{FT, DIM, B} end

"""
    AbstractCollocatedRectilinearGrid{FT, TX, TY, TZ}

Abstract supertype for rectilinear grids with collocated points, with elements of type `FT` and topology `{TX, TY, TZ}`.
"""
abstract type AbstractRectilinearCollocatedGrid{FT, DIM,  B} <: AbstractGrid{FT, DIM, B} end



"""
    fields = add_field(fields::NamedTuple, name::Symbol, newfield)

Adds a new field to the `fields` NamedTuple
"""
function add_field(fields::NamedTuple, name::Symbol, newfield)
    
    fields = merge(fields, NamedTuple{(name,)}( (newfield,)) );
    
    return fields
end

Base.eltype(::AbstractGrid{FT}) where FT = FT
Base.size(grid::AbstractGrid) = (grid.Nx, grid.Ny, grid.Nz)
Base.length(grid::AbstractGrid) = (grid.Lx, grid.Ly, grid.Lz)


# Include the different grid types 
include("regular_collocated_rectilinear_grid.jl")

