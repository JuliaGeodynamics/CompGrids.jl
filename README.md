# CompGrids.jl
Creates computational grids and initializes fields on these grids that can be used with [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) or [PETSc.jl](https://github.com/JuliaParallel/PETSc.jl/tree/jek/gen)

The basic idea is that you specify a backend (`ParallelStencil` or `PETSc`), after which the grid can be created.
A 2D example using ParallelStencil & ImplicitGlobalGrid:
```julia
julia> using CompGrids, MPI, ParallelStencil, ImplicitGlobalGrid, PETSc
julia> @init_backend(ParallelStencil, Threads, true, 2);
julia> grid = RegularRectilinearCollocatedGrid(size=(10,20), extent=(100.0,110), fields=(T=0,P=11.22))
Global grid: 12x22x1 (nprocs: 1, dims: 1x1x1)
RegularRectilinearCollocatedGrid{Float64, 2, Backend{BackendParallelStencil, Float64}}
        Backend: ParallelStencil ( Threads | MPI ) 
       gridtype: Collocated 
         domain: x ∈ [-50.0, 50.0], y ∈ [-55.0, 55.0] 
       topology: (Bounded, Bounded)
     resolution: (10, 20) (global, no halo)
                 (12, 22) (local  + halo)
                 (12, 22) (global + halo)
 grid spacing Δ: (11.11111111111111, 5.7894736842105265)
         fields: (:T, :P)
```
We can initialize the same grid, but using the PETSc MPI-parallel backend, by changing one line:
```julia
julia> @init_backend(PETSc, Threads, true, 2);
julia> grid = RegularRectilinearCollocatedGrid(size=(10,20), extent=(100.0,110), fields=(T=0,P=11.22))
RegularRectilinearCollocatedGrid{Float64, 2, Backend{BackendPETSc, Float64}}
        Backend: PETSc ( Threads | MPI ) 
       gridtype: Collocated 
         domain: x ∈ [-50.0, 50.0], y ∈ [-55.0, 55.0] 
       topology: (Bounded, Bounded)
     resolution: (10, 20) (global, no halo)
                 (10, 20) (local,  no halo)
 grid spacing Δ: (11.11111111111111, 5.7894736842105265)
         fields: (:T, :P) 
```

The resulting `grid` is a structure that holds information about the grid (the global grid in case it is run on one core, and the local portion in case it is run in MPI-parallel). If indicated, it also initializes the fields (as a global vector for the `PETSc` and as 1D/2D/3D arrays for `PS`), which you can access with `grid.fields.T` or `grid.fields[:T]` (for the examples above).

### Development
This is work in progress. If you want to get a feel for it, have a look at the [test](https://github.com/JuliaGeodynamics/CompGrids.jl/tree/main/test) and [examples](https://github.com/JuliaGeodynamics/CompGrids.jl/tree/main/examples) directories. 
- The functionality is fairly complete for collocated grids (using `PS` and `PETSc`).
- Staggered grid support will be added 
- Support for other types of structurally regular grids (e.g., cylindrical, yin-yang?) should be fairly straightforward to add as well, but as we don't use these grids on a daily basis, we would require volunteers to help with this 
- Adding interfaces to other types of meshes (finite elements, AMR) could be added as well, but would require external volunteers 