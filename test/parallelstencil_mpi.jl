using Test, CompGrids, ParallelStencil, MPI, ImplicitGlobalGrid
# Parallel tests

@init_backend(ParallelStencil, Threads, true); # Backend with MPI and ImplicitGlobalGrid

# 1D 
grid = RegularRectilinearCollocatedGrid(size=(16,), x=(-5,10))
if mpirank==1
    @test grid.N[1] == 16
    @test grid.Ng[1] == 18
    @test grid.Nl[1] == 6
    @test length(grid.Face[1]) == grid.Nl[1]
    @test length(grid.Face_global[1]) == grid.N[1]
    @test grid.Face[1][1] ≈ -2.0
end

# 1D with a wider stencil 
grid = RegularRectilinearCollocatedGrid(size=(16,), x=(-5,10), stencilwidth=2)
if mpirank==1
    @test grid.N[1] == 16
    @test grid.Ng[1] == 20
    @test grid.Nl[1] == 8
    @test length(grid.Face[1]) == grid.Nl[1]
    @test length(grid.Face_global[1]) == grid.N[1]
    @test grid.Face[1][1] ≈ -3.0
end

# 2D 
grid = RegularRectilinearCollocatedGrid(size=(16,8), extent=(10,12))
if mpirank==3
    @test grid.N  == (16, 8)
    @test grid.Ng == (18, 10)
    @test grid.Nl == (10, 6)
end

# 3D 
grid = RegularRectilinearCollocatedGrid(size=(16,8, 8), extent=(10,12, 20.5))
if mpirank==3
    @test grid.N  == (16, 8, 8)
    @test grid.Ng == (18, 10, 10)
    @test grid.Nl == (10, 6, 10)
end

# 2D - non-default different processor layout
opts=Dict(:dimy=>1,:dimx=>4)
grid = RegularRectilinearCollocatedGrid(size=(16,8), extent=(10,12), opts=opts)
if mpirank==1
    @test grid.N  == (16, 8)
    @test grid.Ng == (18, 10)
    @test grid.Nl == (6, 10)
end