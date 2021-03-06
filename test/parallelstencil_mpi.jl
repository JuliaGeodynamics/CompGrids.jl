using Test, CompGrids, ParallelStencil, MPI, ImplicitGlobalGrid
# Parallel tests

@init_backend(ParallelStencil, Threads, true, 1); # Backend with MPI and ImplicitGlobalGrid

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


# 2D - with fields
ParallelStencil.@reset_parallel_stencil()
@init_backend(ParallelStencil, Threads, true, 2); # Backend with MPI and ImplicitGlobalGrid
grid = RegularRectilinearCollocatedGrid(size=(16,8), extent=(10,12), fields=(T=0, P=2))

# Loop over all local points (including ghost) and set values
ind = LinearIndices(grid.Nl)
#idx = LinearIndices(grid.ind_local)

idx(id::CartesianIndex) = LinearIndices(grid.Nl)[id]

for i in (grid.ghostcorners.lower:grid.ghostcorners.upper) 
    grid.fields.T[i] = grid.Face[1][i[1]]   + grid.Face[2][i[2]]     # set it to the y-coordinate
    grid.fields.P[idx(i)] = grid.Face[1][i[1]] + grid.Face[2][i[2]]
end


# Loop over non-ghost local points
for i in (grid.corners.lower:grid.corners.upper)
    grid.fields.P[i] = grid.Face_global[1][i[1]]
    grid.fields.T[grid.idx(i)] = grid.Face_global[1][i[1]]
end
@test grid.fields.P[2,2] == grid.Face_global[1][2]

diff = sum(abs.(grid.fields.P - grid.fields.T)) 
@test diff==0.0

if mpirank==2
    #@show grid.fields.P grid.fields.T diff
end

# 2D - with fields and specifying local grid size
ParallelStencil.@reset_parallel_stencil()
grid = RegularRectilinearCollocatedGrid(local_size=(16,8), extent=(10,12), fields=(T=0, P=1))
if mpirank==1
    @test grid.N  == (28, 12)
    @test grid.Ng == (30, 14)
    @test grid.Nl == (16, 8)
end