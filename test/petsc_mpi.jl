using Test, CompGrids, MPI, PETSc, OffsetArrays
# Parallel tests

@init_backend(PETSc, Threads, true); # Backend with MPI and ImplicitGlobalGrid

# 1D 
grid = RegularRectilinearCollocatedGrid(size=(16,), x=(-5,10))
if mpirank==1
    @test grid.N[1] == 16
    @test grid.Ng[1] == 16
    @test grid.Nl[1] == 4
    @test length(grid.Face[1]) == grid.Nl[1]
    @test length(grid.Face_global[1]) == grid.N[1]
    @test grid.Face[1][1] ≈ -1.0
end

# 1D with a wider stencil 
grid = RegularRectilinearCollocatedGrid(size=(16,), x=(-5,10), stencilwidth=2)
if mpirank==1
    @test grid.N[1] == 16
    @test grid.Ng[1] == 16
    @test grid.Nl[1] == 4
    @test length(grid.Face[1]) == grid.Nl[1]
    @test length(grid.Face_global[1]) == grid.N[1]
    @test grid.Face[1][1] ≈ -1.0
end

# 2D 
grid = RegularRectilinearCollocatedGrid(size=(16,8), extent=(10,12))
if mpirank==3
    @test grid.N  == (16, 8)
    @test grid.Ng == (16, 8)
    @test grid.Nl == (8, 4)
end

# 3D 
grid = RegularRectilinearCollocatedGrid(size=(16,8, 8), extent=(10,12, 20.5))
if mpirank==3
    @test grid.N  == (16, 8, 8)
    @test grid.Ng == (16, 8, 8)
    @test grid.Nl == (8, 8, 4)
end

# Add fields as well
grid = RegularRectilinearCollocatedGrid(size=(16,8), extent=(10,12), fields=(T=0,P=1))
if mpirank==2
    ind = LinearIndices(grid.Nl)
    #id = OffsetArray(ind, axis1, axis2)
    #@show grid grid.corners grid.ghostcorners ind
    # Set values using corners
   
    #for i in grid.corners.lower:grid.corners.upper
    #    grid.fields.T[ind[i]] = grid.Face[1][i[1]] + grid.Face[2][i[2]]
    #end

   # for i in (grid.corners.lower[1]):(grid.corners.upper[end])
  #  @show grid.fields.T
end
grid = RegularRectilinearCollocatedGrid(
                    size=(10,20), 
                    extent=(100.0,110), 
                    fields=(T=0,P=11.22))



#=
# 2D - non-default different processor layout
opts=Dict(:dimy=>1,:dimx=>4)
grid = RegularRectilinearCollocatedGrid(size=(16,8), extent=(10,12), opts=opts)
if mpirank==1
    @test grid.N  == (16, 8)
    @test grid.Ng == (16, 8)
    @test grid.Nl == (4, 8)
end
=#
