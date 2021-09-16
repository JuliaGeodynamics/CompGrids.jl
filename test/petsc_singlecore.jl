using Test, CompGrids, MPI, PETSc

@testset "PETSc grids" begin
    size_t = (32,32,32)
    length = (1, 2, 3)

    for idim=1:3
        @init_backend(PETSc, Base, true);
        grid = RegularRectilinearCollocatedGrid(size=size_t[1:idim], extent=length[1:idim])
        @test grid.L[1] == 1.0
        if idim>1
            @test grid.L[2] == 2.0
        end
        if idim>2
            @test grid.L[3] == 3.0
        end
    end
    
    # Create grid with fields
    grid = RegularRectilinearCollocatedGrid(size=(10,20), extent=(100.0,110), fields=(T=0,P=11.22))
    
    # Set values using corners
    ind = LinearIndices(grid.Nl)
    for i in grid.corners.lower:grid.corners.upper
        grid.fields.T[ind[i]] = grid.Face[1][i[1]] + grid.Face[2][i[2]]
    end

    # Check
  #  T = PETSc.unsafe_localarray(grid.fields.T);
 #   T = PETSc.reshapelocalarray(T, grid.PETSc.da)[1,:,:,:];
#    @test T[2,3]==grid.fields.T[idx[2,3]]


end