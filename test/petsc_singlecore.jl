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
    

    # Create fields on the grid as well. 
    grid = RegularRectilinearCollocatedGrid(size=10, extent=100.0, fields=(T=0,P=11.22))
    
    @show grid


   # @test grid.fields[:P][1] == 11.22


end