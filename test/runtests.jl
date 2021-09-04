using Test, CompGrids, ParallelStencil, MPI, PETSc

@testset "ParallelStencil grids" begin
    size_t = (32,32,32)
    length = (1, 2, 3)

    for idim=1:3
        for mpi in (false, true)
            if mpi
                @init_backend(ParallelStencil, Threads, true);
            else
                @init_backend(ParallelStencil, Threads, false);
            end
            grid = RegularRectilinearCollocatedGrid(size=size_t[1:idim], extent=length[1:idim])
            @test grid.L[1] == 1.0
            if idim>1
                @test grid.L[2] == 2.0
            end
            if idim>2
                @test grid.L[3] == 3.0
            end
        end
    end

end


# Grid generation, 2D
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
    

end
