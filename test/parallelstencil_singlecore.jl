using Test, CompGrids, ParallelStencil, MPI

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

    # Specify with x-coordinates
    grid = RegularRectilinearCollocatedGrid(size=(10,), x=(-5, 12))
    @test grid.L[1] == 17.0

    # Specify as single numbers (instead of Tuple)
    grid = RegularRectilinearCollocatedGrid(size=10, extent=100.0)
    @test grid.L[1] == 100.0

    # Create fields on the grid as well
    @init_parallel_stencil(Threads, Float64, 1);
    grid = RegularRectilinearCollocatedGrid(size=10, extent=100.0, fields=(T=0,P=11.22))
    @test grid.fields[:P][1] == 11.22
    
end