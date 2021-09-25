using Test, CompGrids, ParallelStencil, MPI, ImplicitGlobalGrid

@testset "ParallelStencil grids" begin
    size_t = (32,32,32)
    length = (1, 2, 3)

    for idim=1:3
        for mpi in (false, true)
           
            if mpi
                if idim==1
                    ParallelStencil.@reset_parallel_stencil()
                    @init_backend(ParallelStencil, Threads, true,  1);
                elseif idim==2
                    ParallelStencil.@reset_parallel_stencil()
                    @init_backend(ParallelStencil, Threads, true,  2);
                elseif idim==3
                    ParallelStencil.@reset_parallel_stencil()
                    @init_backend(ParallelStencil, Threads, true,  3);
                end
            else
                if idim==1
                    ParallelStencil.@reset_parallel_stencil()
                    @init_backend(ParallelStencil, Threads, false, 1);
                elseif idim==2
                    ParallelStencil.@reset_parallel_stencil()
                    @init_backend(ParallelStencil, Threads, false, 2);
                elseif idim==3
                    ParallelStencil.@reset_parallel_stencil()
                    @init_backend(ParallelStencil, Threads, false, 3);
                end
            end
            grid = RegularRectilinearCollocatedGrid(size=(size_t[1:idim]...,), extent=length[1:idim])
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
    ParallelStencil.@reset_parallel_stencil()
    @init_backend(ParallelStencil, Threads, true, 2);
    grid = RegularRectilinearCollocatedGrid(size=(10,12), extent=(100.0,50), fields=(T=0,P=11.22))
    @test grid.fields[:P][1] == 11.22

    # Different methods to set values (using idx is identical to the PETSc implementation)
    grid.fields.P .= 0.0
    for i in grid.corners.lower:grid.corners.upper
        grid.fields.T[i] = i[1]
        grid.fields.P[grid.idx(i)] = i[1]
    end
    
    # test on one core
    @test sum(abs.(grid.fields.P - grid.fields.T)) == 0.0
    
end