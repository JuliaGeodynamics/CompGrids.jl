using Test
using MPI: mpiexec

# Do the MPI tests first so we do not have mpi running inside MPI
mpi_tests = ("parallelstencil_mpi.jl", )

# Perform the MPI tests 
for file in mpi_tests
    @testset "MPI test: $file" begin
        @test mpiexec() do mpi_cmd
            cmd =
                `$mpi_cmd -n 4 $(Base.julia_cmd()) --startup-file=no --project $file`
            success(pipeline(cmd, stderr = stderr))
        end
    end
end

# Perform non-MPI tests
include("./parallelstencil_singlecore.jl")
include("./petsc_singlecore.jl")
