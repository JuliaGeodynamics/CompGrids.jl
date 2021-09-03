using Test

# Grid generation, 2D
b    = backend(type=:PETSc,arch=:CPU,mpi=false)
grid = RegularRectilinearCollocatedGrid(size=(32, 32), extent=(1, 2), backend=b)
@test grid.L[1] == 1.0
@test grid.L[2] == 2.0

