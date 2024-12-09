# super-computer
Команды для сборки и запуска программы на IBM Polus
--------------------------------------------------------------------------------
# OpenMP:
vim task1.cpp

g++ -fopenmp -std=c++11 task1.cpp -o newrun

vim task1_job.lsf

task1_job.lsf:
#BSUB -n M

#BSUB -W 00:15

#BSUB -o "my_job.%J.out"

#BSUB -e "my_job.%J.err"

#BSUB -R "span[hosts=1]"

OMP_NUM_THREADS=N ./my_job

Здесь параметр N указывает число OpenMP нитей,например, N=2. А параметр M число запрашиваемых ядер и рассчитывается исходя из того, что возможно 8 потоков на одно ядро, т.е. M = [N/8]+1.

bsub < task1_job.lsf

cat my_job.<xxx>.out  

xxx - это номер полученного задания

--------------------------------------------------------------------------------
# MPI:
vim task2.cpp

mpic++ -std=c++11 -o newrun task2.cpp

vim task2_job.lsf

task2_job.lsf:

#BSUB -n M

#BSUB -W 00:15

#BSUB -o "my_job.%J.out"

#BSUB -e "my_job.%J.err"

#BSUB -R "span[ptile=1]"

module load openmpi

mpirun -np N ./newrun

Здесь параметр M задает число запрашиваемых ядер. Параметр N - это количество запрашиваемых MPI-процессов.

bsub < task2_job.lsf

cat my_job.<xxx>.out

xxx - это номер полученного задания

--------------------------------------------------------------------------------
# MPI+OpenMP:
vim task3.cpp

mpic++ -fopenmp -std=c++11 -o newrun task3.cpp

vim task3_job.lsf

task3_job.lsf:

#BSUB -n M

#BSUB -W 00:15

#BSUB -o "my_job.%J.out"

#BSUB -e "my_job.%J.err"

export OMP_NUM_THREADS=N

mpirun -np Y ./newrun

Здесь параметр N указывает число OpenMP нитей на один MPI процесс. Параметр M задает число запрашиваемых ядер и равно числу запрашиваемых MPI-процессов. Параметр Y - это количество запрашиваемых MPI-процессов.

bsub < task3_job.lsf

cat my_job<xxx>.out

xxx - это номер полученного задания
