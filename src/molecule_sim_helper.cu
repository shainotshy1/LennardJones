#include "molecule_sim_helper.cuh"
#include "cuda_memory_utils.cuh"
#include "helper_utils.cuh"

__global__ void simulate_molecules_gpu(int* env_molecules,
	double* pos_x,
	double* pos_y,
	double* vel_x,
	double* vel_y,
	double* radii,
	int num_molecules,
	int num_environments,
	double* env_pos_x,
	double* env_pos_y,
	double* env_dim_x,
	double* env_dim_y,
	int* grid,
	int cell_dim_x,
	int cell_dim_y,
	double global_dim_x,
	double global_dim_y,
	double global_pos_x,
	double global_pos_y,
	double grid_dim)
{
	//To Do: Implement Lennard Jones
	//To Do: Parrallelize molecules of all environments into ONE kernel call:
		//Contains an array of length n where n = # of environments
		//Array has number of molecules in each environment so we know what segments of the pos/vel/radii vectors map to which environments

	int i = blockIdx.x * blockDim.x + threadIdx.x; //global molecule index

	if (i >= num_molecules) return;

	int j = 0; //environment index of molecule
	int k = 0;
	while (i >= k + env_molecules[j]) {
		k += env_molecules[j];
		j++;
	}
	
	simulate_molecules(i,
		pos_x,
		pos_y,
		vel_x,
		vel_y,
		radii,
		grid,
		cell_dim_x,
		cell_dim_y,
		global_dim_x,
		global_dim_y,
		global_pos_x,
		global_pos_y,
		grid_dim);

	bound_molecules(i,
		pos_x,
		pos_y,
		vel_x,
		vel_y,
		radii,
		env_pos_x[j],
		env_pos_y[j],
		env_dim_x[j],
		env_dim_y[j]);
	
	update_molecule(i, 
		pos_x, 
		pos_y, 
		vel_x, 
		vel_y,
		grid,
		global_pos_x,
		global_pos_y,
		grid_dim,
		cell_dim_x,
		cell_dim_y);
}

__device__ void simulate_molecules(int i,
	double* pos_x,
	double* pos_y,
	double* vel_x,
	double* vel_y,
	double* radii,
	int* grid,
	int cell_dim_x,
	int cell_dim_y,
	double global_dim_x,
	double global_dim_y,
	double global_pos_x,
	double global_pos_y,
	double grid_dim)
{
#if 1

	int row = (pos_x[i] - global_pos_x) / grid_dim;
	int col = (pos_y[i] - global_pos_y) / grid_dim;
	int index = (int)(row * cell_dim_x + col);

	if (index >= cell_dim_x * cell_dim_y) {
		printf("ERROR index out of bound molecule");
	}
#else
	int index = find_grid_index(pos_x[i],
		pos_y[i],
		global_pos_x,
		global_pos_y,
		cell_dim_x,
		cell_dim_y,
		grid_dim);
#endif
	if (grid[index] != i) {
		printf("Error %d VS %d\n\n", grid[index], i);
	}
}

__device__ void bound_molecules(int i,
	double* pos_x,
	double* pos_y,
	double* vel_x,
	double* vel_y,
	double* radii,
	double env_pos_x,
	double env_pos_y,
	double env_dim_x,
	double env_dim_y)
{
	double dif_pos_x = env_pos_x + env_dim_x - pos_x[i];
	double dif_pos_y = env_pos_y + env_dim_y - pos_y[i];
	double radius = radii[i];
	if (dif_pos_x >= env_dim_x - radius || dif_pos_x <= radius) {
		vel_x[i] *= -1;
	}
	if (dif_pos_y >= env_dim_y - radius || dif_pos_y <= radius) {
		vel_y[i] *= -1;
	}
}

__device__ void update_molecule(int i, 
	double* pos_x, 
	double* pos_y, 
	double* vel_x, 
	double* vel_y,
	int* grid,
	double global_pos_x,
	double global_pos_y,
	double grid_dim,
	double cell_dim_x,
	double cell_dim_y)
{
	pos_x[i] += vel_x[i];
	pos_y[i] += vel_y[i];

	int row = (pos_x[i] - global_pos_x) / grid_dim;
	int col = (pos_y[i] - global_pos_y) / grid_dim;
	int index = (int)(row * cell_dim_x + col);

	if (index >= cell_dim_x * cell_dim_y) {
		printf("ERROR index out of bound molecule");
	}

	grid[index] = i;
}