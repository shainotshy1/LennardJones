#include "molecule_sim_helper.cuh"

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
	double* dim_x,
	double* dim_y)

{
	//To Do: Implement Lennard Jones
	//To Do: Parrallelize molecules of all environments into ONE kernel call:
		//Contains an array of length n where n = # of environments
		//Array has number of molecules in each environment so we know what segments of the pos/vel/radii vectors map to which environments
	//Bounce off walls of environment

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= num_molecules) return;

	int j = 0;
	int k = 0;
	while (i >= k + env_molecules[j]) {
		k += env_molecules[j];
		j++;
	}

	double dif_pos_x = env_pos_x[j] + dim_x[j] - pos_x[i];
	double dif_pos_y = env_pos_y[j] + dim_y[j] - pos_y[i];
	double radius = radii[i];
	if (dif_pos_x >= dim_x[j] - radius || dif_pos_x <= radius) {
		vel_x[i] *= -1;
	}
	if (dif_pos_y >= dim_y[j] - radius || dif_pos_y <= radius) {
		vel_y[i] *= -1;
	}

	pos_x[i] += vel_x[i];
	pos_y[i] += vel_y[i];
}