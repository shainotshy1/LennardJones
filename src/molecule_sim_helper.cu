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

	int i = blockIdx.x * blockDim.x + threadIdx.x; //global molecule index

	if (i >= num_molecules) return;

	int j = 0; //environment index of molecule
	int k = 0;
	while (i >= k + env_molecules[j]) {
		k += env_molecules[j];
		j++;
	}

	bound_molecules(i,
		pos_x,
		pos_y,
		vel_x,
		vel_y,
		radii,
		env_pos_x[j],
		env_pos_y[j],
		dim_x[j],
		dim_x[j]);
	
	update_molecule(i, pos_x, pos_y, vel_x, vel_y);
}

__device__ void bound_molecules(int i,
	double* pos_x,
	double* pos_y,
	double* vel_x,
	double* vel_y,
	double* radii,
	double env_pos_x,
	double env_pos_y,
	double dim_x,
	double dim_y)
{
	double dif_pos_x = env_pos_x + dim_x - pos_x[i];
	double dif_pos_y = env_pos_y + dim_y - pos_y[i];
	double radius = radii[i];
	if (dif_pos_x >= dim_x - radius || dif_pos_x <= radius) {
		vel_x[i] *= -1;
	}
	if (dif_pos_y >= dim_y - radius || dif_pos_y <= radius) {
		vel_y[i] *= -1;
	}
}

__device__ void update_molecule(int i, 
	double* pos_x, 
	double* pos_y, 
	double* vel_x, 
	double* vel_y)
{
	pos_x[i] += vel_x[i];
	pos_y[i] += vel_y[i];
}