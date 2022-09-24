#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
	double* dim_y);

__device__ void bound_molecules(int i,
	double* pos_x,
	double* pos_y,
	double* vel_x,
	double* vel_y,
	double* radii,
	double env_pos_x,
	double env_pos_y,
	double dim_x,
	double dim_y);

__device__ void update_molecule(int i,
	double* pos_x,
	double* pos_y,
	double* vel_x,
	double* vel_y);