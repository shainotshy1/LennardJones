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
	double* dim_y,
	int* csr_i,
	int* csr_j,
	int* csr_v,
	double global_dim_x,
	double global_dim_y,
	double grid_dim);

//csr_i, csr_j - csr row offsets and col pointers representing global position of molecules
//csr_v - indice values of molecules in global molecule array
__device__ void simulate_molecules(int i,
	double* pos_x,
	double* pos_y,
	double* vel_x,
	double* vel_y,
	double* radii,
	int* csr_i,
	int* csr_j,
	int* csr_v);

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