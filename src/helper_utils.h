#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__host__ __device__ int find_grid_index(double x,
	double y,
	double grid_x,
	double grid_y,
	int cell_dim_x,
	int cell_dim_y,
	double grid_dim);