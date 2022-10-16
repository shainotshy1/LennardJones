#include "helper_utils.cuh"

__host__ __device__ int find_grid_index(double x,
	double y,
	double grid_x,
	double grid_y,
	int cell_dim_x,
	int cell_dim_y,
	double grid_dim)
{
	int row = (x - grid_x) / grid_dim;
	int col = (y - grid_y) / grid_dim;
	int index = (int)(row * cell_dim_x + col);

	if (index >= cell_dim_x * cell_dim_y) {
		return -1;
	}
	return index;
}