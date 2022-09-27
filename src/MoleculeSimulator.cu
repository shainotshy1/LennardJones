#include "MoleculeSimulator.cuh"
#include "cuda_memory_utils.cuh"
#include "CircleShape.h"

MoleculeSimulator::MoleculeSimulator(glm::vec2 global_dim, 
	vector<Environment> environments, 
	double min_radius)
	: global_dim_{ global_dim }, 
	environments_ { environments }, 
	min_radius_{ min_radius}
{
	for (auto env : environments_) {
		num_molecules_ += env.num_molecules_;
	}
	num_environments_ = environments_.size();
	allocate_workspace();
	copy_env_to_memory();
}

MoleculeSimulator::~MoleculeSimulator() 
{	
	delete_on_device(env_molecules_d_);
	delete_on_device(env_pos_x_d_);
	delete_on_device(env_pos_y_d_);
	delete_on_device(env_dim_x_d_);
	delete_on_device(env_dim_y_d_);
	delete_on_device(pos_x_d_);
	delete_on_device(pos_y_d_);
	delete_on_device(vel_x_d_);
	delete_on_device(vel_y_d_);
	delete_on_device(radii_d_);

	delete[] env_molecules_h_;
	delete[] env_pos_x_h_;
	delete[] env_pos_y_h_;
	delete[] env_dim_x_h_;
	delete[] env_dim_y_h_;
	delete[] pos_x_h_;
	delete[] pos_y_h_;
	delete[] vel_x_h_;
	delete[] vel_y_h_;
	delete[] radii_h_;
}

void MoleculeSimulator::allocate_workspace()
{
	pos_x_h_ = new double[num_molecules_];
	pos_y_h_ = new double[num_molecules_];
	vel_x_h_ = new double[num_molecules_];
	vel_y_h_ = new double[num_molecules_];
	radii_h_ = new double[num_molecules_];

	allocate_on_device(&pos_x_d_, num_molecules_);
	allocate_on_device(&pos_y_d_, num_molecules_);
	allocate_on_device(&vel_x_d_, num_molecules_);
	allocate_on_device(&vel_y_d_, num_molecules_);
	allocate_on_device(&radii_d_, num_molecules_);

	env_molecules_h_ = new int[num_environments_];
	env_pos_x_h_ = new double[num_environments_];
	env_pos_y_h_ = new double[num_environments_];
	env_dim_x_h_ = new double[num_environments_];
	env_dim_y_h_ = new double[num_environments_];

	allocate_on_device(&env_molecules_d_, num_environments_);
	allocate_on_device(&env_pos_x_d_, num_environments_);
	allocate_on_device(&env_pos_y_d_, num_environments_);
	allocate_on_device(&env_dim_x_d_, num_environments_);
	allocate_on_device(&env_dim_y_d_, num_environments_);

	double world_grid_dim = sqrt(2) * min_radius_;
	int rows = ceil(global_dim_.x / world_grid_dim);

	grid_i_h_ = new int[rows + 1];
	grid_j_h_ = new int[num_molecules_];
	grid_v_h_ = new int[num_molecules_];

	allocate_on_device(&grid_i_d_, rows + 1);
	allocate_on_device(&grid_j_d_, num_molecules_);
	allocate_on_device(&grid_v_d_, num_molecules_);
}

void MoleculeSimulator::copy_env_to_memory() 
{
	int mol_i = 0;
	for (auto env : environments_) {

		glm::vec2 env_dim = env.dim_;
		glm::vec2 env_pos = env.pos_;
		//double area = dim.x * dim.y;
		//double area_proportion = 0.2; //molecules will maximum take __% of area of environment
		double radius = 5;//sqrt(area * area_proportion / (PI * max_molecules_));
		double vel_mag = 7;
		
		for (int i = 0; i < env.num_molecules_; i++) {
			//TO DO: make rand function in a helper file
			double rand_x = (double)rand() / (float)RAND_MAX;
			double rand_y = (double)rand() / (float)RAND_MAX;
			double x = rand_x * (env_dim.x - 2 * radius) + env_pos.x + radius;
			double y = rand_y * (env_dim.y - 2 * radius) + env_pos.y + radius;
			
			double rand_vel = (double)rand() / (float)RAND_MAX;
			double theta = rand_vel * 2 * PI;
			double vel_x = cos(theta) * vel_mag;
			double vel_y = sin(theta) * vel_mag;

			pos_x_h_[mol_i] = x;
			pos_y_h_[mol_i] = y;
			vel_x_h_[mol_i] = vel_x;
			vel_y_h_[mol_i] = vel_y;
			radii_h_[mol_i] = radius;

			mol_i++;
		}
	}

	copy_to_device(pos_x_h_, pos_x_d_, num_molecules_);
	copy_to_device(pos_y_h_, pos_y_d_, num_molecules_);
	copy_to_device(vel_x_h_, vel_x_d_, num_molecules_);
	copy_to_device(vel_y_h_, vel_y_d_, num_molecules_);
	copy_to_device(radii_h_, radii_d_, num_molecules_);

	for (int i = 0; i < num_environments_; i++) {
		env_molecules_h_[i] = environments_[i].num_molecules_;
		glm::vec2 pos = environments_[i].pos_;
		glm::vec2 dim = environments_[i].dim_;
		env_pos_x_h_[i] = pos.x;
		env_pos_y_h_[i] = pos.y;
		env_dim_x_h_[i] = dim.x;
		env_dim_y_h_[i] = dim.y;
	}

	copy_to_device(env_molecules_h_, env_molecules_d_, num_environments_);
	copy_to_device(env_pos_x_h_, env_pos_x_d_, num_environments_);
	copy_to_device(env_pos_y_h_, env_pos_y_d_, num_environments_);
	copy_to_device(env_dim_x_h_, env_dim_x_d_, num_environments_);
	copy_to_device(env_dim_y_h_, env_dim_y_d_, num_environments_);

	copy_to_device(grid_i_h_, grid_i_d_, 0);
}

void MoleculeSimulator::update_environments()
{
	copy_to_host(pos_x_d_, pos_x_h_, num_molecules_);
	copy_to_host(pos_y_d_, pos_y_h_, num_molecules_);

	for (auto env : environments_) {
		env.display();
	}

	for (int i = 0; i < num_molecules_; i++) {
		glm::vec2 pos = glm::vec2(pos_x_h_[i], pos_y_h_[i]);
		double r = radii_h_[i];
		CircleShape(pos, glm::vec2(r, r)).display();
	}
}

void MoleculeSimulator::simulate_molecules()
{
	glm::vec2 env_pos;
	glm::vec2 env_dim;

	int block_size = 512;
	int num_blocks = (num_molecules_ - 1) / block_size + 1;

	dim3 block(block_size);
	dim3 grid(num_blocks);

	simulate_molecules_gpu << <grid, block >> > (env_molecules_d_,
		pos_x_d_,
		pos_y_d_,
		vel_x_d_,
		vel_y_d_,
		radii_d_,
		num_molecules_,
		num_environments_,
		env_pos_x_d_,
		env_pos_y_d_,
		env_dim_x_d_,
		env_dim_y_d_,
		grid_i_d_,
		grid_j_d_,
		grid_v_d_,
		global_dim_.x,
		global_dim_.y,
		min_radius_);

	cudaDeviceSynchronize();
}