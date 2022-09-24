#pragma once

#include "molecule_sim_helper.cuh"
#include "Molecule.h"
#include "Environment.h"

class MoleculeSimulator 
{
private:
	//member variables
	vector<Environment> environments_;
	int num_molecules_ = 0;
	int num_environments_ = 0;

	int* env_molecules_d_;
	int* env_molecules_h_;

	double* env_pos_x_h_;
	double* env_pos_y_h_;
	double* env_dim_x_h_;
	double* env_dim_y_h_;

	double* env_pos_x_d_;
	double* env_pos_y_d_;
	double* env_dim_x_d_;
	double* env_dim_y_d_;

	double* pos_x_h_;
	double* pos_y_h_;
	double* vel_x_h_;
	double* vel_y_h_;
	double* radii_h_;

	double* pos_x_d_;
	double* pos_y_d_;
	double* vel_x_d_;
	double* vel_y_d_;
	double* radii_d_;

public:
	//constructor
	MoleculeSimulator() = default;
	MoleculeSimulator(vector<Environment>);

	//destructor
	~MoleculeSimulator();

	//methods
	void update_environments();
	void simulate_molecules();
private:
	void allocate_workspace();
	void copy_env_to_memory();
};