#pragma once

#include "molecule_sim_helper.cuh"

class MoleculeSimulator 
{
public:
	//constructor
	MoleculeSimulator();

	//destructor
	~MoleculeSimulator();

	//methods
	void simulate_molecules();
};