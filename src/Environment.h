#pragma once

#include "ofMain.h"
#include "RectangleShape.h"
#include "Molecule.h"
#include "MoleculeSimulator.cuh"

class Environment {
private:
	//member variables
	RectangleShape background_;
	ofTrueTypeFont font1_;
	int font_size_;

	glm::vec2 pos_;
	glm::vec2 dim_;
	glm::vec2 world_dim_;
	double density_;

	int max_molecules_;
	vector<Molecule> molecules_;

	MoleculeSimulator simulator_;

public:
	//constructor
	Environment(glm::vec2 pos, glm::vec2 dim, int max_molecules = 100, double density = 0.25);
	
	//destructor
	~Environment();
	
	//methods
	void reset();
	void update();
	void update_molecules();
	void simulate_interactions();
	void load_molecules(int n, double radius, double vel_mag);

	void set_background_clr(ofColor);
	void set_border_clr(ofColor);
	void display() const;
	void display_background() const;
	void display_molecules() const;
	void display_label() const;
};