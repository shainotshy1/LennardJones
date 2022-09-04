#pragma once

#include "ofMain.h"
#include "RectangleShape.h"
#include "Molecule.h"

class Environment {
private:
	//member variables
	RectangleShape background_;
	ofTrueTypeFont font1_;
	int font_size_;

	glm::vec2 pos_;
	glm::vec2 dim_;
	double density_;

	vector<Molecule> molecules_;

public:
	//constructor
	Environment(glm::vec2 pos, glm::vec2 dim, double density = 0.25);
	
	//destructor
	~Environment();
	
	//methods
	void update();
	void reset();

	void set_background_clr(ofColor);
	void display() const;
	void display_background() const;
	void display_molecules() const;
	void display_label() const;

private:
	
	//private methods
	void load_molecules_(int n);
};