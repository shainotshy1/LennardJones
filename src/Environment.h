#pragma once

#include "ofMain.h"
#include "RectangleShape.h"

struct Environment {
	//member variables
	RectangleShape background_;
	ofTrueTypeFont font1_;
	int font_size_;

	double density_;
	int max_molecules_ = 0;
	int num_molecules_ = 0;

	glm::vec2 pos_;
	glm::vec2 dim_;
	glm::vec2 world_dim_;
	
	//constructor
	Environment() = default;
	Environment(glm::vec2 pos, glm::vec2 dim, int max_molecules = 100, double density = 0.25);
	
	//destructor
	~Environment();
	
	//methods
	void set_background_clr(ofColor);
	void set_border_clr(ofColor);
	void display() const;
	void display_background() const;
	void display_label() const;
};