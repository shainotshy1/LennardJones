#pragma once

#include "ofMain.h"

class Environment {
private:
	//member variables
	ofTrueTypeFont font1_;
	int font_size_;

	glm::vec2 pos_;
	glm::vec2 dim_;
	double density_;

public:
	//constructor
	Environment(glm::vec2 pos, glm::vec2 dim, double density = 0.25);
	
	//destructor
	~Environment();
	
	//methods
	void update();
	void display() const;
	void display_background() const;
	void display_molecules() const;
	void display_label() const;
};