#pragma once

#include "CircleShape.h"

class Molecule 
{
private:
	//member variables
	CircleShape molecule_;

public:
	//constructor
	Molecule(glm::vec2 pos,
		double radius,
		ofColor fill_clr_ = ofColor::red,
		ofColor stroke_clr = ofColor::black);

	//destructor
	~Molecule();

	//methods
	void display() const;
	glm::vec2 get_pos() const;
	void set_pos(glm::vec2);
	void set_color(ofColor);
};