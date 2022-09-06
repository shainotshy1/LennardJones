#pragma once

#include "CircleShape.h"

class Molecule 
{
private:
	//member variables
	CircleShape molecule_; //circle image representation of molecule
	glm::vec2 vel_; //velocity

public:
	//constructors
	Molecule() = default;
	Molecule(glm::vec2 pos,
		glm::vec2 vel,
		double radius,
		ofColor fill_clr_ = ofColor::red,
		ofColor stroke_clr = ofColor::black);

	//destructor
	~Molecule();

	//methods
	void display() const;
	void update();
	glm::vec2 get_pos() const;
	glm::vec2 get_vel() const;
	double get_radius() const;
	void set_pos(glm::vec2);
	void set_vel(glm::vec2);
	void set_color(ofColor);
};