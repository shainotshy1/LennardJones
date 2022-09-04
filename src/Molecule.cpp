#include "Molecule.h"

Molecule::Molecule(glm::vec2 pos,
	double radius,
	ofColor fill_clr,
	ofColor stroke_clr)
	: molecule_{ CircleShape(pos, 
		glm::vec2(radius, radius), 
		fill_clr,
		stroke_clr) }
{
}

Molecule::~Molecule()
{
}

void Molecule::display() const
{
	molecule_.display();
}

glm::vec2 Molecule::get_pos() const
{
	return molecule_.pos_;
}

void Molecule::set_pos(glm::vec2 pos)
{
	molecule_.pos_ = pos;
}

void Molecule::set_color(ofColor clr) 
{
	molecule_.fill_clr_ = clr;
}