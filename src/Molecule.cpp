#include "Molecule.h"

Molecule::Molecule(glm::vec2 pos,
	glm::vec2 vel,
	double radius,
	ofColor fill_clr,
	ofColor stroke_clr)
	: molecule_{ CircleShape(pos, 
		glm::vec2(radius, radius), 
		fill_clr,
		stroke_clr) },
	vel_{ vel }
{
}

Molecule::~Molecule()
{
}

void Molecule::display() const
{
	molecule_.display();
}

void Molecule::update()
{
	double x = molecule_.pos_.x + vel_.x;
	double y = molecule_.pos_.y + vel_.y;

	set_pos(glm::vec2(x, y));
}

glm::vec2 Molecule::get_pos() const
{
	return molecule_.pos_;
}

glm::vec2 Molecule::get_vel() const
{
	return vel_;
}

double Molecule::get_radius() const 
{
	return molecule_.dim_.x;
}

void Molecule::set_pos(glm::vec2 pos)
{
	molecule_.pos_ = pos;
}

void Molecule::set_vel(glm::vec2 vel)
{
	vel_ = vel;
}

void Molecule::set_color(ofColor clr) 
{
	molecule_.fill_clr_ = clr;
}