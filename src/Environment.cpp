#include "Environment.h"
#include <random>
#include <math.h> 

//constructor
Environment::Environment(glm::vec2 pos, glm::vec2 dim, int max_molecules, double density)
	:font_size_{ 25 }, 
	pos_ { pos }, 
	dim_{ dim }, 
	max_molecules_{ max_molecules },
	density_{ CLAMP(density, 0, 1) },
	background_{ RectangleShape(pos, dim) }
{
	font1_.load("font1.ttf", font_size_);
	num_molecules_ = ceil(max_molecules_ * density_);
}

//destructor
Environment::~Environment()
{
}

void Environment::set_background_clr(ofColor clr)
{
	background_.fill_clr_ = clr;
}

void Environment::set_border_clr(ofColor clr) 
{
	background_.stroke_clr_ = clr;
}

//methods
void Environment::display() const
{
	display_background();
	display_label();
}

void Environment::display_background() const
{
	background_.display();
}

void Environment::display_label() const
{
	int padding = 10;
	glm::vec2 pos(pos_.x + padding, pos_.y + font_size_ + padding);
	std::string str = "Density: " + ofToString(density_, 2);
	ofSetColor(ofColor::black);
	font1_.drawString(str, pos.x, pos.y);
}