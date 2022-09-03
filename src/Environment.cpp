#include "Environment.h"

//constructor
Environment::Environment(glm::vec2 pos, glm::vec2 dim, double density)
	:font_size_{ 25 }, 
	pos_ { pos }, 
	dim_{ dim }, 
	density_{ CLAMP(density, 0, 1) }
{
	font1_.load("font1.ttf", font_size_);
}

//destructor
Environment::~Environment()
{
}

void Environment::update()
{
}

//methods
void Environment::display() const
{
	display_background();
	display_molecules();
	display_label();
}

void Environment::display_background() const
{
	ofFill();
	ofSetLineWidth(3);
	ofSetColor(ofColor::lightGray);
	ofDrawRectangle(pos_, dim_.x, dim_.y);
	ofNoFill();
	ofSetColor(ofColor::black);
	ofDrawRectangle(pos_, dim_.x, dim_.y);
}

void Environment::display_molecules() const
{
}

void Environment::display_label() const
{
	int padding = 10;
	glm::vec2 pos(pos_.x + padding, pos_.y + font_size_ + padding);
	std::string str = "Density: " + ofToString(density_, 2);
	ofSetColor(ofColor::black);
	font1_.drawString(str, pos.x, pos.y);
}