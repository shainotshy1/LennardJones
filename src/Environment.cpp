#include "Environment.h"
#include <random>

//constructor
Environment::Environment(glm::vec2 pos, glm::vec2 dim, double density)
	:font_size_{ 25 }, 
	pos_ { pos }, 
	dim_{ dim }, 
	density_{ CLAMP(density, 0, 1) },
	background_{ RectangleShape(pos, dim) }
{
	font1_.load("font1.ttf", font_size_);
	reset();
}

//destructor
Environment::~Environment()
{
}

void Environment::update()
{
}

void Environment::reset()
{
	molecules_ = vector<Molecule>{};
	int num_molecules = 5;
	load_molecules_(num_molecules);
}

void Environment::set_background_clr(ofColor clr)
{
	background_.fill_clr_ = clr;
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
	background_.display();
}

void Environment::display_molecules() const
{
	for (Molecule molecule : molecules_) {
		molecule.display();
	}
}

void Environment::display_label() const
{
	int padding = 10;
	glm::vec2 pos(pos_.x + padding, pos_.y + font_size_ + padding);
	std::string str = "Density: " + ofToString(density_, 2);
	ofSetColor(ofColor::black);
	font1_.drawString(str, pos.x, pos.y);
}

//*******************************************************PRIVATE METHODS*****************************************************//

void Environment::load_molecules_(int n)
{
	for (int i = 0; i < n; i++) {

		double radius = 20;

		double rand1 = (double)rand() / (float)RAND_MAX;
		double rand2 = (double)rand() / (float)RAND_MAX;
		double x = rand1 * (dim_.x - 2 * radius) + pos_.x + radius;
		double y = rand2 * (dim_.y - 2 * radius) + pos_.y + radius;
		glm::vec2 pos = glm::vec2(x, y);
		
		Molecule molecule = Molecule(pos, radius);
		molecules_.push_back(molecule);
	}
}