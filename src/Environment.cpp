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
	reset();
}

//destructor
Environment::~Environment()
{
}

void Environment::reset()
{
	molecules_ = vector<Molecule>{};
	double area = dim_.x * dim_.y;
	double area_proportion = 0.2; //molecules will maximum take __% of area of environment
	double radius = sqrt(area * area_proportion / (PI * max_molecules_));
	double vel_mag = 7;

	int num_molecules = round(max_molecules_ * density_);
	load_molecules(num_molecules, radius, vel_mag);
}

void Environment::update()
{
	simulate_interactions();
	update_molecules();
}

void Environment::update_molecules()
{
	for (Molecule& molecule : molecules_) {
		molecule.update();
	}
}

void Environment::simulate_interactions()
{
	simulator_.simulate_molecules();

	//To Do: Implement Lennard Jones
	//Bounce off walls of environment
	for (Molecule& molecule : molecules_) {
		glm::vec2 pos = molecule.get_pos();
		glm::vec2 vel = molecule.get_vel();
		double radius = molecule.get_radius();

		double dif_pos_x = pos_.x + dim_.x - pos.x;
		double dif_pos_y = pos_.y + dim_.y - pos.y;
		if (dif_pos_x >= dim_.x - radius || dif_pos_x <= radius) {
			vel.x *= -1;
		}
		if (dif_pos_y >= dim_.y - radius || dif_pos_y <= radius) {
			vel.y *= -1;
		}

		molecule.set_vel(vel);
	}
}


void Environment::load_molecules(int n, double radius, double vel_mag)
{
	for (int i = 0; i < n; i++) {
		//TO DO: make rand function in a helper file
		double rand_vel = (double)rand() / (float)RAND_MAX;
		double rand_x = (double)rand() / (float)RAND_MAX;
		double rand_y = (double)rand() / (float)RAND_MAX;

		double x = rand_x * (dim_.x - 2 * radius) + pos_.x + radius;
		double y = rand_y * (dim_.y - 2 * radius) + pos_.y + radius;
		glm::vec2 pos = glm::vec2(x, y);

		double theta = rand_vel * 2 * PI;
		double vel_x = cos(theta) * vel_mag;
		double vel_y = sin(theta) * vel_mag;
		glm::vec2 vel = glm::vec2(vel_x, vel_y);

		Molecule molecule = Molecule(pos, vel, radius, ofColor::darkGray, ofColor::black);
		molecules_.push_back(molecule);
	}
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