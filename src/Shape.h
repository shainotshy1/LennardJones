#pragma once

#include "ofMain.h"

struct Shape 
{
	//member variables
	ofColor fill_clr_;
	ofColor stroke_clr_;
	glm::vec2 pos_;
	double radius_;
	double stroke_width_;

	//constructor
	Shape(glm::vec2 pos,
		double radius, 
		double stroke_width, 
		ofColor fill_clr_ = ofColor::lightGray, 
		ofColor stroke_clr = ofColor::black);

	//destructor
	~Shape();

	//virtual methods
	virtual void display() const = 0;
};