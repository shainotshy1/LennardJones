#pragma once

#include "ofMain.h"

struct Shape 
{
private:
	//member variables
	ofColor fill_clr_;
	ofColor stroke_clr_;
	double radius_;
	double stroke_width_;

public:
	//constructor
	Shape(double radius, 
		double stroke_width, 
		ofColor fill_clr_ = ofColor::lightGray, 
		ofColor stroke_clr = ofColor::black);

	//destructor
	~Shape();

	//methods
};