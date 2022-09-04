#pragma once

#include "ofMain.h"

struct Shape 
{
	//member variables
	ofColor fill_clr_;
	ofColor stroke_clr_;
	glm::vec2 pos_;
	glm::vec2 dim_;
	double stroke_width_;

	//constructor
	Shape(glm::vec2 pos,
		glm::vec2 dim, 
		ofColor fill_clr_, 
		ofColor stroke_clr,
		double stroke_width);

	//destructor
	~Shape();

	//methods
	void display() const;

private:

	//private virtual methods
	virtual void draw_shape(ofColor) const = 0;
};