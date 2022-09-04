#pragma once

#include "Shape.h"

struct CircleShape : Shape
{
	//constructor
	CircleShape(glm::vec2 pos,
		glm::vec2 dim,
		double stroke_width = 1,
		ofColor fill_clr_ = ofColor::lightGray,
		ofColor stroke_clr = ofColor::black);

	//destructor
	~CircleShape();

private:

	// Inherited via Shape
	virtual void draw_shape(ofColor) const override;
};