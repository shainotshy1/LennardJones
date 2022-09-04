#pragma once

#include "Shape.h"

struct Circle : Shape
{
	//constructor
	Circle(glm::vec2 pos,
		double radius,
		double stroke_width,
		ofColor fill_clr_ = ofColor::lightGray,
		ofColor stroke_clr = ofColor::black);

	//destructor
	~Circle();

	// Inherited via Shape
	virtual void display() const override;
};