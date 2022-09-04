#pragma once

#include "Shape.h"

struct Rectangle : Shape
{
	//constructor
	Rectangle(glm::vec2 pos,
		double radius,
		double stroke_width,
		ofColor fill_clr_ = ofColor::lightGray,
		ofColor stroke_clr = ofColor::black);

	//destructor
	~Rectangle();

	// Inherited via Shape
	virtual void display() const override;
};