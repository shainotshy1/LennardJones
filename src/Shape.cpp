#include "Shape.h"

Shape::Shape(double radius, double stroke_width, ofColor fill_clr, ofColor stroke_clr) 
	: radius_{ radius },
	stroke_width_{ stroke_width },
	fill_clr_{ fill_clr },
	stroke_clr_{ stroke_clr }
{
}

Shape::~Shape()
{
}
