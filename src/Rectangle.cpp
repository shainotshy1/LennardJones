#include "Rectangle.h"

Rectangle::Rectangle(glm::vec2 pos, double radius, double stroke_width, ofColor fill_clr_, ofColor stroke_clr) :
	Shape(pos, radius, stroke_width, fill_clr_, stroke_clr)
{
}

Rectangle::~Rectangle()
{
}

void Rectangle::display() const
{
}
