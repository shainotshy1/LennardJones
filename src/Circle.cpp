#include "Circle.h"

Circle::Circle(glm::vec2 pos, double radius, double stroke_width, ofColor fill_clr_, ofColor stroke_clr) :
	Shape(pos, radius, stroke_width, fill_clr_, stroke_clr)
{
}

Circle::~Circle()
{
}

void Circle::display() const
{
}
