#include "RectangleShape.h"

RectangleShape::RectangleShape(glm::vec2 pos, glm::vec2 dim, ofColor fill_clr_, ofColor stroke_clr, double stroke_width) :
	Shape(pos, dim, fill_clr_, stroke_clr, stroke_width)
{
}

RectangleShape::~RectangleShape()
{
}

void RectangleShape::draw_shape(ofColor clr) const
{
	ofSetColor(clr);
	ofDrawRectangle(pos_, dim_.x, dim_.y);
}
