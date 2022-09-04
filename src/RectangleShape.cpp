#include "RectangleShape.h"

RectangleShape::RectangleShape(glm::vec2 pos, glm::vec2 dim, double stroke_width, ofColor fill_clr_, ofColor stroke_clr) :
	Shape(pos, dim, stroke_width, fill_clr_, stroke_clr)
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
