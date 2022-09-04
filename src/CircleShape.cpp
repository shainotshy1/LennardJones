#include "CircleShape.h"

CircleShape::CircleShape(glm::vec2 pos, glm::vec2 dim, ofColor fill_clr_, ofColor stroke_clr, double stroke_width) :
	Shape(pos, dim, fill_clr_, stroke_clr, stroke_width)
{
}

CircleShape::~CircleShape()
{
}

void CircleShape::draw_shape(ofColor clr) const
{
	ofSetColor(clr);
	ofDrawCircle(pos_, dim_.x);
}
