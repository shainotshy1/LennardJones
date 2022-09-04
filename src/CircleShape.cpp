#include "CircleShape.h"

CircleShape::CircleShape(glm::vec2 pos, glm::vec2 dim, double stroke_width, ofColor fill_clr_, ofColor stroke_clr) :
	Shape(pos, dim, stroke_width, fill_clr_, stroke_clr)
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
