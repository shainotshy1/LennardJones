#include "Shape.h"

Shape::Shape(glm::vec2 pos, glm::vec2 dim, double stroke_width, ofColor fill_clr, ofColor stroke_clr) 
	: pos_{ pos },
	dim_{ dim },
	stroke_width_{ stroke_width },
	fill_clr_{ fill_clr },
	stroke_clr_{ stroke_clr }
{
}

Shape::~Shape()
{
}

void Shape::display() const
{
	ofFill();
	ofSetLineWidth(stroke_width_);
	draw_shape(fill_clr_);
	ofNoFill();
	draw_shape(stroke_clr_);
}
