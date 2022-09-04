#include "Shape.h"

Shape::Shape(glm::vec2 pos, glm::vec2 dim, ofColor fill_clr, ofColor stroke_clr, double stroke_width)
	: pos_{ pos },
	dim_{ dim },
	fill_clr_{ fill_clr },
	stroke_clr_{ stroke_clr },
	stroke_width_{ stroke_width }
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
