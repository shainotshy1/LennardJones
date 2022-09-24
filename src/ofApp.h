#pragma once

#include "ofMain.h"
#include "Environment.h"
#include "MoleculeSimulator.cuh"

class ofApp : public ofBaseApp{

private:
	//member variables
	glm::vec2 border_pos_;
	glm::vec2 border_dim_;
	MoleculeSimulator* mol_simulator_;
	vector<Environment> environments_;

public:
	//methods
	void setup();
	void exit();
	void set_window();
	void create_environments(int n);
	void update();
	void draw_window();
	void draw_border();
	void draw_environments();
	void draw();

	void keyPressed(int key);
	void keyReleased(int key);
	void mouseMoved(int x, int y );
	void mouseDragged(int x, int y, int button);
	void mousePressed(int x, int y, int button);
	void mouseReleased(int x, int y, int button);
	void mouseEntered(int x, int y);
	void mouseExited(int x, int y);
	void windowResized(int w, int h);
	void dragEvent(ofDragInfo dragInfo);
	void gotMessage(ofMessage msg);
		
};
