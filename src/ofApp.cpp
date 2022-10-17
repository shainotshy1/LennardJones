#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
	set_window();
	create_environments(16);
}
void ofApp::exit()
{
	delete mol_simulator_;
}
//--------------------------------------------------------------
void ofApp::create_environments(int n)
{
	environments_ = vector<Environment>{};

	int corner_x = border_pos_.x;
	int corner_y = border_pos_.y;

	double sqrt_n = sqrt(n);
	int rows = ceil(sqrt_n);
	int* cols_per_row = new int[rows];
	
	int dif = n - rows; //number of additional environments needed to be created
	for (int i = 0; i < rows; i++) {
		int min_cols_per_row = dif / rows + 1;
		int extra_col = (dif - ((min_cols_per_row - 1) * rows)) >= i + 1 ? 1 : 0; //additional for modulating noneven num of environments
		int cols = min_cols_per_row + extra_col;

		cols_per_row[i] = cols;
	}

	int max_molecules = 400;
	int current_cell = 1;
	for (int i = 0; i < rows; i++) {

		int cols = cols_per_row[i];
		for (int j = 0; j < cols; j++) {
			int w = border_dim_.x / cols_per_row[0];
			int h = border_dim_.y / rows;

			//centers environments in row with empty columns
			int empty_cols = cols_per_row[0] - cols;
			int empty_space = empty_cols * w;

			int x = corner_x + j * w + empty_space / 2;
			int y = corner_y + i * h;

			glm::vec2 pos = glm::vec2(x, y);
			glm::vec2 dim = glm::vec2(w, h);
			double density = current_cell * 1.0 / n;

			Environment env = Environment(pos, dim, max_molecules, density);
			env.set_background_clr(ofColor::lightYellow);
			environments_.push_back(env);
			
			current_cell++;
		}
	}

	delete[] cols_per_row;

	mol_simulator_ = new MoleculeSimulator(border_dim_, border_pos_, environments_, 5);
}

//--------------------------------------------------------------
void ofApp::set_window()
{
	double window_scale = 0.8;
	int window_w = ofGetScreenHeight() * window_scale;
	int window_h = ofGetScreenHeight() * window_scale;
	ofSetWindowShape(window_w, window_h);
	ofSetWindowPosition(ofGetScreenWidth() / 2 - window_w / 2,
	ofGetScreenHeight() / 2 - window_h / 2);

	double border_scale = 0.98;
	int border_w = ofGetWindowWidth() * border_scale;
	int border_h = ofGetWindowHeight() * border_scale;
	int x = (ofGetWindowWidth() -  border_w) / 2;
	int y = (ofGetWindowHeight() - border_h) / 2;

	border_dim_ = glm::vec2(border_w, border_h);
	border_pos_ = glm::vec2(x, y);
}


//--------------------------------------------------------------
void ofApp::update() {
	mol_simulator_->simulate_molecules();
}
//--------------------------------------------------------------
void ofApp::draw_environments() 
{
	mol_simulator_->update_environments();
}

//--------------------------------------------------------------
void ofApp::draw_window()
{
	ofSetBackgroundColor(ofColor::white);
}

//--------------------------------------------------------------
void ofApp::draw_border() 
{
	ofNoFill();
	ofSetColor(ofColor::black);
	ofSetLineWidth(5);
	ofSetRectMode(OF_RECTMODE_CORNER);
	ofDrawRectangle(border_pos_.x, border_pos_.y,
		border_dim_.x, border_dim_.y);
}

//--------------------------------------------------------------
void ofApp::draw(){
	draw_window();
	draw_environments();
	draw_border();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
