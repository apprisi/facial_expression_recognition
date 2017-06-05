#include <opencv2/opencv.hpp>
#include <iostream>
#include "register.h"
#include "face_detection.h"
using namespace std;
using namespace cv;

string proto_model_dir = "../models/faceDet/";



int main(){

	MTCNN detector(proto_model_dir);//»À¡≥ºÏ≤‚model
	




	Mat img = imread("E:/others/CK/train/Disgust/S011_005_03141026.png");
	cout << img.rows << "X" << img.cols << " " << img.channels() << endl;
	imshow("CK+ Image", img);
	waitKey(0);
	return 0;

}