//********************************************
// 图像预处理工作，主要用于训练时候图像的扩增
// crop ，倾斜矫正，光照预处理等操作。
//********************************************

#include "utils.h"

using namespace cv;
using namespace std;



// 对检测到人脸进行crop ，crop依据
//矫正后的人眼中心到右眼中心的距离
//distance = rightEye_center.x- eye_center.x
const int crop_vertical_up = 1.3;
const int crop_vectical_down = 3.2;
const int crop_horizontal_left = 1.2;
const int crop_horizontal_right = 1.2;
const int landmark_points_nums = 5;

//**************************************************
// Method: 人脸检测程序入口
// FullName: faceDetect    
// Returns: 
// Parameter: 输入图片
// Timer:2017.6.5
// others : 仿射变换后的图像与输入图像大小相同,5点检测
//***************************************************
void faceDetect(Mat& img){

}


//***************************************************
// Method: 人脸矫正，依据双眼的距离
// FullName: warpFace    
// Returns: 
// Parameter: 输入人脸图像和仿射变换后q的人脸图像 
// Timer:2017.6.5
// others : 仿射变换后的图像与输入图像大小相同,5点检测
//*****************************************************
void  warpFace(Mat& face, const FacePts& pts, FacePts& rotation_pts, Mat& warp_face){
	
	if (face.empty()){
		cout << "warpFace ERROR!: the input face is empty;  (utils.cpp)\n ";
		return ;
	}

	//图像copy
	Mat rotation_face;
	if (face.channels() == 3){
		cvtColor(face, rotation_face, CV_BGR2GRAY);
	}
	else
		rotation_face = face.clone();

	//求取左眼右眼和双眼中心
	Point2f left_eye(pts.x[0],pts.y[0]);
	Point2f right_eye(pts.x[1], pts.y[1]);
	Point2f center_eye;
	center_eye.x = left_eye.x + (right_eye.x-left_eye.x)/2.0; 
	center_eye.y = left_eye.y + (right_eye.y-left_eye.y)/2.0; 

 	//dy为正时头像左偏，顺时针旋转angle调整图像
	double dy = right_eye.y - left_eye.y;
	double dx = right_eye.x - left_eye.x;
	double len = sqrtf(dx*dx + dy*dy);
	double angle = atan2f(dy, dx) * 180 / CV_PI;

	//确定旋转后的图像大小
	//中心点，旋转角度和图像缩放系数
	Mat roi_mat = getRotationMatrix2D(center_eye, angle, 1.0);
	warpAffine(rotation_face, warp_face, roi_mat, cvSize(rotation_face.cols, rotation_face.rows));

	//vector<Point2f> marks;
	//按照仿射变换矩阵，计算变换后各关键点在新图中所对应的位置坐标。
	for (int index = 0; index < landmark_points_nums; index++)
	{
		rotation_pts.x[index] = roi_mat.ptr<double>(0)[0] * pts.x[index] + roi_mat.ptr<double>(0)[1] * pts.y[index] + roi_mat.ptr<double>(0)[2];
		rotation_pts.y[index] = roi_mat.ptr<double>(1)[0] * pts.x[index] + roi_mat.ptr<double>(1)[1] * pts.y[index] + roi_mat.ptr<double>(1)[2];
	}
}







