//********************************************
// ͼ��Ԥ����������Ҫ����ѵ��ʱ��ͼ�������
// crop ����б����������Ԥ����Ȳ�����
//********************************************

#include "utils.h"

using namespace cv;
using namespace std;



// �Լ�⵽��������crop ��crop����
//��������������ĵ��������ĵľ���
//distance = rightEye_center.x- eye_center.x
const int crop_vertical_up = 1.3;
const int crop_vectical_down = 3.2;
const int crop_horizontal_left = 1.2;
const int crop_horizontal_right = 1.2;
const int landmark_points_nums = 5;

//**************************************************
// Method: �������������
// FullName: faceDetect    
// Returns: 
// Parameter: ����ͼƬ
// Timer:2017.6.5
// others : ����任���ͼ��������ͼ���С��ͬ,5����
//***************************************************
void faceDetect(Mat& img){

}


//***************************************************
// Method: ��������������˫�۵ľ���
// FullName: warpFace    
// Returns: 
// Parameter: ��������ͼ��ͷ���任��q������ͼ�� 
// Timer:2017.6.5
// others : ����任���ͼ��������ͼ���С��ͬ,5����
//*****************************************************
void  warpFace(Mat& face, const FacePts& pts, FacePts& rotation_pts, Mat& warp_face){
	
	if (face.empty()){
		cout << "warpFace ERROR!: the input face is empty;  (utils.cpp)\n ";
		return ;
	}

	//ͼ��copy
	Mat rotation_face;
	if (face.channels() == 3){
		cvtColor(face, rotation_face, CV_BGR2GRAY);
	}
	else
		rotation_face = face.clone();

	//��ȡ�������ۺ�˫������
	Point2f left_eye(pts.x[0],pts.y[0]);
	Point2f right_eye(pts.x[1], pts.y[1]);
	Point2f center_eye;
	center_eye.x = left_eye.x + (right_eye.x-left_eye.x)/2.0; 
	center_eye.y = left_eye.y + (right_eye.y-left_eye.y)/2.0; 

 	//dyΪ��ʱͷ����ƫ��˳ʱ����תangle����ͼ��
	double dy = right_eye.y - left_eye.y;
	double dx = right_eye.x - left_eye.x;
	double len = sqrtf(dx*dx + dy*dy);
	double angle = atan2f(dy, dx) * 180 / CV_PI;

	//ȷ����ת���ͼ���С
	//���ĵ㣬��ת�ǶȺ�ͼ������ϵ��
	Mat roi_mat = getRotationMatrix2D(center_eye, angle, 1.0);
	warpAffine(rotation_face, warp_face, roi_mat, cvSize(rotation_face.cols, rotation_face.rows));

	//vector<Point2f> marks;
	//���շ���任���󣬼���任����ؼ�������ͼ������Ӧ��λ�����ꡣ
	for (int index = 0; index < landmark_points_nums; index++)
	{
		rotation_pts.x[index] = roi_mat.ptr<double>(0)[0] * pts.x[index] + roi_mat.ptr<double>(0)[1] * pts.y[index] + roi_mat.ptr<double>(0)[2];
		rotation_pts.y[index] = roi_mat.ptr<double>(1)[0] * pts.x[index] + roi_mat.ptr<double>(1)[1] * pts.y[index] + roi_mat.ptr<double>(1)[2];
	}
}







