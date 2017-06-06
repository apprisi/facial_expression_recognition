//********************************************
// ͼ��Ԥ����������Ҫ����ѵ��ʱ��ͼ�������
// crop ����б����������Ԥ����Ȳ�����
//********************************************

#include "utils.h"
#include "LightPrep.h"
#include <opencv/cv.h>

using namespace cv;
using namespace std;

//��ά��˹�����ľ�ֵ�ͱ�׼��
const float gauss_delta = 1;
const int gauss_ave = 0;

//mtcnnģ���ļ�����·��
string proto_model_dir = "E:/laboratory/FaceEmoCNN/models/faceDet";

// �Լ�⵽��������crop ��crop����
//��������������ĵ��������ĵľ���
//distance = rightEye_center.x- eye_center.x
const float crop_vertical_up = 1.3;
const float crop_vectical_down = 3.2;
const float crop_horizontal_left = 1.3;
const float crop_horizontal_right = 1.3;
const int landmark_points_nums = 5;


//��һ���ߴ��С 48*48
const int norm_width = 48;
const int norm_height = 48;


//**************************************************
// Method: ������˹��
// FullName: setUpGausssian    
// Returns: ��˹�˿�ȣ��߶Ⱥͱ���ĸ�˹��ģ�����
// Parameter: ����ͼƬ
// Timer:2017.6.5
// others : ����任���ͼ��������ͼ���С��ͬ,5����
void setUpGausssian(int kernel_height, int kernel_width, vector<vector<float>>& gauss_matrix){
	float temp_val = 1 / (2.0*CV_PI*gauss_delta*gauss_delta);
	gauss_matrix.resize(kernel_height);
	for (int i = 0; i < kernel_height; i++){
		gauss_matrix[i].resize(kernel_width);
		for (int j = 0; j < kernel_width; ++j){
			gauss_matrix[i][j] = temp_val*exp(-(pow(i - kernel_height / 2, 2) + pow(j - kernel_width / 2, 2))/(2*gauss_delta*gauss_delta));
		}
	}
}


//**************************************************
// Method: ͼ��ǿ�ȹ�һ������
// FullName: intensityNormalization    
// Returns: 
// Parameter: ����ͼ�񣬹�һ��ͼ�񣬸�˹��
// Timer:2017.6.5
// others : ����任���ͼ��������ͼ���С��ͬ,5����
//***************************************************
void intensityNormalization(Mat& src, Mat& normImg, const vector<vector<float>>gauss_matrix){
	if (src.rows != norm_height || src.cols != norm_width || gauss_matrix.empty()){//ͼ��ߴ粻��48*48
		cout << "image size should not be 48*48 in intensityNormalization ,utils.cpp.\n";
		return;
	}
	float sum,delte,sum_ori;
	int window_size = gauss_matrix.size()*gauss_matrix[0].size();
	for (int i = gauss_matrix.size() / 2; i < norm_height - gauss_matrix.size() / 2; ++i){
		for (int j = gauss_matrix[0].size() / 2; j < norm_width - gauss_matrix.size() / 2; ++j){
			sum = 0.0;
			delte = 0.0;
			sum_ori = 0;
			for (int ix = i-gauss_matrix.size()/2; ix < i+gauss_matrix.size()/2; ++ix){
				uchar* data = src.ptr<uchar>(ix);
				for (int jx = j - gauss_matrix[0].size() / 2; jx < j + gauss_matrix[0].size() / 2; ++jx){
					sum +=  gauss_matrix[ix - i + gauss_matrix.size() / 2][jx - j + gauss_matrix[0].size() / 2] * data[jx];
					delte +=  data[jx] * data[jx];
					sum_ori += data[jx];
				}
			}
			float average = sum ;
			float stdd = sqrt(delte / window_size - sum_ori/(window_size)*sum_ori/(window_size));
			normImg.at<uchar>(i, j) = (src.at<uchar>(i, j) - average) / stdd;
		}
	}
	
}


//***************************************************
// Method: ��������������˫�۵ľ���
// FullName: warpFace    
// Returns: 
// Parameter: ��������ͼ��ͷ���任�������ͼ�� 
// Timer:2017.6.5
// others : ����任���ͼ��������ͼ���С��ͬ,5����
//*****************************************************
void  warpFace(cv::Mat& face, cv::Mat& warp_face, FacePts& pts, FacePts& rotation_pts){
	
	if (face.empty()){
		cout << "warpFace ERROR!: the input face is empty;  (utils.cpp)\n ";
		return ;
	}

	for (int  i = 0; i < 5; i++){
		int temp = pts.x[i];
		pts.x[i] = pts.y[i];
		pts.y[i] = temp;
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
	for (int index = 0; index < landmark_points_nums; index++){
		rotation_pts.x[index] = roi_mat.ptr<double>(0)[0] * pts.x[index] + roi_mat.ptr<double>(0)[1] * pts.y[index] + roi_mat.ptr<double>(0)[2];
		rotation_pts.y[index] = roi_mat.ptr<double>(1)[0] * pts.x[index] + roi_mat.ptr<double>(1)[1] * pts.y[index] + roi_mat.ptr<double>(1)[2];
	}
}

//**********************************************************************
// Method:   ��ȡ�ļ����������ļ�
// FullName: getFiles   
// Returns:  void 
// Parameter: ����Ͷ�Ӧ��ģ��·��
// Timer:2017.5.10
//**********************************************************************
void getFiles(std::string path, std::vector<std::string>  &files) {
	struct _finddata_t filefind;
	intptr_t hfile = 0;
	std::string s;
	if ((hfile = _findfirst(s.assign(path).append("/*").c_str(), &filefind)) != -1) {
		do  {
			if (filefind.attrib == _A_SUBDIR) {
				if (strcmp(filefind.name, ".") && strcmp(filefind.name, "..")){
					getFiles(s.assign(path).append("/").append(filefind.name), files);
				}
			}
			else {//�жϺ�׺���Ƿ�ΪͼƬ����
				string temp = filefind.name;
				int pos = temp.find_last_of(".");
				temp = temp.substr(pos, temp.size() - pos);
				if (temp == ".tiff" || temp == ".bmp" || temp == ".png" || temp == "jpg"){
					files.push_back(s.assign(path).append("/").append(filefind.name));
				}
				else{
					;
				}
				
			}
		} while (_findnext(hfile, &filefind) == 0);
	} _findclose(hfile);
}

/*
#define GAUSSFUN(i,j,gauss_data,&y){ \
	float temp_val = 1 / (2.0*CV_PI*gauss_data*gauss_data); \
	float y = temp_val*exp(-(pow(i, 2) + pow(j, 2))/(2*gauss_data*gauss_data));\
}while (0)
*/

//***************************************************
// Method: ѵ����������
// FullName: data_augmentation    
// Returns: 
// Parameter: ����ͼ���������������꣬�������������,��������������
// Timer:2017.6.5
// others : ͨ���ı����������λ�ã���ͼ�������������������
//			�����ı���λ49�������Ҹ�ѡ7����
//*****************************************************
void data_augmentation(Mat& img,FacePts& pts,vector<FacePts>&syntheticPts){
	//����������
	Point2f left_center(pts.x[0],pts.y[0]);
	Point2f right_center(pts.x[1], pts.y[1]);
	//�Ÿ���
	vector<Point2f>Point2f_left;
	vector<Point2f>Point2f_right;
	int stride = 5;
	int augmentation_size = 7;
	for (int i = -stride; i <= stride; i += stride){
		for (int j = -stride; j <= stride; j += stride){
			if ((i == 0 && j != 0))
				continue;
			Point2f_left.push_back(Point2f(left_center.x + i, left_center.y + j));
			Point2f_right.push_back(Point2f(right_center.x + i, right_center.y + j));
		}
	}
	for (int i = 0; i < augmentation_size; ++i){
		for (int j = 0; j < augmentation_size; ++j){
			FacePts temp = pts;
			temp.x[0] = Point2f_left[i].x;
			temp.y[0] = Point2f_left[i].y;
			temp.x[1] = Point2f_right[j].x;
			temp.y[1] = Point2f_right[j].y;
			syntheticPts.push_back(temp);
		}
	}
}



//***************************************************
// Method: cnnѵ������׼������ں���
// FullName: train    
// Returns:   
// Parameter: 
// Timer:2017.6.5
// others : ������ѵ������������⣬��������crop��
//          ���������ǿ�ȹ�һ������Ĳ���=
//*****************************************************
void train(){
	MTCNN detector(proto_model_dir);//�������model
	double threshold[3] = { 0.6, 0.7, 0.7 }; //rpo net�Ա�
	double factor = 0.709;//�ߴ�����������ų߶�
	int minSize = 40;
	string image_dir = "C:\\Users\\lk\\Desktop\\test1";
	vector<string>files;
	//��ȡ��ǰĿ¼�µ������ļ�
	getFiles(image_dir, files);
	CLightPrep* light = new CLightPrep;
	light->Init(cvSize(norm_width, norm_height), true);
	Mat img;
	for (int i = 0; i < files.size(); ++i){
		string path = files[i];
		int pos = path.find_last_of(".");
		string save_dir = path.substr(0, pos);
		img = imread(path.c_str());
		if (img.empty())
			continue;
		vector<FaceInfo> faceInfos;
		detector.Detect(img, faceInfos, minSize, threshold, factor);
		//������˹��
		vector<vector<float>>gauss_matrix;
		setUpGausssian(7, 7, gauss_matrix);
		for (int j = 0; j < faceInfos.size(); ++j){
			vector<FacePts>syntheticPts;
			data_augmentation(img, faceInfos[j].facePts, syntheticPts);//�����������������Ҳ��Ҫ���з���仯���ߴ��һ����ǿ�ȹ�һ
			Mat warp_face = img.clone();
			FacePts rotation_info;
			//����������crop
			for (int k = 0; k < syntheticPts.size();++k){
				warpFace(img, warp_face, syntheticPts[k], rotation_info);
				int distance = 0.5*(rotation_info.x[1] - rotation_info.x[0]);//�����������۾�����
				Point2f center_point(rotation_info.x[0] + (rotation_info.x[1] - rotation_info.x[0]) / 2, rotation_info.y[0] + (rotation_info.y[1] - rotation_info.y[0]) / 2);
				Rect roi(max<int>(0, center_point.x - crop_horizontal_left*distance), max<int>(0, center_point.y - crop_vertical_up*distance), min<int>(img.cols - max<int>(0, center_point.x - crop_horizontal_left*distance),(crop_horizontal_left + crop_horizontal_right)*distance), min<int>(img.rows-max<int>(0, center_point.y - crop_vertical_up*distance), (crop_vectical_down + crop_vertical_up)*distance));
				Mat normImg = warp_face(roi).clone();
				resize(normImg, normImg, cvSize(norm_width, norm_height));
				IplImage ResizeROi = normImg;
				CvMat *temp = cvCreateMat(normImg.rows, normImg.cols,CV_8UC1);  //ע��height��width��˳��  
				cvConvert(&ResizeROi, temp);//��� 
				light->RunLightPrep(temp);
				Mat tmp(temp);
				imshow("dis", tmp);
				//imshow("dis", normImg);
				//GaussianBlur(normImg,normImg, Size(3,3), 0, 0);
				//equalizeHist(normImg, normImg);
				//imshow("final", normImg);
				//Mat itennorm = normImg.clone();
				//intensityNormalization(normImg, itennorm, gauss_matrix);
				//imshow("dis", itennorm);
				waitKey(0);
				string save_path = save_dir + "_" + to_string(k) + ".bmp";
				imwrite(save_path, normImg);
			}
		}
	}	
}




