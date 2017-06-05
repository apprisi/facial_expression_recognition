#ifndef _FACE_DETECTION_H__
#define _FACE_DETECTION_H__
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include "boost/make_shared.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "register.h"



//using caffe::Blob;
//using caffe::Caffe;
//using caffe::Datum;
//using caffe::Net;
//using caffe::Layer;
//using caffe::Solver;
//using caffe::shared_ptr;
//using caffe::string;
//using caffe::Timer;
//using caffe::vector;
//using caffe::ostringstream;
using namespace caffe;

typedef struct FaceRect {
	float x1;
	float y1;
	float x2;
	float y2;
	float score; /**< Larger score should mean higher confidence. */
} FaceRect;

//five points 
typedef struct FacePts {
	float x[5], y[5];
} FacePts;

//face Location and five points 
typedef struct FaceInfo {
	FaceRect bbox;
	cv::Vec4f regression;
	FacePts facePts;
	double roll;
	double pitch;
	double yaw;
} FaceInfo;


class MTCNN {
public:
	MTCNN();
	MTCNN(const string& proto_model_dir);

	void Detect(const cv::Mat& img, std::vector<FaceInfo> &faceInfo, int minSize, double* threshold, double factor);

private:
	void LoadModle(const string& proto_model_dir);
	bool CvMatToDatumSignalChannel(const cv::Mat& cv_mat, Datum* datum);
	void Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels);
	void WrapInputLayer(std::vector<cv::Mat>* input_channels, Blob<float>* input_layer,const int height, const int width);
	void SetMean();
	void GenerateBoundingBox(Blob<float>* confidence, Blob<float>* reg,float scale, float thresh, int image_width, int image_height);
	void ClassifyFace(const std::vector<FaceInfo>& regressed_rects, cv::Mat &sample_single,boost::shared_ptr<Net<float> >& net, double thresh, char netName);
	void ClassifyFace_MulImage(const std::vector<FaceInfo> &regressed_rects, cv::Mat &sample_single,boost::shared_ptr<Net<float> >& net, double thresh, char netName);
	std::vector<FaceInfo> NonMaximumSuppression(std::vector<FaceInfo>& bboxes, float thresh, char methodType);
	void Bbox2Square(std::vector<FaceInfo>& bboxes);
	void Padding(int img_w, int img_h);
	std::vector<FaceInfo> BoxRegress(std::vector<FaceInfo> &faceInfo_, int stage);
	void RegressPoint(const std::vector<FaceInfo>& faceInfo);

private:
	boost::shared_ptr<Net<float> > PNet_;
	boost::shared_ptr<Net<float> > RNet_;
	boost::shared_ptr<Net<float> > ONet_;

	// x1,y1,x2,t2 and score
	std::vector<FaceInfo> condidate_rects_;
	std::vector<FaceInfo> total_boxes_;
	std::vector<FaceInfo> regressed_rects_;
	std::vector<FaceInfo> regressed_pading_;

	std::vector<cv::Mat> crop_img_;
	int curr_feature_map_w_;
	int curr_feature_map_h_;
	int num_channels_;
};


#endif