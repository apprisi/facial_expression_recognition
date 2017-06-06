#ifndef _UTILS_H__
#define _UTILS_H__

#include <opencv2/opencv.hpp>
#include "face_detection.h"
#include <fstream>
#include <io.h>


void faceDetect(cv::Mat img);
void  warpFace(cv::Mat& face, cv::Mat& warp_face,const FacePts& pts, FacePts& rotation_pts);
void setUpGausssian(int kernel_width, int kernel_height, std::vector<std::vector<float>>& gauss_matrix);
void data_augmentation(cv::Mat &face, FacePts& pts, vector<cv::Mat>& SyntheticImg);
void train();
void getFiles(std::string path, std::vector<std::string>  &files);
#endif