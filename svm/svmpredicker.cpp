/************************************************************************************//**
 *  @file       svmpredicker.cpp
 *
 *  @brief      Brief descriptinon of svmpredicker.cpp 
 *
 *  @date       2016-11-11 20:30
 *
 ***************************************************************************************/
#include<iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include<fstream>
#include<sstream>
#include<string>
using namespace cv;
using namespace cv::ml;
using namespace std;

int main(int argc, char** argv)
{
    // Set up Test data
    std::string xmlfilename;
    std::ifstream fin ;
    if(argc != 3){
      std::cout << "Please input test file path, xml\n"; 
      return -1;
    }
    else{
      fin.open(argv[1]);
      xmlfilename = argv[2];
    }
    vector<int> vectorlabel;
    vector< vector<float> > VectrainingData;
    std::string line;
    int seg = 11;
    Mat trainingDataMat(0 , seg, CV_32FC1);
    int a =0;
    while(std::getline(fin, line)){//raw data format "label length rawdata"
      std::stringstream linestream(line);
      int templabel, lengthdata;
      linestream >> templabel;
      linestream >> lengthdata;
      vectorlabel.push_back(templabel);
      vector<float> Vecsum;
      for (int k = 0; k < seg; ++k) {
        float sum=0;
        for (int i = k*lengthdata/seg; i < (k+1)*lengthdata/seg; ++i) {
          float dtemp;
          linestream >> dtemp;
          sum+=(dtemp*(k-seg/2))*(dtemp*(k-seg/2));
        }
        Vecsum.push_back(sum);
        /* std::cout << sum <<"    "<< a  << std::endl; */
      }
      Mat temp(1, seg, CV_32FC1);
      memcpy(temp.data , Vecsum.data(), Vecsum.size()*sizeof(float));
      trainingDataMat.push_back(temp);
      a++;
    }
   

    Mat labelsMat(vectorlabel.size(), 1, CV_32SC1);
    memcpy(labelsMat.data, vectorlabel.data(), vectorlabel.size()*sizeof(int));

    // Set up SVM's parameters

    // Load the SVM
    Ptr<SVM> svm = SVM::create();
    svm = SVM::load<SVM>(xmlfilename);
    //predict sampleMat
    //
    Mat res;
    svm->predict(trainingDataMat, res);
    // Show the Test data

    /* labelsMat.push_back(res); */
    std:: cout <<"answer:" <<  labelsMat << std::endl;
    std:: cout <<"result:\n" << res << std::endl;
    float testfloat = 10;
    fin.close(); 
    return 0;
}
