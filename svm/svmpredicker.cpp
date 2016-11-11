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
    if(argc != 3)
      std::cout << "Please input test file path, xml\n"; 
    else{
      fin.open(argv[1]);
      xmlfilename = argv[2];
    }
    vector< vector<float>  > VecTestData;
    std::string line;
    int seg = 10;
    while(std::getline(fin, line)){//raw data format "length rawdata"
      std::stringstream linestream(line);
      int lengthdata;
      linestream >> lengthdata;
      vector<float> Vecsum;
      for (int k = 0; k < seg; ++k) {
        float sum=0;
        for (int i = k*lengthdata/seg; i < (k+1)*lengthdata/seg; ++i) {
          float dtemp;
          linestream >> dtemp;
          sum+=dtemp;
        }
        Vecsum.push_back(sum);
      }
      VecTestData.push_back(Vecsum);
    }
   
    float **TestData = new float*[VecTestData.size()];
    for (int i = 0; i < VecTestData.size(); ++i) {
      TestData[i] = (float*)VecTestData.at(i).data();
    }

    Mat TestDataMat(VecTestData.size(), seg, CV_32FC1, TestData);

    // Set up SVM's parameters

    // Load the SVM
    Ptr<SVM> svm = SVM::create();
    svm = SVM::load<SVM>(xmlfilename);
    //predict sampleMat
    //
    Mat res;
    svm->predict(TestDataMat, res);
    // Show the Test data
    std::cout << res.size() << std::endl;
    fin.close(); 
    return 0;
}
