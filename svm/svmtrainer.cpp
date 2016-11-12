/************************************************************************************//**
 *  @file       svmtrainer.cpp
 *
 *  @brief      Brief descriptinon of svm.cpp 
 *
 *  @date       2016-11-11 13:46
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
    // Set up training data
    std::ifstream fin ;
    std::string old_xml;
    if(argc == 3){
      old_xml = argv[2];
      fin.open(argv[1]);
    }
    else if(argc != 2){
      std::cout << "Please input path\n"; 
      return -1;
    }
    else
      fin.open(argv[1]);

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

    /* std::cout << labelsMat << std::endl; */
    /* std::cout <<  std::endl; */
    /* std::cout <<  std::endl; */
    /* std::cout <<  std::endl; */
    /* std::cout << trainingDataMat << std::endl; */
    // Set up SVM's parameters

    // Train the SVM
    Ptr<SVM> svm; 
    if(argc != 3){
      svm = SVM::create();
      svm->setType(SVM::C_SVC);
      svm->setDegree(3);
      svm->setKernel(SVM::RBF);
      svm->train(trainingDataMat,ROW_SAMPLE,labelsMat); 
      
      svm->save("stepSVM.xml");
    }
    else{
      svm = SVM::load<SVM>(old_xml);
      svm->train(trainingDataMat,ROW_SAMPLE,labelsMat); 
      
      svm->save("stepSVM.xml");
    }
    //predict sampleMat
    //
    /* Mat res; */
    /* svm->predict(sampleMat, res); */
    /* // Show the training data */
    /* std::cout << res.size() << std::endl; */
    fin.close(); 
    return 0;
}
