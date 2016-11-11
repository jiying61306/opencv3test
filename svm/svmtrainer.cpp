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
    if(argc != 2)
      std::cout << "Please input path\n"; 
    else
      fin.open(argv[1]);

    vector<int> vectorlabel;
    vector< vector<double> > VectrainingData;
    std::string line;
    int seg = 10;
    while(std::getline(fin, line)){//raw data format "label length rawdata"
      std::stringstream linestream(line);
      int templabel, lengthdata;
      linestream >> templabel;
      linestream >> lengthdata;
      vectorlabel.push_back(templabel);
      vector<double> Vecsum;
      for (int k = 0; k < seg; ++k) {
        double sum=0;
        for (int i = k*lengthdata/seg; i < (k+1)*lengthdata/seg; ++i) {
          double dtemp;
          linestream >> dtemp;
          sum+=dtemp;
        }
        Vecsum.push_back(sum);
      }
      VectrainingData.push_back(Vecsum);
    }
   
    int *labels = vectorlabel.data();
    float **trainingData = new float*[VectrainingData.size()];
    for (int i = 0; i < VectrainingData.size(); ++i) {
      trainingData[i] = (float*)VectrainingData.at(i).data();
    }

    Mat labelsMat(vectorlabel.size(), 1, CV_32SC1, labels);

    Mat trainingDataMat(vectorlabel.size(), seg, CV_32FC1, trainingData);

    // Set up SVM's parameters

    // Train the SVM
    Ptr<SVM> svm = SVM::create();

    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->train(trainingDataMat,ROW_SAMPLE,labelsMat); 
    
    svm->save("stepSVM.xml");
    //predict sampleMat
    //
    /* Mat res; */
    /* svm->predict(sampleMat, res); */
    /* // Show the training data */
    /* std::cout << res.size() << std::endl; */
    fin.close(); 
    return 0;
}
