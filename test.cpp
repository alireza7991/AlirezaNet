#include <iostream>
#include <Eigen/Dense>
#include "AlirezaNet.h"

using namespace Alireza;
using namespace std;
using namespace Eigen;

int main() {
    NueralNet net(0.8, 0.01, 2, 1, 10);
    tensor x1(2, 1), x2(2, 1), x3(2, 1), x4(2, 1);
    tensor y1(1, 1), y2(1, 1), y3(1, 1), y4(1, 1);
    x1(0, 0) = 0;
    x1(1, 0) = 0;
    y1(0, 0) = 0;
    x2(0, 0) = 1;
    x2(1, 0) = 0;
    y2(0, 0) = 1;
    x3(0, 0) = 0;
    x3(1, 0) = 1;
    y3(0, 0) = 1;
    x4(0, 0) = 1;
    x4(1, 0) = 1;
    y4(0, 0) = 0;
    vector<tensor> sample, groundTruth;
    sample.push_back(x1);
    groundTruth.push_back(y1);
    sample.push_back(x2);
    groundTruth.push_back(y2);
    sample.push_back(x3);
    groundTruth.push_back(y3);
    sample.push_back(x4);
    groundTruth.push_back(y4);
    // train!
    cout << "Training ... " << endl;
    net.train(sample, groundTruth, 100);
    cout << "Testing ... " << endl;
    tensor e = net.test(x1);
    cout << "0 , 0 => " << e(0, 0) << endl;
    e = net.test(x2);
    cout << "1 , 0 => " << e(0, 0) << endl;
    e = net.test(x3);
    cout << "0 , 1 => " << e(0, 0) << endl;
    e = net.test(x4);
    cout << "1 , 1 => " << e(0, 0) << endl;


}