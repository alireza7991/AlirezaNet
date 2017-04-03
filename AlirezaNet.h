//
// Created by alirezafn on 4/2/17.
//

#ifndef ALIREZANET_ALIREZANET_H
#define ALIREZANET_ALIREZANET_H

#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <vector>
#include <cmath>
#include <iostream>

namespace Alireza {

    using namespace Eigen;
    typedef Matrix<float, Dynamic, Dynamic> tensor;

    using namespace std;

    tensor _sig(tensor input) {
        tensor tmp = input;
        for (unsigned i = 0; i < tmp.rows(); i++) {
            for (unsigned j = 0; j < tmp.cols(); j++) {
                tmp(i, j) = (1 / (1 + std::exp(-tmp(i, j))));
            }
        }
        return tmp;
    }

    tensor _sig_d(tensor input) {
        tensor tmp = input;
        for (unsigned i = 0; i < tmp.rows(); i++) {
            for (unsigned j = 0; j < tmp.cols(); j++) {
                tmp(i, j) = tmp(i, j) * (1 - tmp(i, j));
            }
        }
        return tmp;
    }

    class NueralNet {
    public:
        NueralNet(float momentum, float learningRate, unsigned inputSize, unsigned outputSize, unsigned hiddenSize) {
            this->momentum = momentum;
            this->learningRate = learningRate;
            this->hiddenSize = hiddenSize;
            this->outputSize = outputSize;
            this->inputSize = inputSize;
            l1Weights.resize(hiddenSize + 1, inputSize + 1);
            w1Update.resize(hiddenSize + 1, inputSize + 1);
            input.resize(inputSize + 1, 1);
            w2Update.resize(outputSize, hiddenSize + 1);
            l2Weights.resize(outputSize, hiddenSize + 1);
            w1Update.setZero();
            w2Update.setZero();
            l2Weights.setRandom();
            l2Weights.setRandom();
        }

        void train(vector<tensor> sample, vector<tensor> groundTruth, unsigned epoch) {
            assert(sample.size() == groundTruth.size());
            for (int i = 0; i < epoch; i++) {
                for (int j = 0; j < sample.size(); j++) {
                    std::cout << "training sample ..." << std::endl;
                    train_one(sample[j], groundTruth[j]);
                }
            }
        }

        void train_one(tensor &sample, tensor &groundTruth) {
            forward(sample);
            backward(groundTruth);
        }

        tensor test(tensor sample) {
            forward(sample);
            return output;
        }

    private:
        float momentum;
        float learningRate;
        unsigned inputSize, outputSize, hiddenSize;
        tensor l1Weights, l2Weights;
        tensor w1Update, w2Update;
        tensor input, output, l1Out;
        tensor l1Error, l2Error;

        tensor forward(tensor &sample) {
            for (long i = 0; i < sample.rows(); ++i)
                input(i) = sample(i);
            input(input.rows() - 1) = -1;
            l1Out = _sig(l1Weights * input);
            l1Out(l1Out.rows() - 1) = -1;
            output = l2Weights * l1Out;
        }

        void backward(tensor &groundTruth) {
            l2Error = Eigen::MatrixXf::Ones(outputSize, 1) - output;
            l2Error.cwiseProduct(output);
            tensor totalError = groundTruth - output;
            l2Error.cwiseProduct(totalError);
            l1Error = Eigen::MatrixXf::Ones(hiddenSize + 1, 1) - l1Out;
            l1Error.cwiseProduct(l1Out);
            l1Error.cwiseProduct(l2Weights.transpose() * l2Error);
            w2Update = this->learningRate * l2Error * l1Out.transpose() + momentum * w2Update;
            w1Update = this->learningRate * l1Error * input.transpose() + momentum * w1Update;
            l1Weights += w1Update;
            l2Weights += w2Update;
        }

    };


}

#endif //ALIREZANET_ALIREZANET_H
