#pragma once

#include <vector>
#include <string>
#include <random>
#include <math.h>
#include <iostream>

using namespace std;

using matrix = vector<vector<double>>;

class Layer {
public:
    Layer(int input_size, int output_size);

    // calculates the output of the layer
    matrix forward(matrix inputs);
    matrix backward(matrix inputs, matrix output_gradient, double learning_rate);
    

private:
    matrix weights;
    matrix biases;

    // TODO: Make this a class
    string activation_function = "sigmoid";
};