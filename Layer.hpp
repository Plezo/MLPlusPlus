#pragma once

#include <vector>

using namespace std;

class Layer {
public:
    Layer(int input_size, int output_size) {
        weights = vector<vector<double>>(input_size, vector<double>(output_size));
        biases = vector<vector<double>>(1, vector<double>(output_size));
    }

    // calculates the output of the layer
    vector<vector<double>> calculateOutput(vector<vector<double>> inputs);
    

private:
    vector<vector<double>> weights;
    vector<vector<double>> biases;
};