#include "Layer.hpp"

matrix dot(const matrix &a, const matrix &b) {
    int m = a.size(), n = b[0].size(), p = b.size();
    matrix res = matrix(m, vector<double>(n));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++ j) {
            res[i][j] = 0;
            for (int k = 0; k < p; ++k) {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return res;
}

matrix transpose(const matrix &arr) {
    int m = arr.size(), n = arr[0].size();
    matrix res(n, vector<double>(m));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; ++j)
            res[j][i] = arr[i][j];
    return res;
}

Layer::Layer(int input_size, int output_size) {
    weights = matrix(input_size, vector<double>(output_size));
    biases = matrix(1, vector<double>(output_size));

    for (int i = 0; i < input_size; ++i)
        for (int j = 0; j < output_size; ++j)
            weights[i][j] = ((double) rand() / RAND_MAX) - 0.5;

    for (int i = 0; i < output_size; ++i)
        biases[0][i] = ((double) rand() / RAND_MAX) - 0.5;
}

matrix Layer::forward(matrix inputs) {
    matrix outputs(inputs.size(), vector<double>(weights[0].size()));

    // [X] * [W] + [b] = [Z]
    for (int i = 0; i < inputs.size(); ++i) {
        for (int j = 0; j < weights[0].size(); ++j) {
            outputs[i][j] = 0;
            for (int k = 0; k < weights.size(); ++k) {
                outputs[i][j] += inputs[i][k] * weights[k][j];
            }
            outputs[i][j] += biases[0][j];
        }
    }    

    // activation function
    for (int i = 0; i < outputs.size(); ++i) {
        for (int j = 0; j < outputs[0].size(); ++j) {
            if (activation_function == "sigmoid") {
                outputs[i][j] = 1 / (1 + exp(-outputs[i][j]));
            }
        }
    }

    return outputs;
}

matrix Layer::backward(matrix inputs, matrix output_gradient, double learning_rate) {
    matrix activated_inputs(inputs.size(), vector<double>(inputs[0].size()));
    for (int i = 0; i < inputs.size(); ++i) {
        for (int j = 0; j < inputs[0].size(); ++j) {
            if (activation_function == "sigmoid") {
                double sigmoid = 1 / (1 + exp(-inputs[i][j]));
                activated_inputs[i][j] = sigmoid * (1 - sigmoid);
            }
        }
    }

    matrix outputs = dot(output_gradient, activated_inputs);

    matrix weights_gradient = dot(outputs, transpose(inputs));
    matrix input_gradient = dot(transpose(weights), outputs);

    // TODO: SEG FAULT HERE, weights size is not the same as weights_gradient size
    for (int i = 0; i < weights.size(); ++i) {
        for (int j = 0; j < weights[0].size(); ++j) {
            weights[i][j] -= weights_gradient[i][j] * learning_rate;
        }
    }

    for (int i = 0; i < biases.size(); ++i) {
        for (int j = 0; j < biases[0].size(); ++j) {
            biases[i][j] -= outputs[i][j] * learning_rate;
        }
    }

    return input_gradient;
}