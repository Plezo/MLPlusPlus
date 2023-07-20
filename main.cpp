
#include <iostream>
#include <vector>

using namespace std;

vector< vector<float> > train = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
};

float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

/*

Mean Squared Error (MSE)

*/
float cost(float weight, float bias) {
    float res = 0.0f;
    size_t n = train.size();
    for (size_t i = 0; i < n; i++) {
        float x = train[i][0];
        float y = x * weight + bias;

        float dist = train[i][1] - y;
        res += dist * dist;
    }
    res /= n;
    return res;
}

int main() {
    // srand(time(0));
    srand(69);

    float weight = rand_float() * 10.0f;
    float bias = rand_float() * 5.0f;

    float eps = 1e-3;
    float learning_rate = 1e-2;

    for (int i = 0; i < 100000; i++) {
        float c = cost(weight, bias);
        float dist_w = (cost(weight + eps, bias) - c) / eps;
        float dist_b = (cost(weight, bias + eps) - c) / eps;
        weight -= learning_rate * dist_w;
        bias -= learning_rate * dist_b;
        cout << "cost = " << cost(weight, bias) << ", w = " << weight << ", b = " << bias << endl;
    }

    cout << "w = " << weight << ", b = " << bias << endl;

    return 0;
}