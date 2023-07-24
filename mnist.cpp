/*

TODO:

Write test cases for each math function
Swap all vectors to matrices (instead of having a vector, having a 1xn matrix)

*/

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <random>
#include <math.h>

using namespace std;

template<class T>
using matrix = vector<vector<T>>;

struct Params {
    matrix<double> W1;
    matrix<double> W2;
    matrix<double> b1;
    matrix<double> b2;
};

/* 
https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c

On Yann Lecun's site he mentions how intel processors are little endian, so you need to reverse the bytes.
*/
int ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

void ReadMNIST(matrix<double> &arr) {
    ifstream file ("train-images-idx3-ubyte", ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        arr.resize(number_of_images, vector<double>(n_rows*n_cols));
        cout << "magic number = " << magic_number << endl;
        cout << "number of images = " << number_of_images << endl;
        cout << "rows = " << n_rows << endl;
        cout << "cols = " << n_cols << endl;
        cout << endl;

        for(int i = 0; i < number_of_images; ++i) {
            for(int r = 0; r < n_rows; ++r) {
                for(int c = 0; c < n_cols; ++c) {
                    unsigned char pixel = 0;
                    file.read((char*) &pixel, sizeof(pixel));
                    arr[i][(n_rows*r)+c] = (double) pixel;
                }
            }
        }
    }
}

void ReadMNISTLabels(vector<double> &arr) {
    ifstream file ("train-labels-idx1-ubyte", ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;

        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);

        arr.resize(number_of_images);
        cout << "magic number = " << magic_number << endl;
        cout << "number of images = " << number_of_images << endl;
        cout << endl;

        for(int i = 0; i < number_of_images; ++i) {
            unsigned char label = 0;
            file.read((char*) &label, sizeof(label));
            arr[i] = (double) label;
        }
    }
}

void transpose(matrix<double> &arr) {
    int m = arr.size(), n = arr[0].size();
    matrix<double> res(n, vector<double>(m));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; ++j)
            res[j][i] = arr[i][j];
    arr = res;
}

void dot(const matrix<double> &a, const matrix<double> &b, matrix<double> &res) {
    int m = a.size(), n = b[0].size(), p = b.size();
    res.resize(m, vector<double>(n));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++ j)
            for (int k = 0; k < p; ++k)
                res[i][j] += a[i][k] * b[k][j];
}

void swap(vector<double> &a, vector<double> &b) {
    vector<double> temp = a;
    a = b;
    b = temp;
}

void swap(double &a, double &b) {
    double temp = a;
    a = b;
    b = temp;
}

void shuffle(matrix<double> arr, time_t seed) {
    srand(seed);
    for (int i = arr.size()-1; i > 0; --i) {
        int j = rand() % (i+1);
        swap(arr[i], arr[j]);
    }
}

void shuffle(vector<double> arr, time_t seed) {
    srand(seed);
    for (int i = arr.size()-1; i > 0; --i) {
        int j = rand() % (i+1);
        swap(arr[i], arr[j]);
    }
}

matrix<double> rand_matrix(int m, int n) {
    srand(time(0));
    matrix<double> res(m, vector<double>(n));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++ j)
            res[i][j] = ((double) rand() / RAND_MAX) - 0.5;
    return res;
}

Params init_params() {
    Params params;
    params.W1 = rand_matrix(10, 784);
    params.W2 = rand_matrix(10, 10);
    params.b1 = rand_matrix(10, 1);
    params.b2 = rand_matrix(10, 1);
    return params;
}

void ReLU(matrix<double> &Z) {
    for (int i = 0; i < Z.size(); ++i)
        for (int j = 0; j < Z[i].size(); ++j)
            Z[i][j] = max(Z[i][j], 0.0);
}

// // per vector
// void softmax(vector<double>& Z) {
    
// }


// per matrix
void softmax(matrix<double>& Z) {
    for (int i = 0; i < Z.size(); ++i) {
        double sum = 0;
        for (int j = 0; j < Z[i].size(); ++j)
            Z[i][j] = exp(Z[i][j]);

        for (int j = 0; j < Z[i].size(); ++j)
            Z[i][j] / sum;
    }
        // softmax(Z[i]);
}

void forward_prop(const Params& params, matrix<double> X) {

    // Z1 = W1.dot(X) + b1
    matrix<double> Z1(params.W1.size(), vector<double>(X[0].size()));
    dot(params.W1, X, Z1);
    for (int i = 0; i < Z1.size(); ++i)
        for (int j = 0; j < Z1[i].size(); ++j)
            Z1[i][j] += params.b1[i][j];

    // A1 = ReLU(Z1);
    matrix<double> A1(Z1.size(), vector<double>(Z1[0].size()));
    copy(Z1.begin(), Z1.end(), A1.begin());
    ReLU(A1);

    // Z2 = W2.dot(A1) + b2
    matrix<double> Z2(params.W2.size(), vector<double>(A1[0].size()));
    dot(params.W2, A1, Z2);
    for (int i = 0; i < Z2.size(); ++i)
        for (int j = 0; j < Z2[i].size(); ++j)
            Z2[i][j] += params.b2[i][j];

    // A2 = softmax(Z2)
    matrix<double> A2(Z2.size(), vector<double>(Z2[0].size()));
    copy(Z2.begin(), Z2.end(), A2.begin());
    softmax(A2);


}

// rather than setting variables, just modify them in place for performance
Params gradient_descent(matrix<double> X, vector<double> Y, double alpha, int iterations) {
    Params params = init_params();
    for (int i = 0; i < iterations; ++i) {
        // forward prop
        // Z1 = W1.dot(X) + b1
        // A1 = ReLU(Z1)
        // Z2 = W2.dot(A1) + b2
        // A2 = softmax(Z2)
        // cost = -np.sum(Y*np.log(A2))/m_train

        // Z1 = W1.dot(X) + b1
        matrix<double> Z1(params.W1.size(), vector<double>(X[0].size()));
        dot(params.W1, X, Z1);
        for (int i = 0; i < Z1.size(); ++i)
            for (int j = 0; j < Z1[i].size(); ++j)
                Z1[i][j] += params.b1[i][j];

        // A1 = ReLU(Z1);
        matrix<double> A1(Z1.size(), vector<double>(Z1[0].size()));
        copy(Z1.begin(), Z1.end(), A1.begin());
        ReLU(A1);

        // Z2 = W2.dot(A1) + b2
        matrix<double> Z2(params.W2.size(), vector<double>(A1[0].size()));
        dot(params.W2, A1, Z2);
        for (int i = 0; i < Z2.size(); ++i)
            for (int j = 0; j < Z2[i].size(); ++j)
                Z2[i][j] += params.b2[i][j];

        // A2 = softmax(Z2)
        matrix<double> A2(Z2.size(), vector<double>(Z2[0].size()));
        copy(Z2.begin(), Z2.end(), A2.begin());
        softmax(A2);


    }

    return params;
}

int main() {
    matrix<double> train;
    vector<double> labels;
    ReadMNIST(train);
    ReadMNISTLabels(labels);

    int m = train.size(), n = train[0].size();

    // shuffle data
    time_t seed = time(0);
    shuffle(train, seed);
    shuffle(labels, seed);

    matrix<double> data_dev(1000, vector<double>(n));
    copy(train.begin(), train.begin()+1000, data_dev.begin());
    transpose(data_dev);

    vector<double> Y_dev(1000);
    copy(labels.begin(), labels.begin()+1000, Y_dev.begin());

    matrix<double> X_dev(1000, vector<double>(n));
    copy(data_dev.begin(), data_dev.end(), X_dev.begin());
    // X_dev = X_dev / 255

    matrix<double> data_train(m-1000, vector<double>(n));
    copy(train.begin()+1000, train.end(), data_train.begin());
    transpose(data_train);

    vector<double> Y_train(m-1000);
    copy(labels.begin()+1000, labels.end(), Y_train.begin());

    matrix<double> X_train(m-1000, vector<double>(n));
    copy(data_train.begin(), data_train.end(), X_train.begin());
    // X_train = X_train / 255
    // _, m_train = X_train.shape

    Params params = gradient_descent(X_train, Y_train, 0.1, 500);
    
    return 0;
}