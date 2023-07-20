/*

TODO:

Shuffle the the rows in the MNIST dataset (like np.random.shuffle in Python) (Make sure to shuffle the labels as well to correspond to the same rows)

*/



#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

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

void ReadMNIST(vector<vector<double>> &arr) {
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

void transpose(vector<vector<double>> &arr) {
    int m = arr.size(), n = arr[0].size();
    vector<vector<double>> res(n, vector<double>(m));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; ++j) {
            res[j][i] = arr[i][j];
        }
    }
    arr = res;
}

int main() {
    vector<vector<double>> train;
    vector<double> labels;
    ReadMNIST(train);
    ReadMNISTLabels(labels);

    int m = train.size(), n = train[0].size();

    // shuffle data

    vector<vector<double>> data_dev(1000, vector<double>(n));
    copy(train.begin(), train.begin()+1000, data_dev.begin());
    transpose(data_dev);

    vector<double> Y_dev(1000);
    copy(labels.begin(), labels.begin()+1000, Y_dev.begin());

    vector<vector<double>> X_dev(1000, vector<double>(n));
    copy(data_dev.begin(), data_dev.end(), X_dev.begin());
    // X_dev = X_dev / 255

    vector<vector<double>> data_train(m-1000, vector<double>(n));
    copy(train.begin()+1000, train.end(), data_train.begin());
    transpose(data_train);

    vector<double> Y_train(m-1000);
    copy(labels.begin()+1000, labels.end(), Y_train.begin());

    vector<vector<double>> X_train(m-1000, vector<double>(n));
    copy(data_train.begin(), data_train.end(), X_train.begin());
    // X_train = X_train / 255
    // _, m_train = X_train.shape

    return 0;
}