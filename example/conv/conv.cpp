#include "RainNet.h"

Layer* layer_0 = new ConvolutionalLayer(2, 2, 3, 1, 1, "sigmoid");

float* input = (float*) new float[5 * 5 * 3];
float* target = (float*) new float[1];

int main()
{
    Net network;
    network.learningRate = 1.0;
    network.layers.push_back(layer_0);

    for (size_t i=0; i<5*5*3; ++i)
        input[i] = 1;

    return 0;
}