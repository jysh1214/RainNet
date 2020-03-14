#include "RainNet.h"

Layer* layer_0 = new ConnectedLayer(2, "sigmoid");
Layer* layer_1 = new ConnectedLayer(10, "sigmoid");
Layer* layer_2 = new ConnectedLayer(1, "sigmoid");

float* input = (float*) new float[2];
float* target = (float*) new float[1];

int main()
{
    Net network;
    network.learningRate = 1.0;
    network.layers.push_back(layer_0);
    network.layers.push_back(layer_1);
    network.layers.push_back(layer_2);

    input[0] = 0.35;
    input[1] = 0.9;

    target[0] = 1;
    network.target = target;

    size_t epoch = 10;
    network.train(input, epoch);
    network.predict(input);

    return 0;
}