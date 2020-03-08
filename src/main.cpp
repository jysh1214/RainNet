#include "../include/RainNet.h"

Layer* layer_0 = new ConnectedLayer(2, "sigmoid");
Layer* layer_1 = new ConnectedLayer(2, "sigmoid");
Layer* layer_2 = new ConnectedLayer(1, "sigmoid");

float* input = (float*) new float[2];
float* target = (float*) new float[1];

int main()
{
    Net network;
    network.layers.push_back(layer_0);
    network.layers.push_back(layer_1);
    network.layers.push_back(layer_2);

    input[0] = 0.35;
    input[1] = 0.9;

    target[0] = 0.5;
    network.target = target;

    network.train(input);

    // network.predict(input);
    network.free();

    return 0;
}