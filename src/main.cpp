#include "../include/RainNet.h"

Layer* layer_0 = new ConnectedLayer(2, "sigmoid");
Layer* layer_1 = new ConnectedLayer(2, "sigmoid");
Layer* layer_2 = new ConnectedLayer(1, "sigmoid");

float* input = (float*) new float[2];

int main()
{
    Net network;
    network.layers.push_back(layer_0);
    network.layers.push_back(layer_1);
    network.layers.push_back(layer_2);

    input[0] = 0.35;
    input[1] = 0.9;
    network.predict(input);

    std::cout << (layer_0->next)->getType() << std::endl;
    network.free();

    return 0;
}