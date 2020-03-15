#include "RainNet.h"

Layer* layer_0 = new ConnectedLayer(2, "sigmoid");
Layer* layer_1 = new ConnectedLayer(2, "sigmoid");
Layer* layer_2 = new ConnectedLayer(1, "sigmoid");

tensor* input = new tensor(1, 2, 1); // input: [a_1, a_2, ...]
tensor* target = new tensor(1, 1, 1);

int main()
{
    Net network;
    network.learningRate = 1.0;
    network.lossFunction = "L1";
    network.layers.push_back(layer_0);
    network.layers.push_back(layer_1);
    network.layers.push_back(layer_2);

    input->data[0] = 0.35;
    input->data[1] = 0.9;

    target->data[0] = 1;
    network.target = target;

    size_t epoch = 10;
    network.train(input, epoch);
    // network.predict(input);

    return 0;
}