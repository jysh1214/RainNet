#include "RainNet.h"

static Layer* layer_0 = new ConvolutionalLayer(5, 5, 3, 0, 0, "leaky");
static Layer* layer_1 = new ConvolutionalLayer(3, 3, 2, 1, 1, "leaky");
static Layer* layer_2 = new ConvolutionalLayer(3, 3, 2, 1, 1, "tanh");

static tensor* input = new tensor(5, 5, 3);
static tensor* target = new tensor(5, 5, 2);

int main()
{
    Net network;
    network.learningRate = 0.01;
    network.lossFunction = "L2";
    network.layers.push_back(layer_0);
    network.layers.push_back(layer_1);
    network.layers.push_back(layer_2);

    for (size_t i=0; i<(5*5*3); ++i){
        input->data[i] = 0.8;
    }

    for (size_t i=0; i<(5*5*2); ++i){
        target->data[i] = 1;
    }

    network.target = target;

    size_t epoch = 30;
    network.train(input, epoch);
    network.predict(input);

    return 0;
}