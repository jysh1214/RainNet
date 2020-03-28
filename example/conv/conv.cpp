#include "RainNet.h"

static Layer* layer_0 = new ConvolutionalLayer(5, 5, 3, 0, 0, "INPUT_DATA");
static Layer* layer_1 = new ConvolutionalLayer(3, 3, 2, 1, 1, "leaky"); // 5*5*2
static Layer* layer_2 = new ConvolutionalLayer(3, 3, 2, 1, 1, "tanh"); // 5*5*2

static tensor* input = new tensor(5, 5, 3);
static tensor* target = new tensor(5, 5, 2);

int main()
{
    Net network;
    network.learningRate = 0.001;
    network.lossFunction = "MSE";
    network.layers.push_back(layer_0);
    network.layers.push_back(layer_1);
    network.layers.push_back(layer_2);

    for (size_t i=0; i<(5*5*3); ++i){
        input->data[i] = 0.8;
    }

    for (size_t i=0; i<(5*5*2); ++i){
        target->data[i] = 10;
    }

    network.target = target;

    size_t epoch = 10;
    network.train(input, epoch);
    network.predict(input);

    return 0;
}