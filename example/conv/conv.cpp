#include "RainNet.h"

Layer* layer_0 = new ConvolutionalLayer(3, 5, 5, 0, 0, "sigmoid");
Layer* layer_1 = new ConvolutionalLayer(2, 3, 3, 1, 1, "sigmoid");

tensor* input = new tensor(5, 5, 3);
tensor* target = new tensor(5, 5, 2);

int main()
{
    Net network;
    network.learningRate = 0.01;
    network.lossFunction = "L2";
    network.layers.push_back(layer_0);
    network.layers.push_back(layer_1);

    for (size_t i=0; i<(5*5*3); ++i){
        input->data[i] = i;
    }

    for (size_t i=0; i<(5*5*2); ++i){
        target->data[i] = 1;
    }

    network.target = target;

    size_t epoch = 10;
    network.train(input, epoch);
    network.predict(input);

    return 0;
}