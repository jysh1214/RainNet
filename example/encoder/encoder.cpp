#include "RainNet.h"

Layer* layer_0 = new ConvolutionalLayer(800, 800, 1, 0, 0, "leaky"); // input data
Layer* layer_1 = new ConvolutionalLayer(3, 3, 16, 1, 1, "leaky"); // 800
Layer* layer_2 = new ConvolutionalLayer(3, 3, 32, 1, 2, "leaky"); // 400
Layer* layer_3 = new ConvolutionalLayer(3, 3, 64, 1, 2, "leaky"); // 200
Layer* layer_4 = new ConvolutionalLayer(3, 3, 128, 1, 2, "leaky"); // 100
Layer* layer_5 = new ConvolutionalLayer(3, 3, 128, 1, 2, "leaky"); // 50
Layer* layer_6 = new ConvolutionalLayer(3, 3, 128, 1, 2, "leaky"); // 25
Layer* layer_7 = new ConvolutionalLayer(3, 3, 256, 1, 2, "tanh"); // 13

int main()
{
    tensor* target = new tensor(13, 13, 256);
    for (size_t i=0; i<(13*13*256); ++i)
        target->data[i] = 1;

    Net network;
    network.learningRate = 0.001;
    network.lossFunction = "L2";
    network.layers.push_back(layer_0);
    network.layers.push_back(layer_1);
    network.layers.push_back(layer_2);
    network.layers.push_back(layer_3);
    network.layers.push_back(layer_4);
    network.layers.push_back(layer_5);
    network.layers.push_back(layer_6);
    network.layers.push_back(layer_7);

    Dataset trainData("../../data/test.csv");
    // Dataset testData("../../data/test.csv");

    tensor* input = trainData.dataTensor;
    network.target = target;

    size_t epoch = 500;
    network.train(input, epoch);
    network.predict(input);

    return 0;
}