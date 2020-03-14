#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "net.h"
#include "layer.h"
#include "activation_fumction.h"
#include "matrix_operator.h"
#include "print.h"

struct ConvolutionalLayer: public Layer
{
public:
    ConvolutionalLayer(size_t height, size_t width, size_t batch, std::string activation);
    virtual ~ConvolutionalLayer();

    void forward(Net* net);
    void backward(Net* net);
    void update(Net* net);

    std::string getType();
    void setIndex(size_t i);
    size_t getIndex();
    size_t getSize(); // banned
    size_t getHeight();
    size_t getWidth();
    size_t getChannel();
    
    void setInput(float* input);
    float* getInput();
    void setweight(float* weight);
    float* getweight();
    void printweight();
    void setOutput(float* output);
    float* getOutput();
    void printOutput();

private:
    float(*ActivationFunction)(float);
    float(*ActivationGradient)(float);
    std::string type;
    size_t index;
    size_t height;
    size_t width;
    size_t channel;

    float* input;
    float* weight;
    float* output;
};

#endif