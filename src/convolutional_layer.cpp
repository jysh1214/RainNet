#include "convolutional_layer.h"

ConvolutionalLayer::ConvolutionalLayer(size_t row, size_t col, size_t channel, size_t padding, size_t stride, std::string activation)
{
    this->ActivationFunction = getActivationFunction(activation);
    this->ActivationGradient = getActivationGradient(activation);
    this->type = "ConvolutionalLayer";
    this->kernelRow = row;
    this->kernelCol = col;
    this->channel = channel;
    this->padding = padding;
    this->stride = stride;
}

ConvolutionalLayer::~ConvolutionalLayer()
{
    if (input) delete [] input;
    if (weight) delete [] weight;
    if (output) delete [] output;
}

void ConvolutionalLayer::forward(Net* net)
{
    float* t_output;
    if (this->prev){
        // TODO: conv
        // t_output = conv();
    }
    
    if (t_output) this->output = t_output;
    if (ActivationFunction){
        for (size_t i=0; i<this->size; ++i){
            this->output[i] = ActivationFunction(this->output[i]);
        }
    }

    if (this->next) (this->next)->forward(net);
    if (!this->next){
        if (net->training){
            // count error
            // update weight
        }
        if (!net->training){
            // count error
        }
    }
}

void ConvolutionalLayer::backward(Net* net)
{
    if (this->prev){
        this->update(net);
        (this->prev)->backward(net);
    }
}

void ConvolutionalLayer::update(Net* net)
{
    std::cout << "fuck" << std::endl;
}

std::string ConvolutionalLayer::getType()
{
    return this->type;
}

void ConvolutionalLayer::setIndex(size_t i)
{
    this->index = i;
}

size_t ConvolutionalLayer::getIndex()
{
    return this->index;
}

void ConvolutionalLayer::setInput(float* input)
{
    this->input = input;
}

float* ConvolutionalLayer::getInput()
{
    return this->input;
}

size_t ConvolutionalLayer::getSize()
{
    std::cout << "\nConvolutionalLayer::getSize can't be used.\n" << std::endl;
    exit(0);
}

size_t ConvolutionalLayer::getKernelRow()
{
    return this->kernelRow;
}

size_t ConvolutionalLayer::getKernelCol()
{
    return this->kernelCol;
}

size_t ConvolutionalLayer::getChannel()
{
    return this->channel;
}

void ConvolutionalLayer::setWeight(float* weight)
{

}

float* ConvolutionalLayer::getWeight()
{
    return this->weight;
}

void ConvolutionalLayer::printWeight()
{

}

void ConvolutionalLayer::setOutput(float* output)
{
    this->output = output;
}

float* ConvolutionalLayer::getOutput()
{
    return this->output;
}

void ConvolutionalLayer::printOutput()
{

}