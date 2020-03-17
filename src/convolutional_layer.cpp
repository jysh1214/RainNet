#include "convolutional_layer.h"

ConvolutionalLayer::ConvolutionalLayer(size_t filters, size_t row, size_t col, size_t padding, size_t stride, std::string activation)
{
    this->ActivationFunction = getActivationFunction(activation);
    this->ActivationGradient = getActivationGradient(activation);
    this->type = "ConvolutionalLayer";
    this->filters = filters;
    this->kernelRow = row;
    this->kernelCol = col;
    this->padding = padding;
    this->stride = stride;
}

ConvolutionalLayer::~ConvolutionalLayer()
{
    if (input) delete input;
    if (weight) delete weight;
    if (output) delete output;
}

void ConvolutionalLayer::forward(Net* net)
{
    assert(this->weight && "ConvolutionalLayer::forward ERROR: The weight missing.");

    tensor* t_output;
    if (this->prev){
        t_output = convolution((this->prev)->getOutput(), this->weight, this->padding, this->stride);
    }
    
    if (t_output) this->output = t_output;
    if (ActivationFunction){
        size_t a = 1;
        size_t b = 2;
        size_t c = 3;
        for (size_t i=0; i<(a*b*c); ++i){
            (this->output)->data[i] = ActivationFunction((this->output)->data[i]);
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

void ConvolutionalLayer::setInput(tensor* input)
{
    this->input = input;
}

tensor* ConvolutionalLayer::getInput()
{
    return this->input;
}

size_t ConvolutionalLayer::getSize()
{
    std::cout << "\nConvolutionalLayer::getSize can't be used.\n" << std::endl;
    exit(0);
}

size_t ConvolutionalLayer::getFilters()
{
    return this->filters;
}

size_t ConvolutionalLayer::getKernelRow()
{
    return this->kernelRow;
}

size_t ConvolutionalLayer::getKernelCol()
{
    return this->kernelCol;
}

void ConvolutionalLayer::setWeight(tensor* weight)
{
    this->weight = weight;
}

tensor* ConvolutionalLayer::getWeight()
{
    return this->weight;
}

void ConvolutionalLayer::printWeight()
{

}

void ConvolutionalLayer::setOutput(tensor* output)
{
    this->output = output;
}

tensor* ConvolutionalLayer::getOutput()
{
    return this->output;
}

void ConvolutionalLayer::printOutput()
{

}