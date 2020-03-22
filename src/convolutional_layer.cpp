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
        size_t a = (this->output)->row;
        size_t b = (this->output)->col;
        size_t c = (this->output)->channel;
        for (size_t i=0; i<(a*b*c); ++i){
            (this->output)->data[i] = ActivationFunction((this->output)->data[i]);
        }
    }

    if (this->next) (this->next)->forward(net);
    if (!this->next){
        if (net->training){
            // count error
            net->error = net->LossFunction(net->target, this->output);
            std::cout << "error: " << net->error << std::endl;
            // update weight
            this->update(net);
            (this->prev)->backward(net);

        }
        if (!net->training){
            // count error
            net->error = net->LossFunction(net->target, this->output);
            std::cout << "error: " << net->error << std::endl;
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
    assert(net->error && "ConvolutionalLayer::update ERROR: The error missing.");

    tensor* weightMatrix = tensor2matrix(this->weight, this->filters);
    size_t row = weightMatrix->row;
    size_t col = weightMatrix->col;

    tensor* prevOutput = (this->prev)->getOutput();
    tensor* prevMatrix = tensor2matrix(prevOutput, this->kernelRow, this->kernelCol, this->padding, this->stride);

    for (size_t i=0; i<row; ++i){
        for (size_t j=0; j<col; ++j){
            for (size_t k=0; k<prevMatrix->row; ++k){
                weightMatrix->data[i*col + j] += 
                (net->error * ActivationGradient(net->error) * prevMatrix->data[k*row + j]) * net->learningRate;
            }
        }
    }

    tensor* newWeight = matrix2tensor(weightMatrix, this->kernelRow, this->kernelCol);

    this->weight = nullptr;
    this->weight = newWeight;
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
    print((this->weight)->data, (this->weight)->row, (this->weight)->col, (this->weight)->channel);
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
    print((this->output)->data, (this->output)->row, (this->output)->col, (this->output)->channel);
}