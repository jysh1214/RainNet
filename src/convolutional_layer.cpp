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
    
    print((this->prev->getOutput())->data, (this->prev->getOutput())->row, (this->prev->getOutput())->col, (this->prev->getOutput())->channel);
    print((this->weight)->data, (this->weight)->row, (this->weight)->col, (this->weight)->channel);

    // tensor* prevOut = (this->prev)->getOutput();
    // tensor* flipTensor = flip(prevOut);
    // tensor* flipWeight = flip(this->weight);

    // tensor* n_flipWeight = dot(flipWeight, net->error);

    // tensor* decay = convolution(flipTensor, n_flipWeight, this->padding, this->stride);
    // tensor* n_decay = dot(decay, ActivationGradient(net->error));

    // print(n_decay->data, n_decay->row, n_decay->col, n_decay->channel);

    tensor* fuck = tensor2matrix(this->weight, this->filters);
    print(fuck->data, fuck->row, fuck->col, fuck->channel);

    // size_t z = 0; // weight channel
    // for (size_t x=0; x<(this->weight->channel)/(prevOut->channel); ++x){
    //     for (size_t y=0; y<prevOut->channel; ++y){ // input channel
    //         std::cout << x << ", " << y << ", " << z << std::endl;

            

    //         z++;
    //     }
    // }
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