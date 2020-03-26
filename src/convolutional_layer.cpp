#include "convolutional_layer.h"

ConvolutionalLayer::ConvolutionalLayer(size_t row, size_t col, size_t filters, size_t padding, size_t stride, std::string activation)
{
    this->ActivationFunction = getActivationFunction(activation);
    this->ActivationGradient = getActivationGradient(activation);
    this->type = "ConvolutionalLayer";
    this->filters = filters;
    this->row = row;
    this->col = col;
    this->padding = padding;
    this->stride = stride;
}

ConvolutionalLayer::~ConvolutionalLayer()
{
    if (this->input)  delete this->input;
    if (this->weight) delete this->weight;
    if (this->bias)   delete this->bias;
    if (this->output) delete this->output;
    if (this->error)  delete this->error;
}

void ConvolutionalLayer::forward(Net* net)
{
    assert(this->weight && "ConvolutionalLayer::forward ERROR: The weight missing.");

    this->output = convolution((this->prev)->output, this->weight, this->padding, this->stride);

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

        assert(net->target->row == this->output->row);
        assert(net->target->col == this->output->col);
        assert(net->target->channel == this->output->channel);

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

    tensor* prevOutput = (this->prev)->output;
    tensor* prevMatrix = tensor2matrix(prevOutput, this->row, this->col, this->padding, this->stride);

    for (size_t i=0; i<row; ++i){
        for (size_t j=0; j<col; ++j){
            for (size_t k=0; k<prevMatrix->col; ++k){
                weightMatrix->data[i*col + j] += 
                (net->error * ActivationGradient(net->error) * prevMatrix->data[k*(prevMatrix->col) + i]) * net->learningRate;
            }
        }
    }

    tensor* newWeight = matrix2tensor(weightMatrix, this->row, this->col);
    delete weightMatrix;

    delete this->weight;
    this->weight = nullptr;
    this->weight = newWeight;
}
