#include "connected_layer.h"

ConnectedLayer::ConnectedLayer(size_t size, std::string activation)
{
    this->ActivationFunction = getActivationFunction(activation);
    this->ActivationGradient = getActivationGradient(activation);
    this->type = "ConnectedLayer";
    this->size = size;
}

ConnectedLayer::~ConnectedLayer()
{
    if (input) delete input;
    if (weight) delete weight;
    if (output) delete output;
}

void ConnectedLayer::forward(Net* net)
{
    assert(this->weight && "ConnectedLayer::forward ERROR: The weight missing.");

    tensor* t_output;
    if (this->prev){
        t_output = matrixMultiplication((this->prev)->getOutput(), this->weight);
    }
    
    if (t_output) this->output = t_output;
    if (ActivationFunction){
        for (size_t i=0; i<this->size; ++i){
            (this->output)->data[i] = ActivationFunction((this->output)->data[i]);
        }
    }

    if (this->next) (this->next)->forward(net);
    if (!this->next){
        if (net->training){
            // count error
            net->error = net->LossFunction(net->target, this->output);
            // net->error = (net->target[0] - this->output[0]) * ActivationGradient(this->output[0]);
            net->error *= ActivationGradient(net->error);
            std::cout << "error: " << net->error << std::endl;
            // update weight
            this->update(net);
            (this->prev)->backward(net);
        }
        if (!net->training){
            // count error
            // net->error = (net->target[0] - this->output[0]) * ActivationGradient(this->output[0]);
            net->error = net->LossFunction(net->target, this->output);
            net->error *= ActivationGradient(net->error);
            std::cout << "error: " << net->error << std::endl;
        }
    }
}

void ConnectedLayer::backward(Net* net)
{
    if (this->prev){
        this->update(net);
        (this->prev)->backward(net);
    }
}

void ConnectedLayer::update(Net* net)
{
    assert(net->error && "ConnectedLayer::update ERROR: The error missing.");
    size_t row = (this->prev)->getSize();
    size_t col = this->size;

    for (size_t i=0; i<row; ++i){
        tensor* prevOutput = (this->prev)->getOutput();
        for (size_t j=0; j<col; ++j){
            (this->weight)->data[i*col + j] += (net->error * prevOutput->data[i]) * net->learningRate;
        }
    }
}

std::string ConnectedLayer::getType()
{
    return this->type;
}

void ConnectedLayer::setIndex(size_t i)
{
    this->index = i;
}


size_t ConnectedLayer::getIndex()
{
    return this->index;
}

size_t ConnectedLayer::getSize()
{
    return this->size;
}

size_t ConnectedLayer::getKernelRow()
{
    std::cout << "\nConnectedLayer::getKernelRow can't be used.\n" << std::endl;
    exit(0);
}

size_t ConnectedLayer::getKernelCol()
{
    std::cout << "\nConnectedLayer::getKernelCol can't be used.\n" << std::endl;
    exit(0);
}

size_t ConnectedLayer::getChannel()
{
    std::cout << "\nConnectedLayer::getChannel can't be used.\n" << std::endl;
    exit(0);
}

void ConnectedLayer::setInput(tensor* input)
{
    this->input = input;
}

tensor* ConnectedLayer::getInput()
{
    return this->input;
}

void ConnectedLayer::setWeight(tensor* weight)
{
    this->weight = weight;
}

tensor* ConnectedLayer::getWeight()
{
    return this->weight;
}

void ConnectedLayer::printWeight()
{
    size_t row = (this->prev)->getSize();
    size_t col = this->size;

    // print(this->weight, row, col);
}

void ConnectedLayer::setOutput(tensor* output)
{
    this->output = output;
}

tensor* ConnectedLayer::getOutput()
{
    return this->output;
}

void ConnectedLayer::printOutput()
{
    size_t row = 1;
    size_t col = this->size;

    // print(this->output, row, col);
}

