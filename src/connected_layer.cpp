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
    if (input) delete [] input;
    if (wieght) delete [] wieght;
    if (output) delete [] output;
    if (target) delete [] target;
}

void ConnectedLayer::forward(Net* net)
{
    float* t_output;
    if (this->prev){
        t_output =
        matrixMultiplication((this->prev)->getOutput(), this->wieght, 1, (this->prev)->getSize(), this->size);
    }
    
    if (t_output) this->output = t_output;
    if (ActivationFunction){
        for (size_t i=0; i<this->size; ++i){
            this->output[i] = ActivationFunction(this->output[i]);
        }
    }

    std::cout << this->index << std::endl;
    this->printWieght();

    if (this->next) (this->next)->forward(net);
    if (!this->next){
        if (net->training){
            std::cout << "output: " << std::endl;
            this->printOutput();
            // count error
            net->error = (target[0] - this->output[0]) * ActivationGradient(this->output[0]);
            std::cout << "error: " << net->error << std::endl;
            // update wieght
            this->update(net);
            std::cout << "update: " << std::endl;
            this->printWieght();
            // backward
            (this->prev)->backward(net);
        }
        if (!net->training){
            std::cout << "output: " << std::endl;
            this->printOutput();
            // count error
            float error = (target[0] - this->output[0]) * ActivationGradient(this->output[0]);
            std::cout << "error: " << error << std::endl;
        }
    }
}

void ConnectedLayer::backward(Net* net)
{
    if (this->prev){
        this->update(net);
        std::cout << "update: " << std::endl;
        this->printWieght();
        (this->prev)->backward(net);
    }
}

void ConnectedLayer::update(Net* net)
{
    size_t row = (this->prev)->getSize();
    size_t col = this->size;

    for (size_t i=0; i<row; ++i){
        for (size_t j=0; j<col; ++j){
            float* prevOutput = (this->prev)->getOutput();
            this->wieght[j + col*i] += (net->error * prevOutput[i]) * net->learningRate;
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

void ConnectedLayer::setInput(float* input)
{
    this->input = input;
}

float* ConnectedLayer::getInput()
{
    return this->input;
}

void ConnectedLayer::setWieght(float* wieght)
{
    this->wieght = wieght;
}

float* ConnectedLayer::getWieght()
{
    return this->wieght;
}

void ConnectedLayer::printWieght()
{
    size_t row = (this->prev)->getSize();
    size_t col = this->size;

    print(this->wieght, row, col);
}

void ConnectedLayer::setOutput(float* output)
{
    this->output = output;
}

float* ConnectedLayer::getOutput()
{
    return this->output;
}

void ConnectedLayer::printOutput()
{
    size_t row = 1;
    size_t col = this->size;

    print(this->output, row, col);
}

void ConnectedLayer::setTarget(float* target)
{
    this->target = target;
}
