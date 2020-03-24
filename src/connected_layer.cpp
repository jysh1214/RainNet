#include "connected_layer.h"

ConnectedLayer::ConnectedLayer(size_t size, std::string activation)
{
    this->ActivationFunction = getActivationFunction(activation);
    this->ActivationGradient = getActivationGradient(activation);
    this->type = "ConnectedLayer";
    this->size = size;
    this->input = new tensor(1, this->size, 1);
    this->output = new tensor(1, this->size, 1);
    this->error = new tensor(1, this->size, 1);
    // weight and bias was created in Net::init()
}

ConnectedLayer::~ConnectedLayer()
{
    if (this->input)  delete this->input;
    if (this->weight) delete this->weight;
    if (this->bias)   delete this->bias;
    if (this->output) delete this->output;
    if (this->error)  delete this->error;
}

void ConnectedLayer::forward(Net* net)
{
    if (this->index == 1) this->input = (this->prev)->getOutput();

    // this->input = (prev->output * weight) + bias
    this->output = matrixMultiplication((this->prev)->getOutput(), this->weight);
    for (size_t i=0; i<(this->size); i++)
        (this->output)->data[i] += (this->bias)->data[i];

    if (this->next) (this->next)->forward(net);
    if (!this->next){ // last layer
        // count error
        for (size_t i=0; i<(this->size); i++){
            (this->error)->data[i] = this->ActivationFunction((this->output)->data[i]) - (net->target)->data[i];
            (this->error)->data[i] *= this->ActivationGradient((this->output)->data[i]);
        }
        // update weight
        for (size_t i=0; i<(this->size); i++){
            float sum = 0.0;
            for (size_t j=0; j<(this->size); j++){
                sum += ((this->error)->data[j] * (this->input)->data[j]);
            }
            (this->weight)->data[i] -= net->learningRate * sum;
        }
        // backward
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
    // assert(net->error && "ConnectedLayer::update ERROR: The error missing.");

    // if (!this->next){ // output layer
    //     for (size_t i=0; i<(this->size); i++){
    //         (this->error)->data[i] = net->LossFunction(net->target, this->output) * this->ActivationGradient((this->output)->data[i]);
    //     }
    // }
    // else {
    //     size_t row = (this->prev)->getSize();
    //     size_t col = this->size;
    //     tensor* prevOutput = (this->prev)->getOutput();
        
    //     for (size_t i=0; i<row; ++i){
    //         for (size_t j=0; j<col; ++j){
    //             (this->weight)->data[i*col + j] += 
    //             (net->error * ActivationGradient(net->error) * prevOutput->data[i]) * net->learningRate;
    //         }
    //     }
    // }
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

size_t ConnectedLayer::getFilters()
{
    std::cout << "\nConnectedLayer::getFilters: can't be used.\n" << std::endl;
    exit(0);
}

size_t ConnectedLayer::getKernelRow()
{
    std::cout << "\nConnectedLayer::getKernelRow: can't be used.\n" << std::endl;
    exit(0);
}

size_t ConnectedLayer::getKernelCol()
{
    std::cout << "\nConnectedLayer::getKernelCol: can't be used.\n" << std::endl;
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
    print((this->weight)->data, (this->weight)->row, (this->weight)->col, (this->weight)->channel);
}

void ConnectedLayer::setBias(tensor* bias)
{
    this->bias = bias;
}

tensor* ConnectedLayer::getBias()
{
    return this->bias;
}

void ConnectedLayer::printBias()
{
    print((this->bias)->data, (this->bias)->row, (this->bias)->col, (this->bias)->channel);
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
    print((this->output)->data, (this->output)->row, (this->output)->col, (this->output)->channel);
}

void ConnectedLayer::setError(tensor* error)
{
    this->error = error;
}

tensor* ConnectedLayer::getError()
{
    return this->error;
}

void ConnectedLayer::printError()
{
    print((this->error)->data, (this->error)->row, (this->error)->col, (this->error)->channel);
}