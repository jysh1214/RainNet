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
    this->input = matrixMultiplication((this->prev)->output, this->weight);
    this->output = matrixAdd(this->input, this->bias);
    
    if (this->next) (this->next)->forward(net);
    if (!this->next){
        this->backward(net);
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
    if (!this->next){
        // show cost
        float cost = net->LossFunction(net->target, this->output);
        std::cout << "cost: " << cost << std::endl;
        // count error
        for (size_t i=0; i<(this->size); i++){
            (this->error)->data[i] = this->ActivationFunction((this->output)->data[i]) - (net->target)->data[i];
            (this->error)->data[i] *= this->ActivationGradient((this->input)->data[i]);
        }
        // update weight
        for (size_t i=0; i<(this->size); i++){
            float sum = 0.0;
            for (size_t j=0; j<(this->size); j++){
                sum += (this->error)->data[j] * (this->input)->data[j];
            }
            (this->weight)->data[i] -= net->learningRate * sum;
        }
        // update bias
        for (size_t i=0; i<(this->size); i++){
            float sum = 0.0;
            for (size_t j=0; j<(this->size); j++){
                sum += (this->error)->data[j];
            }
            (this->bias)->data[i] -= net->learningRate * sum;
        }
    }
    else{
        // count error
        this->error = matrixMultiplication((this->next)->weight, (this->next)->error);
        for (size_t i=0; i<(this->size); i++){
            (this->error)->data[i] *= this->ActivationGradient((this->input)->data[i]);
        }
        // update weight
        for (size_t i=0; i<(this->size); i++){
            float sum = 0.0;
            for (size_t j=0; j<(this->size); j++){
                sum += (this->error)->data[j] * (this->input)->data[j];
            }
            (this->weight)->data[i] -= net->learningRate * sum;
        }
        // update bias
        for (size_t i=0; i<(this->size); i++){
            float sum = 0.0;
            for (size_t j=0; j<(this->size); j++){
                sum += (this->error)->data[j];
            }            
            (this->bias)->data[i] -= net->learningRate * sum;
        }
    }
}
