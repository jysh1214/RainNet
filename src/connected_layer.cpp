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
    if (this->input)
        delete this->input;
    if (this->weight)
        delete this->weight;
    if (this->bias)
        delete this->bias;
    if (this->output)
        delete this->output;
    if (this->error)
        delete this->error;
}

void ConnectedLayer::forward(Net *net)
{
    this->input = matrixMul((this->prev)->output, 0, this->weight, 0);
    this->input = matrixAdd(this->input, this->bias);
    for (size_t i = 0; i < (this->size); i++)
        (this->output)->data[i] = this->ActivationFunction((this->input)->data[i]);

    if (this->next)
        (this->next)->forward(net);
    if (!this->next)
        this->backward(net);
}

void ConnectedLayer::backward(Net *net)
{
    if (this->prev)
    {
        this->update(net);
        (this->prev)->backward(net);
    }
}

void ConnectedLayer::update(Net *net)
{
    if (!this->next)
    {
        // show cost
        float cost = net->LossFunction(net->target, this->output);
        std::cout << "cost: " << cost << "\n";
        // count error
        for (size_t i = 0; i < (this->size); i++)
        {
            (this->error)->data[i] = this->ActivationFunction((this->output)->data[i]) - (net->target)->data[i];
            (this->error)->data[i] *= this->ActivationGradient((this->input)->data[i]);
        }
        // update weight
        for (size_t i = 0; i < (this->size); i++)
        {
            float sum = 0.0;
            for (size_t j = 0; j < (this->size); j++)
            {
                sum += (this->error)->data[j] * (this->input)->data[j];
            }
            (this->weight)->data[i] -= net->learningRate * sum;
        }
        // update bias
        for (size_t i = 0; i < (this->size); i++)
        {
            float sum = 0.0;
            for (size_t j = 0; j < (this->size); j++)
            {
                sum += (this->error)->data[j];
            }
            (this->bias)->data[i] -= net->learningRate * sum;
        }
    }
    else
    {
        // count error
        this->error = matrixMul((this->next)->error, 0, (this->next)->weight, 1);
        for (size_t i = 0; i < (this->size); i++)
        {
            (this->error)->data[i] *= this->ActivationGradient((this->input)->data[i]);
        }
        // update weight
        for (size_t i = 0; i < (this->size); i++)
        {
            float sum = 0.0;
            tensor *temp = matrixMul((this->prev)->output, 1, this->error, 0);
            for (size_t j = 0; j < (temp->row * temp->col); j++)
            {
                sum += temp->data[j];
            }
            delete temp;
            (this->weight)->data[i] -= net->learningRate * sum;
        }
        // update bias
        for (size_t i = 0; i < (this->size); i++)
        {
            float sum = 0.0;
            for (size_t j = 0; j < (this->size); j++)
            {
                sum += (this->error)->data[j];
            }
            (this->bias)->data[i] -= net->learningRate * sum;
        }
    }
}
