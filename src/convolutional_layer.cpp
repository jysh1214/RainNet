#include "convolutional_layer.h"

ConvolutionalLayer::ConvolutionalLayer(size_t row, size_t col, size_t filters, size_t padding, size_t stride, std::string activation)
{
    this->ActivationFunction = getActivationFunction(activation);
    this->ActivationGradient = getActivationGradient(activation);
    this->type = "ConvolutionalLayer";
    this->row = row;
    this->col = col;
    this->filters = filters;
    this->padding = padding;
    this->stride = stride;
}

ConvolutionalLayer::~ConvolutionalLayer()
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

void ConvolutionalLayer::forward(Net *net)
{
    tensor *prevMatrix = tensor2matrix((this->prev)->output, this->row, this->col, this->padding, this->stride);
    tensor *weightMatrix = tensor2matrix(this->weight, this->filters);
    tensor *biasMatrix = tensor2matrix(this->bias, this->filters);

    tensor *inputMatrix = matrixMul(prevMatrix, 0, weightMatrix, 0);
    tensor *newMatrix = matrixAdd(inputMatrix, biasMatrix);

    size_t outputRow = ((this->prev)->output->row - this->row + 2 * this->padding) / (this->stride) + 1;
    size_t outputCol = ((this->prev)->output->col - this->col + 2 * this->padding) / (this->stride) + 1;

    this->input = matrix2tensor(newMatrix, outputRow, outputCol);
    // this->output = matrix2tensor(outputMatrix, outputRow, outputCol);

    for (size_t i = 0; i < (outputRow * outputCol * this->filters); i++)
        (this->output)->data[i] = this->ActivationFunction((this->input)->data[i]);

    delete prevMatrix;
    delete weightMatrix;
    delete biasMatrix;
    delete inputMatrix;
    delete newMatrix;

    if (this->next)
        (this->next)->forward(net);
    if (!this->next)
        this->backward(net);
}

void ConvolutionalLayer::backward(Net *net)
{
    if (this->prev)
    {
        this->update(net);
        (this->prev)->backward(net);
    }
}

void ConvolutionalLayer::update(Net *net)
{
    size_t outputRow = (this->output)->row;
    size_t outputCol = (this->output)->col;

    if (!this->next)
    {
        // show cost
        float cost = net->LossFunction(net->target, this->output);
        std::cout << "cost: " << cost << "\n";
        // count error
        for (size_t i = 0; i < (outputRow * outputCol * this->filters); i++)
        {
            (this->error)->data[i] = this->ActivationFunction((this->output)->data[i]) - (net->target)->data[i];
            (this->error)->data[i] *= this->ActivationGradient((this->input)->data[i]);
        }
        // update weight
        for (size_t i = 0; i < (this->row * this->col * (this->weight)->channel); i++)
        {
            float sum = 0.0;
            for (size_t j = 0; j < (outputRow * outputCol * this->filters); j++)
            {
                sum += (this->error)->data[j] * (this->input)->data[j];
            }
            // (this->weight)->data[i] -= net->learningRate * sum;
        }
        // update bias
        for (size_t i = 0; i < (this->row * this->col * (this->bias)->channel); i++)
        {
            float sum = 0.0;
            for (size_t j = 0; j < (outputRow * outputCol * this->filters); j++)
            {
                sum += (this->error)->data[j];
            }
            // (this->bias)->data[i] -= net->learningRate * sum;
        }
    }
    else
    {
        // count error
        tensor *nextWeightMatrix = tensor2matrix((this->next)->weight, (this->next)->filters);
        tensor *nextErrorMatrix = tensor2matrix((this->next)->error, (this->next)->filters);
        // this->error = matrixMul((this->next)->error, 0, (this->next)->weight, 1);
        // std::cout << nextErrorMatrix->row << ", " << nextErrorMatrix->col << std::endl;
        // std::cout << nextWeightMatrix->col << ", " << nextWeightMatrix->row << std::endl;
        tensor *errorMatrix = matrixMul(nextErrorMatrix, 0, nextWeightMatrix, 1);
        this->error = matrix2tensor(errorMatrix, outputRow, outputCol);

        for (size_t i = 0; i < (outputRow * outputCol * this->filters); i++)
        {
            (this->error)->data[i] *= this->ActivationGradient((this->input)->data[i]);
        }
        // update weight
        for (size_t i = 0; i < (this->row * this->col * (this->weight)->channel); i++)
        {
            float sum = 0.0;
            tensor *prevOutputMatrix = tensor2matrix((this->prev)->output, (this->prev)->filters);
            tensor *temp = matrixMul(prevOutputMatrix, 1, errorMatrix, 0);
            for (size_t j = 0; j < (temp->row * temp->col); j++)
            {
                sum += temp->data[j];
            }
            delete prevOutputMatrix;
            delete temp;
            (this->weight)->data[i] -= net->learningRate * sum;
        }
        // update bias
        for (size_t i = 0; i < (outputRow * outputCol * this->filters); i++)
        {
            float sum = 0.0;
            for (size_t j = 0; j < (outputRow * outputCol * this->filters); j++)
            {
                sum += (this->error)->data[j];
            }
            // (this->bias)->data[i] -= net->learningRate * sum;
        }

        delete nextWeightMatrix;
        delete nextErrorMatrix;
        delete errorMatrix;
    }
}
