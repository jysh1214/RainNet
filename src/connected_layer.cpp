#include "connected_layer.h"

ConnectedLayer::ConnectedLayer(size_t size, std::string activation)
{
    this->ActivationFunction = getActivationFunction(activation);
    this->type = "ConnectedLayer";
    this->size = size;
}

ConnectedLayer::~ConnectedLayer()
{
    if (input) delete [] input;
    if (wieght) delete [] wieght;
    if (output) delete [] output;
}

void ConnectedLayer::forward(bool training)
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
    this->printOutput();

    if (this->next) (this->next)->forward(training);
    if (!this->next){
        if (training){
            assert(this->target && "\nLayer::forward ERROR: The target is missing.\n");
            // count error
            float error = (target[0] - this->output[0])*(1 - this->output[0])*this->output[0];
            std::cout << "error: " << error << std::endl;
            // update weight

            // backward
        }
        if (!training){

        }
    }
}

void ConnectedLayer::backward()
{

}

void ConnectedLayer::update()
{

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

    if (this->wieght){
        for (size_t i=0; i<row; ++i){
            std::cout << "[";
            for (size_t j=0; j<col; ++j){
                printf("%6f", this->wieght[j + this->size*i]);
                if (j != col-1) std::cout << " ,";
            }
            std::cout << "]";
            std::cout << std::endl;
        }
    }
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

    if (this->output){
        for (size_t i=0; i<row; ++i){
            std::cout << "[";
            for (size_t j=0; j<col; ++j){
                printf("%6f", this->output[j + this->size*i]);
                if (j != col-1) std::cout << " ,";
            }
            std::cout << "]";
            std::cout << std::endl;
        }
    }
}

void ConnectedLayer::setTarget(float* target)
{
    this->target = target;
}
