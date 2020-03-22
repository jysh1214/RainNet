#include "net.h"

/**
 * createLayerList - create double linked list from the std::vector
 * @vec: the std::vector
*/
void createLayerList(std::vector<Layer*>& vec)
{
    for (size_t i=0; i<vec.size(); ++i){
        vec[i]->setIndex(i);
        if (i!=0 && i!=vec.size()-1){
            vec[i]->next = vec[i+1];
            vec[i]->prev = vec[i-1];
        }
        else {
            if (i == 0){
                vec[i]->next = vec[i+1];
                vec[i]->prev = nullptr;                
            }
            if (i == vec.size()-1){
                vec[i]->next = nullptr;
                vec[i]->prev = vec[i-1];                
            }
        }
    }
}

/**
 * initRandomweight - create weight between [-1, 1] randomly
 * 
 * for:
 * 1. connected layer
 * 2. convolutional layer
 * 
 * NOTE: first layer size equal to input data
*/
void Net::createRandomWeight()
{
    srand(time(NULL));

    for (size_t i=1; i<this->layers.size(); ++i){
        std::string layerType = (this->layers[i])->getType();
        tensor* weight;

        if (layerType == "ConnectedLayer"){
            size_t a = (this->layers[i]->prev)->getSize();
            size_t b = (this->layers[i])->getSize();

            weight = new tensor(a, b, 1);
            for (size_t j=0; j<a*b; ++j){
                weight->data[j] = (float(rand()%200)/100) - 1;
            }
        }
        else if (layerType == "ConvolutionalLayer"){
            // weight size: row * col * ((input channel) * filters)
            size_t a = (this->layers[i])->getKernelRow();
            size_t b = (this->layers[i])->getKernelCol();
            size_t c = (this->layers[i]->prev)->getFilters();
            size_t d = (this->layers[i])->getFilters();

            weight = new tensor(a, b, c*d);
            for (size_t j=0; j<(a*b*c*d); ++j){
                weight->data[j] = (float(rand()%200)/100) - 1;
            }
            
        }

        (this->layers[i])->setWeight(weight);
    }
}

/**
 * init - init the network
 * 
 * load weight or create weight
*/
void Net::init()
{
    createLayerList(this->layers);

    // input layer: input = output
    (this->layers[0])->setInput(this->input);
    (this->layers[0])->setOutput(this->input);

    if (!loadweight){
        this->createRandomWeight();
    }
    if (loadweight){
        // TODO: load weight
    }

}

/**
 * predict - use existed weight
*/
void Net::predict(tensor* input)
{
    this->training = false;
    this->input = input;
    // this->init();
    (this->layers[1])->forward(this);
}

void Net::train(tensor* input, size_t epoch)
{
    assert(this->LossFunction && "Net::train ERROR: Loss function is not setted.");
    assert(this->learningRate && "Net::train ERROR: Learning rate is not setted.");

    this->LossFunction = getLossFunction(this->lossFunction);

    this->training = true;
    this->input = input;
    this->init();

    for (size_t i=0; i<epoch; ++i)
        (this->layers[1])->forward(this);
}
