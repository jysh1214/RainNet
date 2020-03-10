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
 * initRandomWieght - create wieght between [-1, 1] randomly
 * 
 * NOTE: first layer size equal to input data
*/
void Net::initRandomWieght()
{
    srand(time(NULL));

    for (size_t i=1; i<this->layers.size(); ++i){
        size_t a = (this->layers[i]->prev)->getSize();
        size_t b = (this->layers[i])->getSize();

        float* wieght = (float*) new float[a*b];
        for (size_t j=0; j<a*b; ++j){
            wieght[j] = (float(rand()%200)/100) - 1;
        }
        (this->layers[i])->setWieght(wieght);
    }
}

/**
 * init - init the network
 * 
 * 1. load wieght or create wieght
 * 2. set input layer
 * 3. set output layer
*/
void Net::init()
{
    createLayerList(this->layers);
    if (!loadWieght){
        this->initRandomWieght();
    }
    if (loadWieght){

    }

    // input layer
    (this->layers[0])->setInput(this->input);
    (this->layers[0])->setOutput(this->input);

    // output layer
    if (this->training){
        assert(this->target && "\nNet::init() ERROR: The target is not setted.\n");
        size_t size = this->layers.size();
        (this->layers[size-1])->setTarget(this->target);
    }
}

void Net::predict(float* input)
{
    this->training = false;
    this->input = input;
    // this->init();
    (this->layers[1])->forward(this);
}

/**
 * TODO: epoch
*/
void Net::train(float* input)
{
    assert(this->learningRate && "\nNet::train ERROR: learning rate is not setted.\n");
    this->training = true;
    this->input = input;
    this->init();
    (this->layers[1])->forward(this);
}
