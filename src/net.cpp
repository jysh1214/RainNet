#include "net.h"

/**
 * createLayerList - create double linked list from the std::vector
 * @vec: the std::vector
*/
void createLayerList(std::vector<Layer*>& vec)
{
    assert(vec.size() != 0 && "createLayerList ERROR: Add the layer first.");
    for (size_t i=0; i<vec.size(); ++i){
        vec[i]->index = i;
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

    for (size_t i=1; i<this->layers.size(); i++){
        std::string layerType = (this->layers[i])->type;
        tensor* weight;

        if (layerType == "ConnectedLayer"){
            size_t a = (this->layers[i]->prev)->size;
            size_t b = (this->layers[i])->size;

            weight = new tensor(a, b, 1);
            for (size_t j=0; j<a*b; j++)
                weight->data[j] = (float(rand()%200)/100) - 1;

        }
        else if (layerType == "ConvolutionalLayer"){
            // weight size: row * col * (filters * prev->filters)
            size_t a = this->layers[i]->row;
            size_t b = this->layers[i]->col;
            size_t c = this->layers[i]->filters;
            size_t d = (this->layers[i]->prev)->filters;
            
            weight = new tensor(a, b, c*d);
            for (size_t j=0; j<a*b*c*d; j++)
                weight->data[j] = (float(rand()%200)/100) - 1;

        }
        else {
            /* add new layer */
        }
        

        (this->layers[i])->weight = weight;
    }
}

/**
 * createRandomBias - create bias between [0.01, 1.01] randomly
 * 
 * for:
 * 1. connected layer
 * 2. convolutional layer
 * 
 * NOTE: first layer size equal to input data
*/
void Net::createRandomBias()
{
    srand(time(NULL));

    for (size_t i=1; i<this->layers.size(); i++){
        std::string layerType = (this->layers[i])->type;
        tensor* bais;

        if (layerType == "ConnectedLayer"){
            size_t a = (this->layers[i])->size;
            bais = new tensor(1, a, 1);
            for (size_t j=0; j<a; j++)
                bais->data[j] = (float(rand()%100)/100) + 0.01;
            
        }
        else if (layerType == "ConvolutionalLayer"){
            size_t prevRow = (this->layers[i]->prev)->output->row;
            size_t prevCol = (this->layers[i]->prev)->output->col;
            size_t thisRow = (this->layers[i])->row;
            size_t thisCol = (this->layers[i])->col;
            size_t padding = (this->layers[i])->padding;
            size_t stride = (this->layers[i])->stride;
            size_t outputRow = (prevRow - thisRow + 2*padding)/stride + 1;
            size_t outputCol = (prevCol - thisCol + 2*padding)/stride + 1;
            size_t filters = (this->layers[i])->filters;
            bais = new tensor(outputRow, outputCol, filters);
            std::cout << outputRow << ", " << outputCol << ", " << filters << std::endl;
            for (size_t j=0; j<outputRow*outputCol*filters; j++)
                bais->data[j] = (float(rand()%100)/100) + 0.01;

        }
        else {
            /* add new layer */
        }

        (this->layers[i])->bias = bais;
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
    (this->layers[0])->input = this->input;
    (this->layers[0])->output = this->input;

    if (!loadweight){
        this->createRandomWeight();
        this->createRandomBias();
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
