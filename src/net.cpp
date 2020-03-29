#include "net.h"

/**
 * createLayerList - create doubly linked list from the std::vector
 * @vec: the std::vector
*/
void createLayerList(std::vector<Layer *> &vec)
{
    assert(vec.size() != 0 && "createLayerList ERROR: Add the layer first.");
    for (size_t i = 0; i < vec.size(); ++i)
    {
        vec[i]->index = i;
        if (i != 0 && i != vec.size() - 1)
        {
            vec[i]->next = vec[i + 1];
            vec[i]->prev = vec[i - 1];
        }
        else
        {
            if (i == 0)
            {
                vec[i]->next = vec[i + 1];
                vec[i]->prev = nullptr;
            }
            if (i == vec.size() - 1)
            {
                vec[i]->next = nullptr;
                vec[i]->prev = vec[i - 1];
            }
        }
    }
}

/**
 * createLayer - malloc memory to each layer
*/
void Net::mallocLayerMemory()
{
    for (size_t i = 1; i < this->layers.size(); i++)
    {
        std::string layerType = (this->layers[i])->type;
        if (layerType == "ConnectedLayer")
        {
            size_t a = (this->layers[i]->prev)->size;
            size_t b = (this->layers[i])->size;

            // (this->layers[i])->input = new tensor(1, b, 1);
            (this->layers[i])->weight = new tensor(a, b, 1);
            (this->layers[i])->bias = new tensor(1, b, 1);
            (this->layers[i])->output = new tensor(1, b, 1);
            (this->layers[i])->error = new tensor(1, b, 1);
        }
        if (layerType == "ConvolutionalLayer")
        {
            size_t a = (this->layers[i]->prev)->output->row;
            size_t b = (this->layers[i]->prev)->output->col;
            size_t c = (this->layers[i]->prev)->filters; // = (this->layers[i]->prev)->output->channel
            size_t d = (this->layers[i])->row;
            size_t e = (this->layers[i])->col;
            size_t f = (this->layers[i])->filters;
            size_t g = (a - d + 2 * (this->layers[i]->padding)) / (this->layers[i]->stride) + 1; // output row
            size_t h = (b - e + 2 * (this->layers[i]->padding)) / (this->layers[i]->stride) + 1; // output col

            (this->layers[i])->input = new tensor(g, h, f);
            // weight->channel = (prev->filters)*(this->filters)
            (this->layers[i])->weight = new tensor(d, e, c * f);
            (this->layers[i])->bias = new tensor(g, h, f);
            (this->layers[i])->output = new tensor(g, h, f);
            (this->layers[i])->error = new tensor(g, h, f);
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
    for (size_t i = 1; i < this->layers.size(); i++)
    {
        size_t a = (this->layers[i])->weight->row;
        size_t b = (this->layers[i])->weight->col;
        size_t c = (this->layers[i])->weight->channel;

        for (size_t j = 0; j < (a * b * c); j++)
            (this->layers[i])->weight->data[j] = (float(rand() % 200) / 100) - 1;
    }
}

/**
 * createRandomBias - create bias between [0, 1] randomly
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
    for (size_t i = 1; i < this->layers.size(); i++)
    {
        size_t a = (this->layers[i])->bias->row;
        size_t b = (this->layers[i])->bias->col;
        size_t c = (this->layers[i])->bias->channel;

        for (size_t j = 0; j < (a * b * c); j++)
            (this->layers[i])->bias->data[j] = (float(rand() % 100) / 100);
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

    this->mallocLayerMemory();

    if (!loadweight)
    {
        this->createRandomWeight();
        this->createRandomBias();
    }
    if (loadweight)
    {
        // TODO: load weight
    }
}

/**
 * predict - use existed weight
*/
void Net::predict(tensor *input)
{
    this->training = false;
    this->input = input;
    // this->init();
    (this->layers[1])->forward(this);
}

void Net::train(tensor *input, size_t epoch)
{
    assert(this->LossFunction && "Net::train ERROR: Loss function is not setted.");
    assert(this->learningRate && "Net::train ERROR: Learning rate is not setted.");

    this->LossFunction = getLossFunction(this->lossFunction);

    this->training = true;
    this->input = input;
    this->init();

    for (size_t i = 0; i < epoch; ++i)
        (this->layers[1])->forward(this);
}
