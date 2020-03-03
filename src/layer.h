#ifndef LAYER_H
#define LAYER_H

class Layer
{
public:
    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual void update() = 0;

};

#endif