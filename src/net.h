#ifndef NET_H
#define NET_H

#include "layer.h"

#include <stdlib.h>
#include <vector>

struct Net
{
    std::vector<Layer*> layers;
};

#endif