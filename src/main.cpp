#include "../include/rain_net.h"

Layer* layer_0 = new ConnectedLayer(1, "sigmoid");

int main()
{
    Net net;
    net.layers.push_back(layer_0);
    return 0;
}