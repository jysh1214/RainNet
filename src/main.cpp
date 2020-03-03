#include "../include/rain_net.h"

Layer* layer_0 = new ConnectedLayer(2, "sigmoid");
Layer* layer_1 = new ConnectedLayer(2, "sigmoid");
Layer* layer_2 = new ConnectedLayer(1, "sigmoid");

int main()
{
    Net net;
    net.layers.push_back(layer_0);
    net.layers.push_back(layer_1);
    net.layers.push_back(layer_2);
    return 0;
}