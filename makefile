.PHONY: clean dirs

SRC=src
OBJ=obj

all: dirs ./RainNet

./RainNet: $(SRC)/main.cpp\
$(OBJ)/connected_layer.o
	g++ -Wall $^ -o $@

$(OBJ)/connected_layer.o: $(SRC)/connected_layer.cpp $(SRC)/connected_layer.h
	g++ -Wall -c $< -o $@

dirs:
	mkdir -p $(SRC) $(OBJ)

clean:
	rm -rf $(OBJ) ./RainNet .vscode

state:
	wc src/*