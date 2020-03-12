.PHONY: clean dirs

SRC=src
OBJ=obj
EXAMPLE=example
CXXFLAG=-fopenmp -Wno-unused-result -Wno-unused-function -Isrc/ -Iinclude/

all: dirs ./RainNet

./RainNet: $(EXAMPLE)/main.cpp\
$(OBJ)/connected_layer.o\
$(OBJ)/convolutional_layer.o\
$(OBJ)/dataset.o\
$(OBJ)/net.o
	g++ -Wall $^ -o $@ $(CXXFLAG)

$(OBJ)/connected_layer.o: $(SRC)/connected_layer.cpp $(SRC)/connected_layer.h
	g++ -Wall -c $< -o $@ $(CXXFLAG)

$(OBJ)/convolutional_layer.o: $(SRC)/convolutional_layer.cpp $(SRC)/convolutional_layer.h
	g++ -Wall -c $< -o $@ $(CXXFLAG)

$(OBJ)/dataset.o: $(SRC)/dataset.cpp $(SRC)/dataset.h
	g++ -Wall -c $< -o $@ $(CXXFLAG)

$(OBJ)/net.o: $(SRC)/net.cpp $(SRC)/net.h
	g++ -Wall -c $< -o $@ $(CXXFLAG)

dirs:
	mkdir -p $(SRC) $(OBJ)

clean:
	rm -rf $(OBJ) ./RainNet .vscode

state:
	wc src/*