.PHONY: clean dirs

SRC=src
OBJ=obj
CXXFLAG=-fopenmp -Wno-unused-result -Wno-unused-function

all: dirs ./RainNet

./RainNet: $(SRC)/main.cpp\
$(OBJ)/connected_layer.o\
$(OBJ)/net.o
	g++ -Wall $^ -o $@ $(CXXFLAG)

$(OBJ)/connected_layer.o: $(SRC)/connected_layer.cpp $(SRC)/connected_layer.h
	g++ -Wall -c $< -o $@ $(CXXFLAG)

$(OBJ)/net.o: $(SRC)/net.cpp $(SRC)/net.h
	g++ -Wall -c $< -o $@ $(CXXFLAG)

dirs:
	mkdir -p $(SRC) $(OBJ)

clean:
	rm -rf $(OBJ) ./RainNet .vscode

state:
	wc src/*