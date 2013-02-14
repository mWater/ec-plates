CFLAGS = $SHELL(pkg-config --cflags opencv)
LIBS = $SHELL(pkg-config --libs opencv)

CPP_FILES := $(wildcard *.cpp)
OBJ_FILES := $(notdir $(CPP_FILES:.cpp=.o))


main: $(OBJ_FILES)
	g++ -o $@ $^ `pkg-config --libs opencv` 

%.o: %.cpp
	g++ `pkg-config --cflags opencv` -c -o $@ $<

