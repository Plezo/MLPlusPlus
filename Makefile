CC=g++
C++FLAGS = -I. -g -Wall -std=c++20
DEPS = Layer.hpp
OBJ = mnist.o Layer.o

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(C++FLAGS)

mnist: $(OBJ)
	$(CC) -o $@ $^ $(C++FLAGS)

clean:
	(rm -f *.o)
