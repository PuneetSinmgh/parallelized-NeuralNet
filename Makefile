CC = g++
CFLAGS = -std=c++11

clean:
	rm -rf *.o MNIST_binary_int a.out

all:
	$(CC) $(CFLAGS) -c -fopenmp OO_DNN.cpp main.cpp
	$(CC) -fopenmp main.o OO_DNN.o
	./a.out
	
