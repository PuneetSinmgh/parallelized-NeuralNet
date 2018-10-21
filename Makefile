CC = g++
CFLAGS = -std=c++11

clean:
	rm -rf *.o MNIST_binary_int a.out

all:
	$(CC) $(CFLAGS) -c main.cpp OO_DNN.cpp
	$(CC) main.o OO_DNN.o
	./a.out
	
