CC = g++
CFLAGS = -std=c++11

clean:
	rm -rf *.o MNIST_binary

all:
	$(CC) $(CFLAGS) MNIST_binary.cpp -o MNIST_binary
