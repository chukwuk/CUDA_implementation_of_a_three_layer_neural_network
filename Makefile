CC=nvcc

all: main

main: main.cu matrixmul.o 
	$(CC) matrixmul.o main.cu -o main

matrixmul.o: matrixmul.cu matrixmul.h 
	$(CC) -c matrixmul.cu -o matrixmul.o

clean:
	rm -f main matrixmul.o
