CC = g++
CCFLAGS = -g -std=c++17 -O3 -march=native -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread
TARGET = main

.PHONY: all clean

all: $(TARGET)

clean:
	rm -rf *.o $(TARGET) cudaTest

$(TARGET): $(TARGET).o
	$(CC) $^  $(CCFLAGS) -o $@ 
	
$(TARGET).o: $(TARGET).cpp
	$(CC) $^ $(CCFLAGS) -c  $@
	