CXX = mpic++
# "-Wall" is for unused variable
CXXFLAGS = -std=c++0x -Wextra -Wshadow -Werror -O3 -DNDEBUG

INCLUDES =
LDFLAGS =
LIBS =

TARGET = main
OBJS = $(TARGET).o

all: $(TARGET)

$(TARGET): $(OBJS) Makefile
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS) $(LIBS)

$(TARGET).o: $(TARGET).cpp neural_net.h neural_net.cpp function.h Makefile
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $(TARGET).cpp

clean:
	@$(RM) -rf *.o $(TARGET)
