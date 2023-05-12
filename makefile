FLAGS = -DDEBUG
LIBS = -lm
ALWAYS_REBUILD = makefile
NVCC = nvcc
NVCCFLAGS = -arch=sm_61

nbody: nbody.o compute.o
	$(NVCC) $(NVCCFLAGS) $(FLAGS) $^ -o $@ $(LIBS)

nbody.o: nbody.cu planets.h config.h vector.h $(ALWAYS_REBUILD)
	$(NVCC) $(NVCCFLAGS) $(FLAGS) -c $<

compute.o: compute.cu config.h vector.h $(ALWAYS_REBUILD)
	$(NVCC) $(NVCCFLAGS) $(FLAGS) -c $<

clean:
	rm -f *.o nbody 
