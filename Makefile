SOURCE_FILES=source/quantus_cpu.cpp source/quantus_cuda.cpp source/quantus_gamma.cpp
COEFF_SOURCE_FILES=source/quantus_gamma_coeffs.cpp
OBJECT_FILES=quantus_cpu.o quantus_cuda.o quantus_gamma.o quantus_gamma_coeffs.o

example: example.cu libquantus.a
	nvcc -arch=sm_20 -O2 -o example example.cu libquantus.a -lcurand -Xcompiler -fopenmp

libquantus.a: $(SOURCE_FILES) $(COEFF_SOURCE_FILES) armoury
	rm -f $@
	c++ -I$(BOOST_ROOT) -Iinclude -I. -O2 -c $(COEFF_SOURCE_FILES) -fopenmp
	nvcc -arch=sm_20 -I$(BOOST_ROOT) -Iinclude -I. -O2 -c $(SOURCE_FILES)
	ar cq $@ $(OBJECT_FILES)
	rm -f $(OBJECT_FILES)

clean:
	rm -f libquantus.a $(OBJECT_FILES) example
