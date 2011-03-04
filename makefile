NVCC = nvcc

debug=0

#decide whether to build debugging or optimised code (higher level means more, i.e. --debug 3 provides more debugging than --debug 2 )
ifeq (${debug},1)
  DEBUG_OR_OPT= --debug --device-debug 3
else
  DEBUG_OR_OPT = --optimize 2
endif


#Note -arch specifies the virtual architecture and should only be one value.
# -code specifies what physical GPU architectures or virtual architectures to target (e.g. compute_20)
# if code is set to a virtual architecture then final stage compiling is left off which will
# result in a start up delay as the CUDA driver must do JIT compilation.
#
# Here are the recommended settings for different systems
#
# Bluecrystal (Compute 2.0) : -arch=compute_20 -code=sm_20
# Hercules (Computer 1.3) : -arch=compute_13 -code=sm_13,sm_20

NVCC_OPTS = -arch=compute_20 -code=sm_20 ${DEBUG_OR_OPT} --compiler-options -Wall
EXEC_NAME = casey


#Override implicit rules so we generate dependency files
%.o : %.cu
	#compile object
	${NVCC} --compile -o $@ ${NVCC_OPTS} $< 
	#make dependency file for object
	${NVCC} --generate-dependencies ${NVCC_OPTS} $< > $*.dep


OBJECTS = main.o 


#Default target (does linking of objects)
${EXEC_NAME} : ${OBJECTS}
	${NVCC} ${NVCC_OPTS}  $^ --link -o ${EXEC_NAME}

#include prerequesite files in make file
-include $(OBJECTS:.o=.dep) 

.PHONY : clean
clean : 
	rm *.o *.dep ${EXEC_NAME}

