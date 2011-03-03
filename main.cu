#include <iostream>
using namespace std;

const int k1tok3 = 1;
const int threadDim = 8;
const int xblocks = 10;
const int yblocks = 10;
const int arraySize = xblocks*threadDim * yblocks*threadDim + 2; // penultimate cell in array is top boundary, final is bottom boundry

#include "bitsnbobs.cu"

//__global__ monte_kernel(double *nx, double *ny, double *inp, curandState *state, double aoa, double iTk, int offset);
__global__ void energy_kernel(double *nx, double *ny, bool *inp, double *blockEnergies);

int main()
{
	int i;
	double nx[arraySize], ny[arraySize], *dev_nx, *dev_ny;
	bool inp[arraySize], *dev_inp; // is nanoparticle
	double energy, blockEnergies[xblocks*yblocks], *dev_blockEnergies;

	// Initialise grid
	gridInit(nx, ny, arraySize);

	// Nanoparticle adders go here

	// Allocate and copy arrays to GPU	
	cudaMalloc( (void**) &dev_nx, arraySize*sizeof(double));
	cudaMalloc( (void**) &dev_ny, arraySize*sizeof(double));
	cudaMalloc( (void**) &dev_inp, arraySize*sizeof(bool));
	cudaMalloc( (void**) &dev_blockEnergies, xblocks*yblocks*sizeof(double));
	cudaMemcpy( dev_nx, nx, arraySize*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_ny, ny, arraySize*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_inp, inp, arraySize*sizeof(bool), cudaMemcpyHostToDevice);

	// Calculate initial energy
	dim3 threads(threadDim, threadDim);
	dim3 blocks(xblocks, yblocks);
	energy_kernel<<<blocks, threads>>>(dev_nx, dev_ny, dev_inp, dev_blockEnergies);

	// Copy back and sum blockEnergies
	cudaMemcpy(blockEnergies, dev_blockEnergies, xblocks*yblocks*sizeof(double), cudaMemcpyDeviceToHost);
	energy=0;
	for(i=0; i<xblocks*yblocks; i++)
	{
		energy+=blockEnergies[i];
	}

	cout << energy << endl;
	
	return 0;
}

__global__ void energy_kernel(double *nx, double *ny, bool *inp, double *blockEnergies)
{
	__shared__ double energy[threadDim*threadDim];
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int id = threadIdx.x + threadIdx.y * blockDim.x;
	int blockID = blockIdx.x + blockIdx.y * gridDim.x;
	int i = blockDim.x * blockDim.y / 2;

	energy[id] = calcEnergy(x, y, nx, ny);

	// sum for the block
	__syncthreads();
	
	while(i>0)
	{
		if(id < i ) energy[id] += energy[id + i];
		__syncthreads();
		i/=2;
	}

	if(id==0) blockEnergies[blockID] = energy[0];

}
