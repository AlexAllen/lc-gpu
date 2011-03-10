
#include <curand_kernel.h>
using namespace std;

const int k1tok3 = 1;
const int threadDim = 8;
const int xblocks = 4;
const int yblocks = 2;
const int height = yblocks*threadDim;
const int width = xblocks*threadDim;
const int arraySize = xblocks*threadDim * yblocks*threadDim + 2; // penultimate cell in array is top boundary, final is bottom boundry

#include "bitsnbobs.cu"

__global__ void monte_kernel(double *nx, double *ny, bool *inp, curandState *state, double *aoa, double *iTk, int *hits, int offset);
__global__ void annealing_kernel(double *aoa, double *iTk, int *hits, int j);
__global__ void energy_kernel(double *nx, double *ny, bool *inp, double *blockEnergies);
__global__ void empty_kernel(int *hits)
{
	hits[blockIdx.x + blockIdx.y*gridDim.x] = 0;
}

int main()
{
	int i, j, loopMax = 1000000;
	double nx[arraySize], ny[arraySize], *dev_nx, *dev_ny;
	bool inp[arraySize], *dev_inp; // is nanoparticle
	char filename[] = "grid.dump";
	double energy, blockEnergies[xblocks*yblocks], *dev_blockEnergies;
	double aoa = PI*0.5, iTk = 1, *dev_aoa, *dev_iTk;
	int *dev_hits;
	curandState *dev_state;

	// Initialise grid
	gridInit(nx, ny, inp, arraySize);

	// Nanoparticle adders go here

	// Allocate space and copy stuff on/to GPU memory
	danErrHndl( cudaMalloc( (void**) &dev_nx, arraySize*sizeof(double) ) );
	danErrHndl( cudaMalloc( (void**) &dev_ny, arraySize*sizeof(double) ) );
	danErrHndl( cudaMalloc( (void**) &dev_inp, arraySize*sizeof(bool) ) );
	danErrHndl( cudaMalloc( (void**) &dev_aoa, sizeof(double) ) );
	danErrHndl( cudaMalloc( (void**) &dev_iTk, sizeof(double) ) );
	danErrHndl( cudaMalloc( (void**) &dev_hits, xblocks*yblocks*sizeof(int) ) );
	danErrHndl( cudaMalloc( (void**) &dev_state, (arraySize-2)*sizeof(curandState) ) );
	danErrHndl( cudaMalloc( (void**) &dev_blockEnergies, xblocks*yblocks*sizeof(double) ) );
	danErrHndl( cudaMemcpy( dev_nx, nx, arraySize*sizeof(double), cudaMemcpyHostToDevice ) );
	danErrHndl( cudaMemcpy( dev_ny, ny, arraySize*sizeof(double), cudaMemcpyHostToDevice ) );
	danErrHndl( cudaMemcpy( dev_inp, inp, arraySize*sizeof(bool), cudaMemcpyHostToDevice ) );
	danErrHndl( cudaMemcpy( dev_aoa, &aoa, sizeof(double), cudaMemcpyHostToDevice) );
	danErrHndl( cudaMemcpy( dev_iTk, &iTk, sizeof(double), cudaMemcpyHostToDevice) );

	// Calculate initial energy
	dim3 threads(threadDim, threadDim);
	dim3 blocks(xblocks, yblocks);
	empty_kernel<<<blocks, 1>>>(dev_hits);
	energy_kernel<<<blocks, threads>>>(dev_nx, dev_ny, dev_inp, dev_blockEnergies);

	// Copy back and sum blockEnergies
	danErrHndl( cudaMemcpy(blockEnergies, dev_blockEnergies, xblocks*yblocks*sizeof(double), cudaMemcpyDeviceToHost) );
	energy=0;
	for(i=0; i<xblocks*yblocks; i++)
	{
		energy+=blockEnergies[i];
	}

	cout << "Initial energy is: " << energy << endl;

	// Watch out for poisonous adders
	dim3 lessBlocks(xblocks/4,yblocks/2);

	cout << "0%";

	// The monte carlo loop
	for(j=0;j<loopMax;j++)
	{
		for(i=0;i<8;i++)
		{
			monte_kernel<<<lessBlocks, threads>>>(dev_nx, dev_ny, dev_inp, dev_state, dev_aoa, dev_iTk, dev_hits, i);
		}

		if(!(j%100)) cout << "\r" << (double) 100 * j / loopMax << "%                          ";
		if(!(j%500) && j!=0) annealing_kernel<<<2,1>>>(dev_aoa, dev_iTk, dev_hits, j);
		
	}

	cout << "\r100%                   " << endl;

	// This is a comment
        energy_kernel<<<blocks, threads>>>(dev_nx, dev_ny, dev_inp, dev_blockEnergies);
	danErrHndl( cudaMemcpy(blockEnergies, dev_blockEnergies, xblocks*yblocks*sizeof(double), cudaMemcpyDeviceToHost) );
	energy = 0;
	for(i=0; i<xblocks*yblocks; i++)
	{
		energy += blockEnergies[i];
	}

	cout << "Final energy is: " << energy << endl;

	// Get the finished arrays back and dump to file in a dans-gnuplot-script friendly way
	cudaMemcpy(nx, dev_nx, arraySize*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(ny, dev_ny, arraySize*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&aoa, dev_aoa, sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&iTk, dev_iTk, sizeof(double), cudaMemcpyDeviceToHost);
	cout << aoa << " " << iTk << endl;
	outputGrid(nx, ny, inp, filename);
	
	return 0;
}

__global__ void annealing_kernel(double *aoa, double *iTk, int *hits, int j)
{
	if(blockIdx.x == 0) // deal with the angle of acceptance
	{
		double oldaoa = *aoa;
		int totalHits = 0;

		for(int i=0;i<xblocks*yblocks;i++)
		{
			totalHits += hits[i];
			hits[i] = 0;
		}

		if( *aoa > 0.1 ) *aoa *= 2 * (double) totalHits / (500*width*height);
		if( *aoa > PI*0.5) *aoa = 0.5*PI;
		if( *aoa < 0.1 ) *aoa = 0.1;
	}
	else if(blockIdx.x == 1) // deal with the temperature
	{
		if(!(j%150000))
		{
			*iTk *= 1.01;
		}
	}
	else // break stuff
	{
		*aoa = -9.87e654321;
		*iTk = 1.2345e67890;
		hits[0] = -42;
	}
}

__global__ void monte_kernel(double *nx, double *ny, bool *inp, curandState *state, double *aoa, double *iTk, int *hits, int offset)
{

	// calculate cell of interest
	int threadx = threadIdx.x + blockIdx.x * blockDim.x;
	int thready = threadIdx.y + blockIdx.y * blockDim.y;
	int blockID = blockIdx.x + blockIdx.y * gridDim.x;
	int ID = threadx + thready * blockDim.x * gridDim.x;
	int offsetx = offset%4;
	int offsety = offset/4;
	int x = (thready%2 ? 4*threadx+2 : 4*threadx) + offsetx;
	int y = 2*thready + offsety;
	int index = getIndex(x,y);

	// Don't mess with nanoparticles
	if(inp[index]) return;

	double before=0, after=0, dE, rollOfTheDice, angle = PI*(*aoa)*(2*curand_uniform(&state[ID])-1)/180;
	double oldNx = nx[index];
	double oldNy = ny[index];

	before = calcEnergy(x,y,nx,ny);
	before += calcEnergy(x+1,y,nx,ny);
	before += calcEnergy(x-1,y,nx,ny);
	before += calcEnergy(x,y+1,nx,ny);
	before += calcEnergy(x,y-1,nx,ny);

	//rotate director anti-clockwise by angle "angle"
	nx[index] = cos(angle)*oldNx - sin(angle)*oldNy;
	ny[index] = sin(angle)*oldNx + cos(angle)*oldNy;

	after = calcEnergy(x,y,nx,ny);
	after += calcEnergy(x+1,y,nx,ny);
	after += calcEnergy(x-1,y,nx,ny);
	after += calcEnergy(x,y+1,nx,ny);
	after += calcEnergy(x,y-1,nx,ny);

	dE = after - before;

	// Decide the fate of the change
	if(dE>0)
	{
		rollOfTheDice = curand_uniform(&state[ID]);
		if(rollOfTheDice > exp(-dE*(*iTk))) // then reject change
		{
			nx[index] = oldNx;
			ny[index] = oldNy;
		}
		else atomicAdd(&(hits[blockID]), 1);
	}
	else atomicAdd(&(hits[blockID]), 1);
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
