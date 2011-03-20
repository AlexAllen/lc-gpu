#include <fstream>
#include <curand_kernel.h>
using namespace std;

const char fn[] = "annealing.dump";
const int threadDim = 8;
const int xblocks = 4;
const int yblocks = 4;
const int height = yblocks*threadDim;
const int width = xblocks*threadDim;
const int k1tok3 = 1;
const int arraySize = xblocks*threadDim * yblocks*threadDim + 2; // penultimate cell in array is top boundary, final is bottom boundry

#include "bitsnbobs.cu"

__global__ void monte_kernel(double *nx, double *ny, bool *inp, curandState *state, int *hits, double aoa, double iTk,  int offset);
__global__ void energy_kernel(double *nx, double *ny, bool *inp, double *blockEnergies);
__global__ void empty_kernel(int *hits)
{
	hits[blockIdx.x + blockIdx.y*gridDim.x] = 0;
}

int main()
{
	int i, j, loopMax = 5000000;
	double nx[arraySize], ny[arraySize], *dev_nx, *dev_ny;
	bool inp[arraySize], *dev_inp; // is nanoparticle
	char filename[] = "grid.dump";
	double energy, blockEnergies[xblocks*yblocks], *dev_blockEnergies;
	double aoa = PI*0.5, iTk = 0.05;
	int *dev_hits;
	curandState *dev_state;
	ofstream out(fn);
	if(!out)
	{
		cout << "Opening files FTL!" << endl;
		return -180;
	}

	// Initialise grid
	gridInit(nx, ny, inp, arraySize);

	// Nanoparticle adders go here

	// Allocate space and copy stuff on/to GPU memory
	danErrHndl( cudaMalloc( (void**) &dev_nx, arraySize*sizeof(double) ) );
	danErrHndl( cudaMalloc( (void**) &dev_ny, arraySize*sizeof(double) ) );
	danErrHndl( cudaMalloc( (void**) &dev_inp, arraySize*sizeof(bool) ) );
	danErrHndl( cudaMalloc( (void**) &dev_hits, xblocks*yblocks*sizeof(int)/8 ) );
	danErrHndl( cudaMalloc( (void**) &dev_state, (arraySize-2)*sizeof(curandState) ) );
	danErrHndl( cudaMalloc( (void**) &dev_blockEnergies, xblocks*yblocks*sizeof(double) ) );
	danErrHndl( cudaMemcpy( dev_nx, nx, arraySize*sizeof(double), cudaMemcpyHostToDevice ) );
	danErrHndl( cudaMemcpy( dev_ny, ny, arraySize*sizeof(double), cudaMemcpyHostToDevice ) );
	danErrHndl( cudaMemcpy( dev_inp, inp, arraySize*sizeof(bool), cudaMemcpyHostToDevice ) );

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

	int totalHits, hits[xblocks*yblocks/8];

	// The monte carlo loop
	for(j=0;j<loopMax;j++)
	{
		for(i=0;i<8;i++)
		{
			monte_kernel<<<lessBlocks, threads>>>(dev_nx, dev_ny, dev_inp, dev_state, dev_hits, aoa, iTk, intRnd()%8);
		}

		if(!(j%10))
		{
			danErrHndl( cudaMemcpy(hits, dev_hits, xblocks*yblocks*sizeof(int)/8, cudaMemcpyDeviceToHost) );
			
	        	totalHits = 0;
	        	for(int i=0;i<xblocks*yblocks/8;i++)
	        	{
	        	        totalHits += hits[i];
	       		 	hits[i] = 0;
	        	}

	        	if( aoa > 0.002 ) aoa *= 2 * (double) totalHits / (width*height);
	        	if( aoa > PI*0.5) aoa = 0.5*PI;
	       		if( aoa < 0.002 ) aoa = 0.002;
		}

	        if(!(j%1000))
	        {
	        	iTk *= 1.01;
			if(iTk > 1e7 ) iTk = 1e7;
	        }

		if(!(j%1000))
		{
			energy_kernel<<<blocks, threads>>>(dev_nx, dev_ny, dev_inp, dev_blockEnergies);
			danErrHndl( cudaMemcpy(blockEnergies, dev_blockEnergies, xblocks*yblocks*sizeof(double), cudaMemcpyDeviceToHost) );
			energy = 0;
       			for(int k=0; k<xblocks*yblocks; k++)
        		{
                		energy += blockEnergies[k];
        		}

			out << j << " " << aoa << " " << iTk << " " << energy << endl;
		}

		danErrHndl( cudaMemcpy(dev_hits, hits, xblocks*yblocks*sizeof(int)/8, cudaMemcpyHostToDevice) );

		
		if(!(j%100)) cout << "\r" << (double) 100 * j / loopMax << "%                          ";
		
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
	danErrHndl( cudaMemcpy(nx, dev_nx, arraySize*sizeof(double), cudaMemcpyDeviceToHost));
	danErrHndl( cudaMemcpy(ny, dev_ny, arraySize*sizeof(double), cudaMemcpyDeviceToHost));
	cout << aoa << " " << iTk << endl;
	outputGrid(nx, ny, inp, filename);
	
	return 0;
}

__global__ void monte_kernel(double *nx, double *ny, bool *inp, curandState *state, int *hits, double aoa, double iTk, int offset)
{
	__shared__ int localHits[threadDim*threadDim]; // for local people 

	// calculate cell of interest
	int threadx = threadIdx.x + blockIdx.x * blockDim.x;
	int thready = threadIdx.y + blockIdx.y * blockDim.y;
	int blockID = blockIdx.x + blockIdx.y * gridDim.x;
	int globalID = threadx + thready * blockDim.x * gridDim.x;
	int threadID = threadIdx.x + threadIdx.y * blockDim.x;
	int offsetx = offset%4;
	int offsety = offset/4;
	int x = (thready%2 ? 4*threadx+2 : 4*threadx) + offsetx;
	int y = 2*thready + offsety;
	int index = getIndex(x,y);

	// Don't mess with nanoparticles
	if(inp[index]) return;

	localHits[threadID] = 0;

	double before=0, after=0, dE, rollOfTheDice, angle = PI*aoa*(2*curand_uniform(&state[globalID])-1)/180;
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
		rollOfTheDice = curand_uniform(&state[globalID]);
		if(rollOfTheDice > exp(-dE*iTk)) // then reject change
		{
			nx[index] = oldNx;
			ny[index] = oldNy;
		}
		else localHits[threadID]++;
	}
	else localHits[threadID]++;

	// sum localHits
	__syncthreads();

	int i = blockDim.x * blockDim.y/2;

	while(i>0)
	{
		if(threadID < i) localHits[threadID] += localHits[threadID + i];
		
		__syncthreads();
		i /= 2;
	}

	if(threadID == 0) hits[blockID] += localHits[0];
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
