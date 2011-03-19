#include "randgen.cpp"
#include <ctime>
#include <cmath>
#include <fstream>
#include <iostream>
using namespace std;

// wolframalpha ftw!
const double PI = 3.1415926535897932384626433832795028841971693993751058209749; 

// Initialises grid as random
void gridInit(double nx[], double ny[], bool inp[], int size);

// A proper mod function that always works
__device__ int mod(int a, int b);

// Returns a cell inside the grid for x, y
__device__ int getIndex(int x, int y);

// Calculate A dot B
__device__ double AdotB(double ax, double ay, double bx, double by);

// Calculates the energy of the cell
__device__ double calcEnergy(int x, int y, double *nx, double *ny);

// Dan's error handling CUDA
bool danErrHndl(cudaError_t error);

// Prints the grid to file
void outputGrid(double nx[], double ny[], bool inp[], char filename[]);

void gridInit(double nx[], double ny[], bool inp[], int size)
{
	int i;
	double angle; 
	setSeed();

	for(i=0; i<size-2; i++)
	{
		angle = rnd()*2*PI;
		nx[i] = cos(angle);
		ny[i] = sin(angle);
		inp[i] = false;
	}

	//Initialise Top boundary condition
	nx[i] = 1;
	ny[i] = 0;
	inp[i] = false;
	i++;
	
	//Initialise bottom boundary condition
	nx[i] = 1;
	ny[i] = 0;
	inp[i] = false;
}

__device__ inline int mod(int a, int b)
{
	return (a%b + b)%b;
}

__device__ int getIndex(int x, int y)
{
	if(y<height && y>=0)
	{
		// Left and right boundary conditions are always periodic
		x = mod(x,width);
		return x + y*width;
	}
	else if(y >= height)
	{
		// Top boundary condition is held in penultimate array element
		return arraySize - 2;
	}
	else if (y < 0)
	{
		// Bottom boundary condition is held in the last array element
		return arraySize - 1; 
	}

	return -0xffffffff;
}

__device__ double AdotB(double ax, double ay, double bx, double by)
{
	return ax*bx + ay*by;
}

__device__ double calcEnergy(int x, int y, double *nx, double *ny)
{
	/*    |4|    y|
	*   |1|2|3|   |
	*     |0|     |_____ x 
	*/
	
	int index[5], flip = 1;
	double firstTerm = 0, secondTerm = 0, thirdTerm = 0;
	double dnxdx_f, dnxdx_b, dnxdy_f, dnxdy_b;
	double dnydx_f, dnydx_b, dnydy_f, dnydy_b;

	// Get indicies to use in derivative calculations
	index[0] = getIndex(x, y-1);
	index[1] = getIndex(x-1, y);
	index[2] = getIndex(x, y);
	index[3] = getIndex(x+1, y);
	index[4] = getIndex(x, y+1);

	// Do derivative calculations, flipping if smallest angle between vectors is > PI/2
	if( AdotB( nx[index[3]], ny[index[3]], flip*nx[index[2]], flip*ny[index[2]] )<0 ) flip *= -1;
	dnxdx_f = nx[index[3]] - flip*nx[index[2]];
	dnydx_f = ny[index[3]] - flip*ny[index[2]];

	if( AdotB( flip*nx[index[2]], flip*ny[index[2]], nx[index[1]], ny[index[1]] )<0 ) flip *= -1;
	dnxdx_b = flip*nx[index[2]] - nx[index[1]];
	dnydx_b = flip*ny[index[2]] - ny[index[1]];

	if( AdotB( nx[index[4]], ny[index[4]], flip*nx[index[2]], flip*ny[index[2]] )<0 ) flip *= -1;
	dnxdy_f = nx[index[4]] - flip*nx[index[2]];
	dnydy_f = ny[index[4]] - flip*ny[index[2]];

	if( AdotB( flip*nx[index[2]], flip*ny[index[2]], nx[index[0]], ny[index[0]] )<0 ) flip *= -1;
	dnxdy_b = flip*nx[index[2]] - nx[index[0]];
	dnydy_b = flip*ny[index[2]] - ny[index[0]];

	// Calculate each of the terms of the Frank equation
	firstTerm = (dnxdx_f + dnydy_f)*(dnxdx_f + dnydy_f);
	firstTerm += (dnxdx_b + dnydy_f)*(dnxdx_b + dnydy_f);
        firstTerm += (dnxdx_f + dnydy_b)*(dnxdx_f + dnydy_b);
        firstTerm += (dnxdx_b + dnydy_b)*(dnxdx_b + dnydy_b);
	firstTerm /= 4;

	secondTerm = nx[index[3]]*nx[index[3]] + ny[index[3]]*ny[index[3]];

	thirdTerm = (dnydx_f - dnxdy_f)*(dnydx_f - dnxdy_f);
	thirdTerm += (dnydx_b - dnxdy_f)*(dnydx_b - dnxdy_f);
	thirdTerm += (dnydx_f - dnxdy_b)*(dnydx_f - dnxdy_b);
	thirdTerm += (dnydx_b - dnxdy_b)*(dnydx_b - dnxdy_b);
	thirdTerm /= 4;

	// Put them all together for total energy
	return 0.5*(firstTerm + k1tok3*secondTerm*thirdTerm);
}

bool danErrHndl(cudaError_t error)
{
	switch(error)
	{
		case cudaSuccess:
		return true;


		default:
		cerr << "OMG I failed. I suck. Infact I suck in this particular way: " << cudaGetErrorString(error) << endl;
		return false;
	}	

}

void outputGrid(double nx[], double ny[], bool inp[], char filename[])
{
	int i;
	ofstream out(filename);
	if(!out)
	{
		cout << "Opening files is way overrated. You didn't really want it done anyway.\n";
		exit(-9);
	}

	out << "# Boundary Conditions\n";
	for(i=-1;i<width+1;i++) // top and bottom
	{
		out << (double) i - 0.5*nx[arraySize-1] << " " << -1.0 - 0.5*ny[arraySize-1] << " " << nx[arraySize-1] << " " << ny[arraySize-1] << endl;
		out << (double) i - 0.5*nx[arraySize-2] << " " << (double) height-0.5*ny[arraySize-2] << " " << nx[arraySize-2] << " " << ny[arraySize-2] << endl;
	}
	for(i=0;i<height;i++) // left and right periodic
	{
		out << -1.0 - 0.5*nx[width*(i+1)-1] << " " << (double) i- 0.5*ny[width*(i+1)-1] << " " << nx[width*(i+1)-1] << " " << ny[width*(i+1)-1] << endl;
		out << (double) width- 0.5*nx[width*i]  << " " << (double)i-0.5*ny[width*i] << " " << nx[width*i] << " " << ny[width*i] << endl;
	}

	out << "\n\n\n";

	out << "# Liquid Crystal bits\n";
	for(i=0; i<height*width; i++)
        {
		if(!inp[i]) out << (double) (i%width) - 0.5*nx[i] << " " << (double) (i/width) -0.5*ny[i] << " " << nx[i] << " " << ny[i] << endl;
	}

	out << "\n\n\n";

	out << "# Nanoparticles\n";
	for(i=0; i<height*width; i++)
	{
		if(inp[i]) out << (double) (i%width) -0.5*nx[i] << " " << (double) (i/width)-0.5*ny[i] << " " << nx[i] << " " << ny[i] << endl;
	}
}
