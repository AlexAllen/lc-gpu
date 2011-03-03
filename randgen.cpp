#include <ctime>

// Period Parameters
const int N = 624;
const int M = 397;
const unsigned int matrixA = 0x9908b0df; // constant vector a
const unsigned int upperMask = 0x80000000; // most significant w-r bits
const unsigned int lowerMask = 0x7fffffff; // least significant r bits

// Tempering Parameters

const unsigned int temperingMaskB = 0x9d2c5680;
const unsigned int temperingMaskC = 0xefc60000;
#define temperingShiftU(y) (y >> 11)
#define temperingShiftS(y) (y << 7)
#define temperingShiftT(y) (y << 15)
#define temperingShiftL(y) (y >> 18)

static unsigned long mt[N]; // the array for the state vector
static int mti = N+1; // mti == N+1 means mt[N] is not initialized

// initializing the array with a non-zero seed
void sgenrand(unsigned long seed)
{
	mt[0] = seed & 0xffffffff;
	for(mti=1; mti<N; mti++) mt[mti] = (69069 * mt[mti-1]) & 0xffffffff;
}

double rnd()
{
	unsigned long y;
	static unsigned long mag01[2] = { 0x0, matrixA };
	// mag01[x] = x * matrixA for x=0,1
		
	if(mti >= N) // generate N words at one time
	{
		int kk;

		// if sgenrand() has not been called a default initial seed is used
		if(mti == N+1) sgenrand(4357);

		for(kk=0;kk<N-M;kk++)
		{
			y = (mt[kk]&upperMask)|(mt[kk+1]&lowerMask);
			mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1];
		}

		for(;kk<N-1;kk++)
		{
			y = (mt[kk]&upperMask)|(mt[kk+1]&lowerMask);
			mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1];
		}
	
		y = (mt[N-1]&upperMask)|(mt[0]&lowerMask);
		mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1];

		mti = 0;
	}

	y = mt[mti++];
	y ^= temperingShiftU(y);
	y ^= temperingShiftS(y) & temperingMaskB;
	y ^= temperingShiftT(y) & temperingMaskC;
	y ^= temperingShiftL(y);

	return ( (double) y / (unsigned long) 0xffffffff);
}

unsigned long intRnd()
{
        unsigned long y;
        static unsigned long mag01[2] = { 0x0, matrixA };
        // mag01[x] = x * matrixA for x=0,1

        if(mti >= N) // generate N words at one time
        {
                int kk;

                // if sgenrand() has not been called a default initial seed is used
                if(mti == N+1) sgenrand(4357);

                for(kk=0;kk<N-M;kk++)
                {
                        y = (mt[kk]&upperMask)|(mt[kk+1]&lowerMask);
                        mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1];
                }

                for(;kk<N-1;kk++)
                {
                        y = (mt[kk]&upperMask)|(mt[kk+1]&lowerMask);
                        mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1];
                }

                y = (mt[N-1]&upperMask)|(mt[0]&lowerMask);
                mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1];

                mti = 0;
        }

        y = mt[mti++];
        y ^= temperingShiftU(y);
        y ^= temperingShiftS(y) & temperingMaskB;
        y ^= temperingShiftT(y) & temperingMaskC;
        y ^= temperingShiftL(y);

        return y;
}

void setSeed()
{
	time_t seed;

	seed = time(NULL);

	sgenrand(seed);
}
