#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

#include <stdio.h>
#include "gputimer.h"

__device__
unsigned int floatFlip(unsigned int value)
{
	unsigned int mask = (-(value >> 31)) | 0x80000000;
	return value ^ mask;
}

__device__ 
unsigned int floatFlipInverse(unsigned int value)
{
	int mask = ((value >> 31) - 1) | 0x80000000;
	return value ^ mask;
}

__global__ 
void convertFloats(unsigned int* vals, int num_elems)
{
	int lid = threadIdx.x;
	int gid = lid + (blockDim.x * blockIdx.x);
	if(gid >= num_elems)
	{
		return;
	}
	vals[gid] = floatFlip(vals[gid]);
}

__global__ 
void invertFloats(unsigned int* vals, int num_elems)
{
	int lid = threadIdx.x;
	int gid = lid + (blockDim.x * blockIdx.x);
	if(gid >= num_elems)
	{
		return;
	}
	vals[gid] = floatFlipInverse(vals[gid]);
}

__global__
void markFlags(unsigned int* vals, uint4* flags, int dual_pos, int num_elems)
{
	int lid = threadIdx.x;
	int gid = lid + (blockDim.x * blockIdx.x);
	if(gid >= num_elems)
	{
		return;
	}

	unsigned int val = vals[gid];
	unsigned int mask = 3 << (dual_pos * 2);
	val &= mask;
	val >>= (dual_pos * 2);

	uint4 flag = {0,0,0,0};
	*((&flag.x) + val) = 1;
	flags[gid] = flag;
}

__global__
void scanExclusiveSumWithBlockTotals(uint4* in, uint4* out, uint4* block_totals, int num_elems)
{
	extern __shared__ uint4 sh_vals[];

	int lid = threadIdx.x;
	int gid = lid + (blockDim.x * blockIdx.x);
	if(gid >= num_elems)
	{
		return;
	}

	sh_vals[lid] = in[gid];
	__syncthreads();

	for(int offset = 1; offset < blockDim.x; offset <<= 1)
	{
		int left = lid - offset;
		uint4 left_val;
		if(left >= 0)
		{
			left_val = sh_vals[left];
		}
		__syncthreads();
		if(left >= 0)
		{	
			uint4 val = sh_vals[lid];
			val.x += left_val.x;
			val.y += left_val.y;
			val.z += left_val.z;
			val.w += left_val.w;
			sh_vals[lid] = val;
		}
		__syncthreads();
	}

	if(lid == 0)
	{
		uint4 flag = {0,0,0,0};
		out[gid] = flag;
	}
	else
	{
		out[gid] = sh_vals[lid - 1];
	}

	if(lid == blockDim.x - 1 || gid == num_elems - 1)
	{
		block_totals[blockIdx.x] = sh_vals[lid];
	}
}

__global__
void scanInclusiveSum(uint4* in, uint4* out, int num_elems)
{
	extern __shared__ uint4 sh_vals[];

	int lid = threadIdx.x;
	int gid = lid + (blockDim.x * blockIdx.x);
	if(gid >= num_elems)
	{
		return;
	}

	sh_vals[lid] = in[gid];
	__syncthreads();

	for(int offset = 1; offset < blockDim.x; offset <<= 1)
	{
		int left = lid - offset;
		uint4 left_val;
		if(left >= 0)
		{
			left_val = sh_vals[left];
		}
		__syncthreads();
		if(left >= 0)
		{	
			uint4 val = sh_vals[lid];
			val.x += left_val.x;
			val.y += left_val.y;
			val.z += left_val.z;
			val.w += left_val.w;
			sh_vals[lid] = val;
		}
		__syncthreads();
	}
	out[gid] = sh_vals[lid];
}

__global__
void addBlockTotals(uint4* fours, uint4* block_totals, int num_elems)
{
	int lid = threadIdx.x;
	int gid = lid + (blockDim.x * blockIdx.x);
	if(gid >= num_elems)
	{
		return;
	}
	if(blockIdx.x == 0)
	{
		return;
	}

	uint4 val = fours[gid];
	uint4 offsets = block_totals[blockIdx.x - 1];

	val.x += offsets.x;
	val.y += offsets.y;
	val.z += offsets.z;
	val.w += offsets.w;

	fours[gid] = val;
}

__global__
void scatterAddresses(unsigned int* vals_in, 
					unsigned int* vals_out, 
					uint4* flags, 
					uint4* addresses, 
					int dual_pos,
					uint4 offsets,
					int num_elems)
{
	int lid = threadIdx.x;
	int gid = lid + (blockDim.x * blockIdx.x);
	if(gid >= num_elems)
	{
		return;
	}

	unsigned int val = vals_in[gid];
	unsigned int mask = 3 << (dual_pos * 2);
	unsigned int val_anded = val & mask;
	val_anded >>= (dual_pos * 2);

	uint4 addr_list = addresses[gid];
	unsigned int addr = *((&addr_list.x) + val_anded);
	unsigned int offset = *((&offsets.x) + val_anded);
	addr += offset;
	vals_out[addr] = val;
}

void radixSort(float* h_vals, int num_elems)
{
	int num_threads = 1024;
	int num_blocks = ceil(num_elems / ((float) num_threads));

	unsigned int* d_vals;
	unsigned int* d_vals_buffer;
	uint4* d_flags;
	uint4* d_addresses;
	uint4* d_block_totals;

	cudaMalloc(&d_vals, sizeof(unsigned int) * num_elems);
	cudaMalloc(&d_vals_buffer, sizeof(unsigned int) * num_elems);
	cudaMalloc(&d_flags, sizeof(uint4) * num_elems);
	cudaMalloc(&d_addresses, sizeof(uint4) * num_elems);
	cudaMalloc(&d_block_totals, sizeof(uint4) * num_blocks);

	cudaMemcpy(d_vals, h_vals, sizeof(unsigned int) * num_elems, cudaMemcpyHostToDevice);

    GpuTimer timer;
 	timer.Start();
	convertFloats<<<num_blocks, num_threads>>>(d_vals, num_elems);
	for(int bit_pair = 0; bit_pair < 16; bit_pair++)
	{
		markFlags<<<num_blocks, num_threads>>>(d_vals, d_flags, bit_pair, num_elems);

		scanExclusiveSumWithBlockTotals<<<num_blocks, num_threads, sizeof(uint4) * num_threads>>>(d_flags, d_addresses, d_block_totals, num_elems);
		scanInclusiveSum<<<1, num_blocks, sizeof(uint4) * num_blocks>>>(d_block_totals, d_block_totals, num_blocks);
		addBlockTotals<<<num_blocks, num_threads>>>(d_addresses, d_block_totals, num_elems);
		
		uint4 totals = {0,0,0,0};
		cudaMemcpy(&totals, &d_block_totals[num_blocks - 1], sizeof(uint4), cudaMemcpyDeviceToHost);
		uint4 offsets = {0, totals.x, totals.x + totals.y, totals.x + totals.y + totals.z};

		scatterAddresses<<<num_blocks, num_threads>>>(d_vals, d_vals_buffer, d_flags, d_addresses, bit_pair, offsets, num_elems);

		unsigned int* temp = d_vals;
		d_vals = d_vals_buffer;
		d_vals_buffer = temp;
	}
	invertFloats<<<num_blocks, num_threads>>>(d_vals, num_elems);
	timer.Stop();

	printf("My radixsort sorted %d keys in %g ms\n", num_elems, timer.Elapsed());


	cudaMemcpy(h_vals, d_vals, sizeof(float) * num_elems, cudaMemcpyDeviceToHost);
}

void myRadixSort(void)
{
	int num_elems = 1024 * 1024;

	float* h_vals;
	h_vals = (float*) malloc(sizeof(float) * num_elems);

	for(int i = 0; i < num_elems; i++)
	{
		h_vals[i] = (float) (i % 999999);
	}
	
	GpuTimer timer;
 	timer.Start();
  	
  	radixSort(h_vals, num_elems);

  	timer.Stop();

	printf("My radixsort sorted %d keys in %g ms\n", num_elems, timer.Elapsed());

	for(int i = 0; i < 100; i++)
	{
		//printf("%f \n", h_vals[i]);
	}

	for(int i = 1; i < num_elems; i++)
	{
		if(h_vals[i - 1] > h_vals[i])
		{
			printf("Error at index %d: %f > %f \n", i, h_vals[i - 1], h_vals[i]);
			break;
		}
	}
}

void thrustSort(void)
{
  	int N = 1000000;
  	thrust::host_vector<float> h_vec(N);
  	std::generate(h_vec.begin(), h_vec.end(), rand);

  	thrust::device_vector<float> d_vec = h_vec;

  	GpuTimer timer;
 	timer.Start();
  	thrust::sort(d_vec.begin(), d_vec.end());
  	timer.Stop();

  	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
  
  	printf("Thrust sorted %d keys in %g ms\n", N, timer.Elapsed());
}

int main(void)
{
    myRadixSort();
	thrustSort();
	return 0;
}