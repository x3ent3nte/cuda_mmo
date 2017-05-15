#include <stdio.h>

struct uint4
{
	unsigned int a;
	unsigned int b;
	unsigned int c;
	unsigned int d;
}

__device__
unsigned int flipFloat(unsigned int value)
{
	unsigned int mask = (-(value >> 31)) | 0x80000000;
	return value ^ mask;
}

__device__ 
unsigned int flipFloatInverse(unsigned int value)
{
	unsigned int mask = ((value >> 31) - 1) | 0x80000000;
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
	vals[gid] = flipFloat(vals[gid]);
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
	vals[gid] = flipFloatInverse(vals[gid]);
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
	unsigned int mask = 3 << dual_pos;
	val &= mask;
	val >>= dual_pos;

	uint4 flag = {0,0,0,0}
	flag[val]++;
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
			val[0] += left_val[0];
			val[1] += left_val[1];
			val[2] += left_val[2];
			val[3] += left_val[3];
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
		block_totals[blockIdx.x] = sh_nums[lid];
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
			val[0] += left_val[0];
			val[1] += left_val[1];
			val[2] += left_val[2];
			val[3] += left_val[3];
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

	val[0] += offsets[0];
	val[1] += offsets[1];
	val[2] += offsets[2];
	val[3] += offsets[3];

	fours[gid] = val;
}

__global__
void scatterAddresses(unsigned int* vals_in, 
					unsigned int* vals_out, 
					uint4* flags, 
					uint4* addresses, 
					unsigned int off1, 
					unsigned int off2, 
					unsigned int off3, 
					int num_elems)
{
	int lid = threadIdx.x;
	int gid = lid + (blockDim.x * blockIdx.x);
	if(gid >= num_elems)
	{
		return;
	}

	unsigned int val = vals_in[gid];
	unsigned int mask = 3 << dual_pos;
	unsigned int val_anded &= mask;
	val_anded >>= dual_pos;

	unsigned int addr = addresses[gid][val_anded];
	int offset = 0;
	switch(val_anded)
	{
		case 1: offset = off1; break;
		case 3: offset = off2; break;
		case 2: offset = off3; break;
	}
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
	uint4* addresses;
	uint4* d_block_totals;

	cudaMalloc(&d_vals, sizeof(unsigned int) * num_elems);
	cudaMalloc(&d_vals_buffer, sizeof(unsigned int) * num_elems);
	cudaMalloc(&d_flags, sizeof(uint4) * num_elems);
	cudaMalloc(&d_addresses, sizeof(uint4) * num_elems);
	cudaMalloc(&d_block_totals, sizeof(uint4) * num_blocks);

	cudaMemcpy(d_vals, h_vals, sizeof(float) * num_elems, cudaMemcpyHostToDevice);

	convertFloats<<<num_blocks, num_threads>>>(d_vals, num_elems);
	for(int bit_pair = 0; bit_pair < 16; bit_pair++)
	{
		markFlags<<<num_blocks, num_threads>>>(d_vals, d_flags);
		scanExclusiveSumWithBlockTotals <<<num_blocks, num_threads, sizeof(uint4) * num_elems>>>(d_flags, d_block_totals, num_elems);
		scanInclusiveSum<<<num_blocks, num_threads, sizeof(uint4) * num_elems>>>(d_block_totals, num_elems);
		addBlockTotals<<<num_blocks, num_threads>>>(d_flags, d_block_totals);
		
		uint4 totals = {0,0,0,0};
		cudaMemcpy(&totals, &d_block_totals[num_blocks - 1], sizeof(uint4), cudaMemcpyDeviceToHost);

		scatterAddresses<<<num_blocks, num_threads>>>(d_vals, d_vals_buffer, d_flags, d_addresses, totals.a, totals. a + totals.b, totals.a + totals.b + totals.c, num_elems);

		unsigned int* temp = d_vals;
		d_vals = d_vals_buffer;
		d_vals_buffer = temp;
	}
	invertFloats<<<num_blocks, num_threads>>>(d_vals, num_elems);

	cudaMemcpy(h_vals, d_vals, sizeof(float) * num_elems, num_elems, cudaMemcpyDeviceToHost);
}

int main()
{
	int num_elems = 1024 * 1024;

	float* h_vals;
	h_vals = (float*) malloc(sizeof(float) * num_elems)

	for(int i = 0; i < num_elems; i++)
	{
		h_vals[i] = num_elems - i;
	}

	radixSort(h_vals, num_elems);
}





















