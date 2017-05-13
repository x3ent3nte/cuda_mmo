#include <stdio.h>
#include <random>

unsigned int* vals_buffer;
unsigned int* pos_buffer;

unsigned int* flags;
unsigned int* addresses;
unsigned int* block_offsets;

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
void convertFloats(unsigned int* vals, unsigned int num_elems)
{
	int local_index = threadIdx.x;
	int global_index = local_index + (blockDim.x * blockIdx.x);

	if(global_index >= num_elems)
	{
		return;
	}

	vals[global_index] = floatFlip(vals[global_index]);
}

__global__
void invertFloats(unsigned int* vals, unsigned int num_elems)
{
	int local_index = threadIdx.x;
	int global_index = local_index + (blockDim.x * blockIdx.x);

	if(global_index >= num_elems)
	{
		return;
	}

	vals[global_index] = floatFlipInverse(vals[global_index]);
}

__global__
void flipFlags(unsigned int* flags, unsigned int num_elems)
{
	int local_index = threadIdx.x;
	int global_index = local_index + (blockDim.x * blockIdx.x);

	if(global_index >= num_elems)
	{
		return;
	}	

	flags[global_index] ^= 1;
}

__global__ 
void markFlags(unsigned int* vals, unsigned int* flags, bool high, unsigned int bit_pos, unsigned int num_elems)
{
	int local_index = threadIdx.x;
	int global_index = local_index + (blockDim.x * blockIdx.x);

	if(global_index >= num_elems)
	{
		return;
	}

	unsigned int value = vals[global_index]; 
	unsigned int mask = 1 << bit_pos;
	unsigned int anded = value & mask;

	if(high)
	{
		if(anded == 0)
		{
			flags[global_index] = 0;
		}
		else
		{
			flags[global_index] = 1;
		}
	}
	else
	{
		if(anded == 0)
		{
			flags[global_index] = 1;
		}
		else
		{
			flags[global_index] = 0;
		}
	}
}

__global__
void scanExclusiveSumWithBlockOffsets(unsigned int* nums, unsigned int* c_nums, unsigned int* block_offsets, unsigned int num_elems)
{
	extern __shared__ unsigned int sh_nums[];

	int local_index = threadIdx.x;
	int global_index = local_index + (blockDim.x * blockIdx.x);

	if(global_index >= num_elems)
	{
		return;
	}

	sh_nums[local_index] = nums[global_index];
	__syncthreads();

	for(int offset = 1; offset < blockDim.x; offset <<= 1)
	{
		int left = local_index - offset;
		int left_val = 0;
		if(left >= 0)
		{
			left_val = sh_nums[left];
		}
		__syncthreads();
		sh_nums[local_index] += left_val;
		__syncthreads();
	}

	if(local_index == 0)
	{
		c_nums[global_index] = 0;
	}
	else
	{
		c_nums[global_index] = sh_nums[local_index - 1];
	}

	if(local_index == blockDim.x - 1 || global_index == num_elems - 1)
	{
		block_offsets[blockIdx.x] = sh_nums[local_index];
	}

}

__global__
void scanInclusiveSum(unsigned int* nums, unsigned int* c_nums, unsigned int num_elems)
{
	extern __shared__ unsigned int sh_nums[];

    int local_index = threadIdx.x;
    int global_index = (blockDim.x * blockIdx.x) + local_index;

    if(global_index >= num_elems)
    {
        return;
    }

    sh_nums[local_index] = nums[global_index];
    __syncthreads();

    for(int offset = 1; offset < blockDim.x; offset <<= 1)
    {
        int left = local_index - offset;
        int left_val = 0;
        if(left >= 0)
        {
            left_val = sh_nums[left];
        }
        __syncthreads();
       	sh_nums[local_index] += left_val;
       
        __syncthreads();
    }

    c_nums[global_index] = sh_nums[local_index];
}

__global__ 
void addBlockOffsets(unsigned int* c_nums, unsigned int* block_offsets, unsigned int num_elems)
{
	int local_index = threadIdx.x;
    int global_index = (blockDim.x * blockIdx.x) + local_index;

    if(global_index >= num_elems)
    {
        return;
    }

    if(blockIdx.x == 0)
    {
    	return;
    }

    c_nums[global_index] += block_offsets[blockIdx.x - 1];
}

__global__
void scatterAddresses(unsigned int* vals_in, unsigned int* pos_in, unsigned int* vals_out, unsigned int* pos_out, unsigned int* flags, unsigned int* addresses, unsigned int offset, unsigned int num_elems)
{
	int local_index = threadIdx.x;
    int global_index = (blockDim.x * blockIdx.x) + local_index;

    if(global_index >= num_elems)
    {
        return;
    }

    if(flags[global_index] == 1)
    {
    	int addr = addresses[global_index] + offset;
    	vals_out[addr] = vals_in[global_index];
    	pos_out[addr] = pos_in[global_index];
    }
}

__global__
void setOrderFlag(unsigned int* vals, unsigned int* order_flag, unsigned int num_elems)
{
	int local_index = threadIdx.x;
    int global_index = (blockDim.x * blockIdx.x) + local_index;

    if(global_index >= num_elems)
    {
        return;
    }

    if(global_index == 0)
    {
    	return;
    }

    if(vals[global_index - 1] > vals[global_index])
    {
    	order_flag[0] = 1;
    }
}

bool isSorted(unsigned int* vals, unsigned int num_elems)
{
	unsigned int num_threads = 1024;
	unsigned int num_blocks = ceil(num_elems / (float) num_threads);

	unsigned int h_order_flag = 0;
	unsigned int* d_order_flag;
	cudaMalloc(&d_order_flag, sizeof(unsigned int));
	cudaMemset(d_order_flag, 0, sizeof(unsigned int));
	setOrderFlag<<<num_threads, num_blocks>>>(vals, d_order_flag, num_elems);
	cudaMemcpy(&h_order_flag, &d_order_flag[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
	return h_order_flag == 0;
}

void initRadixMemory(unsigned int num_elems)
{
	cudaMalloc(&vals_buffer, sizeof(unsigned int) * num_elems);
	cudaMalloc(&pos_buffer, sizeof(unsigned int) * num_elems);

	cudaMalloc(&flags, sizeof(unsigned int) * num_elems);
	cudaMalloc(&addresses, sizeof(unsigned int) * num_elems);
	cudaMalloc(&block_offsets, sizeof(unsigned int) * ceil(num_elems / 1024.0f));
}

void radixSortFloat(unsigned int* vals, unsigned int* pos, unsigned int num_elems)
{
	unsigned int* vals_one = vals;
	unsigned int* vals_two = vals_buffer;
	unsigned int* pos_one = pos;
	unsigned int* pos_two = pos_buffer;

	unsigned int num_threads = 1024;
	unsigned int num_blocks = ceil(num_elems / (float) num_threads);

	convertFloats<<<num_blocks, num_threads>>>(vals_one, num_elems);

	for(unsigned int bit_pos = 0; bit_pos < 32; bit_pos++)
	{	
		if(isSorted(vals_one, num_elems))
		{
			//printf("in order!!! \n");
			break;
		}

		markFlags<<<num_blocks, num_threads>>>(vals_one, flags, false, bit_pos, num_elems);
		scanExclusiveSumWithBlockOffsets<<<num_blocks, num_threads, sizeof(unsigned int) * num_threads>>>(flags, addresses, block_offsets, num_elems);
		scanInclusiveSum<<<1, num_blocks, sizeof(int) * num_blocks>>>(block_offsets, block_offsets, num_blocks);
		addBlockOffsets<<<num_blocks, num_threads>>>(addresses, block_offsets, num_elems);
		scatterAddresses<<<num_blocks, num_threads>>>(vals_one, pos_one, vals_two, pos_two, flags, addresses, 0, num_elems);
		
		unsigned int offset = 0;
		cudaMemcpy(&offset, &block_offsets[num_blocks - 1], sizeof(unsigned int), cudaMemcpyDeviceToHost);
		//printf("number of 0's %d \n", offset);

		flipFlags<<<num_blocks, num_threads>>>(flags, num_elems);
		scanExclusiveSumWithBlockOffsets<<<num_blocks, num_threads, sizeof(unsigned int) * num_threads>>>(flags, addresses, block_offsets, num_elems);
		scanInclusiveSum<<<1, num_blocks, sizeof(int) * num_blocks>>>(block_offsets, block_offsets, num_blocks);
		addBlockOffsets<<<num_blocks, num_threads>>>(addresses, block_offsets, num_elems);
		scatterAddresses<<<num_blocks, num_threads>>>(vals_one, pos_one, vals_two, pos_two, flags, addresses, offset, num_elems);
	
		unsigned int* vals_temp = vals_one;
		vals_one = vals_two;
		vals_two = vals_temp;

		unsigned int* pos_temp = pos_one;
		pos_one = pos_two;
		pos_two = pos_temp;
	}
	invertFloats<<<num_blocks, num_threads>>>(vals_one, num_elems);
	cudaMemcpy(vals, vals_one, sizeof(int) * num_elems, cudaMemcpyDeviceToDevice);
	cudaMemcpy(pos, pos_one, sizeof(int) * num_elems, cudaMemcpyDeviceToDevice);
}

int main()
{
	srand((int) time(NULL));
	unsigned int num_elems = 1024 * 1024;

	initRadixMemory(num_elems);

	float* h_vals;
	h_vals = (float*) malloc(sizeof(float) * num_elems);

	unsigned int* h_pos;
	h_pos = (unsigned int*) malloc(sizeof(unsigned int) * num_elems);

	for(int i = 0; i < num_elems; i++)
	{
		h_vals[i] = (((float) rand() / RAND_MAX) * 1000) - 500;
		//h_vals[i] = (float) i;
		h_pos[i] = i;
	}

	unsigned int* d_vals;
	cudaMalloc(&d_vals, sizeof(unsigned int) * num_elems);

	unsigned int* d_pos;
	cudaMalloc(&d_pos, sizeof(unsigned int) * num_elems);

	cudaMemcpy(d_vals, h_vals, sizeof(unsigned int) * num_elems, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos, h_pos, sizeof(unsigned int) * num_elems, cudaMemcpyHostToDevice);

	radixSortFloat(d_vals, d_pos, num_elems);
	
	cudaMemcpy(h_vals, d_vals, sizeof(unsigned int) * num_elems, cudaMemcpyDeviceToHost);

	for(int i = 0; i < 100; i++)
	{
		printf("%f \n", h_vals[i]);
	}

	return 0;
}








