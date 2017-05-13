int* partial_reduces;
int* partial_reduces_buffer;

__device__
int intAdd(int a, int b)
{
	return a + b;
}

__device__
int intMax(int a, int b)
{
	if(a > b)
	{
		return a;
	}
	else
	{
		return b;
	}
}

__device__
int intMin(int a, int b)
{
	if(a < b)
	{
		return a;
	}
	else
	{
		return b;
	}
}

__global__
void reduce(int* nums, int* reduces, int num_elems, int (*func) (int, int))
{
	extern __shared__ int sh_nums[];

	int local_index = threadIdx.x;
	int global_offset = blockDim.x * blockIdx.x;
	int global_index = local_index + global_offset;

	if(global_index >= num_elems)
	{
		return;
	}

	sh_nums[local_index] = nums[global_index];
	__syncthreads();

	for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		int right = local_index + offset;
		if(right + global_offset <= num_elems)
		{
			sh_nums[local_index] = (*func)(sh_nums[local_index], sh_nums[right]);
		}
		__syncthreads();
	}

	if(local_index == 0)
	{
		reduces[blockIdx.x] = sh_nums[local_index];
	}
}

void initReduce()
{
	cudaMalloc(&partial_reduces, sizeof(int) * 1024 * 1024);
	cudaMalloc(&partial_reduces_buffer, sizeof(int) * 1024 * 1024);
}

int sumNums(int* nums, int num_elems)
{
	const int num_threads = 1024;
	int num_blocks = ceil(num_elems / (float) num_threads);

	int* partial_one = partial_reduces;
	int* partial_two = partial_reduces_buffer;
	reduce<<<num_blocks, num_threads, sizeof(int) * num_threads>>>(nums, partial_one, num_elems, intAdd);

	while(num_blocks > 1)
	{
		int prev_num_blocks = num_blocks;
		num_blocks = ceil(num_blocks / (float) num_threads);
		reduce<<<num_blocks, num_threads, sizeof(int) * num_threads>>>(partial_one, partial_two, prev_num_blocks, intAdd);

		int* partial_temp = partial_one;
		partial_one = partial_two;
		partial_two = partial_temp;
	}

	int result = 0;
	cudaMemcpy(&result, &partial_one[0], sizeof(int), cudaMemcpyDeviceToHost);
	return result;
}