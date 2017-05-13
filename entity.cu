#include "vector.cu"

struct Entity
{
	Vec3 pos;
	Vec3 target;
};

__device__
Entity entitySpawn(float scope)
{
	Vec3 pos = {1.0f, 1.0f, 1.0f};
	Vec3 target = {99.0f, 99.0f, 99.0f};

	Entity spawned = {pos, target};
	return spawned;
}

__global__
void entityInitialize(Entity* ents, int num_elems)
{
	int local_index = threadIdx.x;
	int global_index = local_index + (blockDim.x * blockIdx.x);

	if(global_index >= num_elems)
	{
		return;
	}

	Entity ent = entitySpawn(1000.0f);
	ents[global_index] = ent;
}

__global__
void entityProcess(Entity* ents, int num_elems, float time_delta)
{
	extern __shared__ Entity sh_ents[];

	int local_index = threadIdx.x;
	int global_index = local_index + (blockDim.x * blockIdx.x);

	if(global_index > num_elems)
	{
		return;
	}

	sh_ents[local_index] = ents[global_index];

	Entity ent = sh_ents[local_index];

	//move
	Vec3 delta = vec3Sub(ent.target, ent.pos);
	float delta_mag = vec3Mag(delta);

	if(delta_mag < 1.0f)
	{
		ent.pos = ent.target;
		//ent.target = vec3Add(ent.target, vec3Random(5000));
	}
	else
	{
		Vec3 delta_norm = vec3Scale(delta, 1 / delta_mag);
		ent.pos = vec3Add(ent.pos, delta_norm);
	}
	__syncthreads();
	//end move

	ents[global_index] = ent;
}