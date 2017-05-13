#include <stdio.h>
#include <random>
#include <chrono>

#include "entity.cu"
#include "radix_sort.cu"

#define NUM_ENTITIES 1024
#define THREADS 1024
#define BLOCKS NUM_ENTITIES / THREADS

int main()
{
	using namespace std::chrono;
	srand((int) time(NULL));

	Entity* h_ents;
	h_ents = (Entity*) malloc(sizeof(Entity) * NUM_ENTITIES);

	float scope = 1000.0f;

	Entity* d_ents;
	cudaMalloc(&d_ents, sizeof(Entity) * NUM_ENTITIES);

	entityInitialize<<<BLOCKS, THREADS>>>(d_ents, NUM_ENTITIES);
	initRadixMemory();

	float time_delta = 0.1f;

	for(int i = 0; i < 500; i++)
	{
		radixSortFloat(d_ents, NUM_ENTITIES);
		entityProcess<<<BLOCKS, THREADS, sizeof(Entity) * THREADS>>>(d_ents, NUM_ENTITIES, time_delta);
	}

	cudaMemcpy(h_ents, d_ents, sizeof(Entity) * NUM_ENTITIES, cudaMemcpyDeviceToHost);

	for(int i = 0; i < NUM_ENTITIES; i++)
	{
		Entity ent = h_ents[i];
		//printf("pos: %f %f %f target: %f %f %f \n", ent.pos.x, ent.pos.y, ent.pos.z, ent.target.x, ent.target.y, ent.target.z);
	}
}
