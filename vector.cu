struct Vec3
{
	float x;
	float y;
	float z;
};

__device__
Vec3 vec3Add(Vec3 a, Vec3 b)
{
	Vec3 added = {a.x + b.x, a.y + b.y, a.z + b.z};
	return added;
}

__device__
Vec3 vec3Sub(Vec3 a, Vec3 b)
{
	Vec3 added = {a.x - b.x, a.y - b.y, a.z - b.z};
	return added;
}

__device__
Vec3 vec3Scale(Vec3 a, float f)
{
	Vec3 scaled = {a.x * f, a.y * f, a.z * f};
	return scaled;
}

__device__
float vec3Mag(Vec3 a)
{
	return sqrt((a.x * a.x) + (a.y * a.y) + (a.z * a.z));
}

__device__
Vec3 vec3Normal(Vec3 a)
{
	float mag = vec3Mag(a);
	return vec3Scale(a, 1 / mag);
}

__device__ 
float vec3Dot(Vec3 a, Vec3 b)
{
	return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

__device__ 
Vec3 vec3Cross(Vec3 a, Vec3 b)
{
	Vec3 cross = {};
	cross.x = (a.y * b.z) - (a.z * b.y);
	cross.y = (a.z * b.x) - (a.x * b.z);
	cross.z = (a.x * b.y) - (a.y * b.x);
	return cross;
}

__device__ 
float vec3AngleBetween(Vec3 a, Vec3 b)
{
	float dot = vec3Dot(a, b);
	float mag_mult = vec3Mag(a) * vec3Mag(b);
	float ratio = dot / mag_mult;
	if(ratio > 1)
	{
		ratio = 1;
	}
	if(ratio < -1)
	{
		ratio = -1;
	}
	return acos(ratio);
}

/*
float floatRand()
{
	return ((double) rand() / (RAND_MAX));
}

Vec3 vec3Random(float scope)
{
	const float PI = 3.14159265358979f;
	float yaw = floatRand() * 2 * PI;
	float pitch = (floatRand() * 2 * PI) - PI;

	float x = cos(yaw) * cos(pitch);
	float y = sin(pitch);
	float z = sin(yaw) * cos(pitch);

	Vec3 vec = {x, y, z};
	return vec3Scale(vec, floatRand() * scope);
}*/