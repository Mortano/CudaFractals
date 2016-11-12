#pragma once
#include <vector_functions.h>

#include "host_defines.h"
#include <math.h>
#include "VectorTypesUtil.h"

#pragma region Mat4f
__global__ struct Mat4f
{
   float4 right;
   float4 up;
   float4 forward;
   float4 position;
};

inline __device__ __host__ Mat4f make_mat4f( float4 right, float4 up, float4 forward, float4 position )
{
   Mat4f ret;
   ret.right = right;
   ret.up = up;
   ret.forward = forward;
   ret.position = position;
   return ret;
}
#pragma endregion

#pragma region CudaCamera
__global__ struct CudaCamera
{
   Mat4f matrix;
   float fov, focalLength;
};

//! \brief Creates a CudaCamera structure from the given values
inline __host__ __device__ CudaCamera make_cuda_camera( const Mat4f& matrix, float fov, float focalLength )
{
   CudaCamera ret;
   ret.focalLength = focalLength;
   ret.fov = fov;
   ret.matrix = matrix;
   return ret;
}

#pragma endregion

#pragma region Ray
__global__ struct Ray
{
   float3 origin;
   float3 dir;
};

inline __device__ Ray rayFromPixel( float x, float y, const CudaCamera& camera, dim3 wndSize )
{
   auto aspect = wndSize.x / static_cast<float>( wndSize.y );
   auto tanFov = tanf( camera.fov / 2 );

   auto cx = aspect * ( ( x * 2.f / wndSize.x ) - 1.f ) * tanFov;
   auto cy = ( ( y * 2.f / wndSize.y ) - 1.f ) * tanFov;

   auto fwd = XYZ( camera.matrix.forward );
   auto right = XYZ( camera.matrix.right );
   auto up = XYZ( camera.matrix.up );
   auto pos = XYZ( camera.matrix.position );

   Ray ret;
   ret.origin = pos + fwd * camera.focalLength + right*cx + up*cy;
   ret.dir = Normalize( ret.origin - pos );
   return ret;
}
#pragma endregion

#pragma region SurfaceDescription
__device__ __host__ struct SurfaceDescription
{
   cudaSurfaceObject_t surfaceObject;
   dim3 dimensions;
   dim3 offset;
};

inline __device__ __host__ SurfaceDescription make_surface_description( cudaSurfaceObject_t surfaceObject, dim3 dimensions, dim3 offset )
{
   SurfaceDescription ret;
   ret.dimensions = dimensions;
   ret.offset = offset;
   ret.surfaceObject = surfaceObject;
   return ret;
}
#pragma endregion

