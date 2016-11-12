#pragma once

#include <curand_kernel.h>
#include "CudaRandom.h"

//Mostly taken from http://blog.hvidtfeldts.net/ 

inline __device__ float3 ortho( float3 v )
{
   //  See : http://lolengine.net/blog/2013/09/21/picking-orthogonal-vector-combing-coconuts
   return abs( v.x ) > abs( v.z ) ? make_float3( -v.y, v.x, 0.0f ) : make_float3( 0.0f, -v.z, v.y );
}

//! \brief Returns a sample direction in a hemisphere around the given direction. The sample can be biased
inline __device__ float3 getSampleBiased( float3 dir, float power, curandState& randState )
{
   dir = Normalize( dir );
   float3 o1 = Normalize( ortho( dir ) );
   float3 o2 = Normalize( Cross( dir, o1 ) );
   auto r = make_float2( curand_uniform( &randState ), curand_uniform( &randState ) );
   r.x = r.x*2.f*3.14159f;
   r.y = powf( r.y, 1.0 / ( power + 1.0 ) );
   float oneminus = sqrt( 1.0 - r.y*r.y );
   return o1*cos( r.x )*oneminus + o2*sin( r.x )*oneminus + dir*r.y;
}

//! \brief Returns an unbiased random sample direction around the given direction
inline __device__ float3 getSample( float3 dir, curandState& randState )
{
   return getSampleBiased( dir, 0.0f, randState ); // <- unbiased!
}

inline __device__ float3 getCosineWeightedSample( float3 dir, curandState& randState )
{
   return getSampleBiased( dir, 1.0, randState );
}