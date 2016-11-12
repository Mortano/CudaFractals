#pragma once

#include "cuda_runtime.h"
#include <cmath>

#pragma region float2
__device__ inline float2 operator+( float2 l, float2 r )
{
   return make_float2( l.x + r.x, l.y + r.y );
}

__device__ inline float dot( float2 l, float2 r )
{
   return l.x*r.x + l.y*r.y;
}
#pragma endregion

#pragma region float3

__device__ inline float3 operator+( float3 l, float3 r )
{
   return make_float3( l.x + r.x, l.y + r.y, l.z + r.z );
}

__device__ inline float3 operator-( float3 l, float3 r )
{
   return make_float3( l.x - r.x, l.y - r.y, l.z - r.z );
}

__device__ inline float3 operator*( float3 f, float scalar )
{
   return make_float3( f.x * scalar, f.y * scalar, f.z * scalar );
}

__device__ inline float3 operator*( float3 l, float3 r )
{
   return make_float3( l.x * r.x, l.y*r.y, l.z* r.z );
}

__device__ inline float Magnitude( float3 f )
{
   return sqrtf( f.x * f.x + f.y * f.y + f.z * f.z );
}

__device__ inline float MagnitudeSqr( float3 f )
{
   return ( f.x * f.x + f.y * f.y + f.z * f.z );
}

__device__ inline float3 Normalize( float3 f )
{
   auto mag = 1.f / Magnitude( f );
   return make_float3( f.x * mag, f.y * mag, f.z * mag );
}

__device__ inline float AbsDot( float3 a, float3 b )
{
   return fabsf( a.x * b.x + a.y * b.y + a.z * b.z );
}

__device__ inline float Dot( float3 a, float3 b )
{
   return ( a.x * b.x + a.y * b.y + a.z * b.z );
}

__device__ inline float3 Cross( float3 a, float3 b )
{
   return make_float3( a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x );
}

__device__ inline float3 FoldXY( float3 f )
{
   auto tmpX = f.x;
   f.x = -f.y;
   f.y = -tmpX;
   return f;
}

__device__ inline float3 FoldXZ( float3 f )
{
   auto tmpX = f.x;
   f.x = -f.z;
   f.z = -tmpX;
   return f;
}

__device__ inline float3 FoldYZ( float3 f )
{
   auto tmpY = f.y;
   f.y = -f.z;
   f.z = -tmpY;
   return f;
}

__device__ inline float3 ModXY(float3 f, float mod)
{
   float ignore;
   f.x = modf( fabs(f.x) / mod, &ignore );
   f.y = modf( fabs(f.y) / mod, &ignore );
   return f;
}

__device__ inline float3 ModXZ( float3 f, float mod )
{
   float ignore;
   f.x = modf( fabs(f.x) / mod, &ignore );
   f.z = modf( fabs(f.z) / mod, &ignore );
   return f;
}

__device__ inline float3 ModYZ( float3 f, float mod )
{
   float ignore;
   f.z = modf( fabs(f.z) / mod, &ignore );
   f.y = modf( fabs(f.y) / mod, &ignore );
   return f;
}

__device__ inline float3 RotateX( float3 f, float angle )
{
   auto cosA = cos( angle );
   auto sinA = sin( angle );
   return make_float3( f.x, cosA * f.y - sinA * f.z, sinA * f.y + cosA * f.z );
}

//! \brief Converts a uchar4 color to a float3 color with range [0;1]
__device__ inline float3 float3FromUcharColor( uchar4 col )
{
   return make_float3( col.x / 255.f, col.y / 255.f, col.z / 255.f );
}

#pragma endregion

#pragma region float4
__device__ inline float3 XYZ( float4 v )
{
   return make_float3( v.x, v.y, v.z );
}

//! \brief Adds a float4 and a float3 component-wise
__device__ inline float4 operator+( const float4& l, const float3& r )
{
   return make_float4( l.x + r.x, l.y + r.y, l.z + r.z, l.w );
}
#pragma endregion

#pragma region uchar4
__device__ inline uchar4 operator+( uchar4 l, uchar4 r )
{
   return make_uchar4( l.x + r.x, l.y + r.y, l.z + r.z, l.w + r.w );
}

__device__ inline uchar4 operator*( uchar4 v, float scalar )
{
   return make_uchar4( static_cast<unsigned char>( v.x * scalar ),
                       static_cast<unsigned char>( v.y * scalar ),
                       static_cast<unsigned char>( v.z * scalar ),
                       static_cast<unsigned char>( v.w * scalar ) );
}

__device__ inline uchar4 operator*( uchar4 v, float3 i )
{
   return make_uchar4( static_cast<unsigned char>( v.x * i.x ),
                       static_cast<unsigned char>( v.y * i.y ),
                       static_cast<unsigned char>( v.z * i.z ),
                       v.w );
}
#pragma endregion