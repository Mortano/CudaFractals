#pragma once
#include <cmath>
#include <vector_functions.h>
#include "../VectorTypesUtil.h"

inline __device__ float distSphere( float3 point, float radius )
{
   return Magnitude( point ) - radius;
}
 
inline __device__ float distSierpinski( float3 point )
{
   const auto CenterX = 0.f;
   const auto CenterZ = 0.f;
   const auto Height = 1.5f;
   const auto ToCenter = Height * 1.f / 3.f; //tan²(30°)
   const auto wHalf = Height * 0.5773502f; //tan(30°)
   float3 a1 = make_float3( CenterX, Height, CenterZ ); //Top vertex
   float3 a2 = make_float3( CenterX, 0, CenterZ + 2 * ToCenter ); //Bottom back vertex
   float3 a3 = make_float3( -wHalf, 0, CenterZ - ToCenter ); //Bottom front left vertex
   float3 a4 = make_float3( wHalf, 0, CenterZ - ToCenter ); //Bottom front right vertex
   float3 c;
   int n = 0;
   float dist, d;
   const int Iterations = 10;
   const float Scale = 2.f;
   while ( n < Iterations )
   {
      c = a1; dist = MagnitudeSqr( point - a1 );
      d = MagnitudeSqr( point - a2 ); if ( d < dist ) { c = a2; dist = d; }
      d = MagnitudeSqr( point - a3 ); if ( d < dist ) { c = a3; dist = d; }
      d = MagnitudeSqr( point - a4 ); if ( d < dist ) { c = a4; dist = d; }
      point = point*Scale - c*( Scale - 1.0 );
      n++;
   }

   return Magnitude( point ) * pow( Scale, float( -n ) );
}

inline __device__ float distMandelbulb( float3 pos )
{
   float3 z = pos;
   float dr = 1.0f;
   float r = 0.0;
   const int Iterations = 10;
   const int Power = 4;
   for ( int i = 0; i < Iterations; i++ )
   {
      r = Magnitude( z );
      if ( r >= 2.f ) break;

      // convert to polar coordinates
      float theta = acos( z.z / r );
      float phi = atan2f( z.y, z.x );
      dr = pow( r, Power - 1.0f )*Power*dr + 1.0f;

      // scale and rotate the point
      float zr = pow( r, Power );
      theta = theta*Power;
      phi = phi*Power;

      // convert back to cartesian coordinates
      z = make_float3( sin( theta )*cos( phi ), sin( phi )*sin( theta ), cos( theta ) ) * zr;
      z = z + pos;
   }
   return 0.5f*log( r )*r / dr;
}

inline __device__ float distGround( float3 pos, float y )
{
   return fabs( pos.y - y );
}