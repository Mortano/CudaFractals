#pragma once
#include "PathTracing.h"
#include "../CUDA/SamplingUtil.h"
#include "../CudaTypes.h"
#include <cfloat>

namespace path
{
   //! \brief Trace a shadow ray and return true if it hit an object
   __device__ bool isInShadow( const Ray& ray, DistFunc_t distEstimator )
   {
      const auto Iterations = 50;
      const auto DistMin = 0.002f;
      auto totalDistance = 0.f;
      for ( int i = 0; i < Iterations; i++ )
      {
         auto p = ray.origin + ray.dir * totalDistance;
         auto dist = distEstimator( p, nullptr );
         if ( dist <= DistMin )
         {
            return true;
         }
         if ( totalDistance >= 10 ) break;
         totalDistance += dist;
      }
      return false;
   }

   //! \brief Calculates the normal at the given point for the given distance estimator
   __device__ float3 normalAt( float3 point, DistFunc_t distEstimator )
   {
      const auto XDir = make_float3( 0.001f, 0, 0 );
      const auto YDir = make_float3( 0, 0.001f, 0 );
      const auto ZDir = make_float3( 0, 0, 0.001f );
      auto vec = make_float3( distEstimator( point + XDir, nullptr ) - distEstimator( point - XDir, nullptr ),
                              distEstimator( point + YDir, nullptr ) - distEstimator( point - YDir, nullptr ),
                              distEstimator( point + ZDir, nullptr ) - distEstimator( point - ZDir, nullptr ) );
      return Normalize( vec );
   }

   //! \brief Samples the background in the given direction
   __device__ uchar4 getBackground( float3 dir, cudaTextureObject_t cubemap )
   {
      return texCubemap<uchar4>( cubemap, dir.x, dir.y, dir.z );
   }

   //! \brief Trace a regular ray and return true if it hits something. In case of a hit, the hit point and hit normal 
   //! are stored in the respective out variables
   __device__ bool traceRay( const Ray& ray, float3& hitPoint, float3& hitNormal, float4& hitColor, DistFunc_t distEstimator )
   {
      const auto Iterations = 100;
      const auto DistMin = 0.002f;

      auto totalDistance = 0.f;
      for ( auto i = 0; i < Iterations; i++ )
      {
         auto p = ray.origin + ray.dir * totalDistance;
         auto dist = distEstimator( p, &hitColor );
         if ( dist <= DistMin )
         {
            hitPoint = p;
            hitNormal = normalAt( p, distEstimator );
            return true;
         }
         if ( totalDistance >= 10 ) break;
         totalDistance += dist;
      }
      return false;
   }

   __device__ bool traceRay_improved(const Ray& ray, float3& hitPoint, float3& hitNormal, float4& hitColor, DistFunc_t distEstimator)
   {
      auto omega = 1.2f; //Relaxation factor, will be set to 1 when relaxation failure is detected
      const auto MaxIterations = 64;
      const auto TMax = 20.f;
      const auto PixelRadius = 0.01f;

      auto t = 0.f;
      auto candidateError = FLT_MAX;
      auto candidateT = t;
      auto prevRadius = 0.f;
      auto stepLength = 0.f;
      auto functionSign = distEstimator( ray.origin, nullptr ) < 0 ? -1.f : 1.f;

      for( auto i = 0; i < MaxIterations; i++ )
      {
         auto signedRadius = functionSign * distEstimator( ray.origin + ray.dir * t, &hitColor );
         auto absRadius = fabs( signedRadius );

         auto sorFail = omega > 1 &&
                        ( absRadius + prevRadius ) < stepLength;
         if( sorFail )
         {
            //Relaxation failed, start at previous point with regular tracing
            stepLength -= ( omega * stepLength );
            omega = 1.f;
         }
         else
         {
            stepLength = signedRadius * omega;
         }

         prevRadius = absRadius;
         auto error = absRadius / t;
         if( !sorFail && error < candidateError )
         {
            candidateT = t;
            candidateError = error;
         }

         if ( !sorFail && error < PixelRadius || t > TMax ) break;
         t += stepLength;
      }

      if ( t > TMax || candidateError > PixelRadius ) return false;

      hitPoint = ray.origin + ray.dir * t;
      hitNormal = normalAt( hitPoint, distEstimator );
      return true;
   }

   __device__ float3 sampleAt( float x,
                               float y,
                               const CudaCamera& camera,
                               dim3 texDim,
                               cudaTextureObject_t cubemap,
                               curandState& randState,
                               DistFunc_t distFunc )
   {
      const auto RayDepth = 4;

      auto curRay = rayFromPixel( x, y, camera, texDim );
      auto luminosity = make_float3( 1, 1, 1 );

      float3 hitPoint, hitNormal;
      float4 hitColor;

      for ( auto i = 0; i < RayDepth; i++ )
      {
         if ( traceRay( curRay, hitPoint, hitNormal, hitColor, distFunc ) )
         {
            auto rndDir = Normalize( getCosineWeightedSample( hitNormal, randState ) );
            //luminosity = luminosity * 2.0 * Dot( rndDir, hitNormal );
            luminosity = luminosity * XYZ( hitColor ); //TODO Variable albedo
            curRay.dir = rndDir;
            curRay.origin = hitPoint + hitNormal * 0.01f;
         }
         else
         {
            auto bgColor = getBackground( curRay.dir, cubemap );
            return luminosity * float3FromUcharColor( bgColor );
         }
      }

      return make_float3( 0, 0, 0 );
   }

   __device__ float3 sampleAt_improved( float x, float y, const CudaCamera& camera, dim3 texDim, cudaTextureObject_t cubemap, curandState& randState, DistFunc_t distFunc )
   {
      const auto RayDepth = 4;

      auto curRay = rayFromPixel( x, y, camera, texDim );
      auto luminosity = make_float3( 1, 1, 1 );

      float3 hitPoint, hitNormal;
      float4 hitColor;

      for ( auto i = 0; i < RayDepth; i++ )
      {
         if ( traceRay_improved( curRay, hitPoint, hitNormal, hitColor, distFunc ) )
         {
            auto rndDir = Normalize( getCosineWeightedSample( hitNormal, randState ) );
            //luminosity = luminosity * 2.0 * Dot( rndDir, hitNormal );
            luminosity = luminosity * XYZ( hitColor ); //TODO Variable albedo
            curRay.dir = rndDir;
            curRay.origin = hitPoint + hitNormal * 0.01f;
         }
         else
         {
            auto bgColor = getBackground( curRay.dir, cubemap );
            return luminosity * float3FromUcharColor( bgColor );
         }
      }

      return make_float3( 0, 0, 0 );
   }
}
