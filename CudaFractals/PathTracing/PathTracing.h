#pragma once
#include <host_defines.h>
#include <vector_functions.h>
#include <curand_kernel.h>

struct CudaCamera;
struct Mat4f;
struct Ray;
using DistFunc_t = float( *)( float3, float4* );

namespace path
{
   //! \brief Trace a shadow ray and return true if it hit an object
   __device__ bool isInShadow( const Ray& ray, DistFunc_t distEstimator );

   //! \brief Calculates the normal at the given point for the given distance estimator
   __device__ float3 normalAt( float3 point, DistFunc_t distEstimator );

   //! \brief Samples the background in the given direction
   __device__ uchar4 getBackground( float3 dir, cudaTextureObject_t cubemap );

   //! \brief Trace a regular ray and return true if it hits something. In case of a hit, the hit point, hit color and hit normal 
   //! are stored in the respective out variables
   __device__ bool traceRay( const Ray& ray, float3& hitPoint, float3& hitNormal, float4& hitColor, DistFunc_t distEstimator );

   //! \brief Trace a regular ray using improved tracing methods (e.g. over-relaxation)
   __device__ bool traceRay_improved( const Ray& ray, float3& hitPoint, float3& hitNormal, float4& hitColor, DistFunc_t distEstimator );

   __device__ float3 sampleAt( float x,
                                      float y,
                                      const CudaCamera& camera,
                                      dim3 texDim,
                                      cudaTextureObject_t cubemap,
                                      curandState& randState,
                                      DistFunc_t distFunc );

   //! \brief Sample light at the given pixel coordinate using improved sampling techniques (over-relaxation)
   __device__ float3 sampleAt_improved( float x,
                                        float y,
                                        const CudaCamera& camera,
                                        dim3 texDim,
                                        cudaTextureObject_t cubemap,
                                        curandState& randState,
                                        DistFunc_t distFunc );

}