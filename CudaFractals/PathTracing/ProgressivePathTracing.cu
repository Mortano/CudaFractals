
#include "ProgressivePathTracing.h"
#include "../Fractals/DistanceEstimators.h"
#include <device_launch_parameters.h>
#include "PathTracing.h"
#include "../VectorTypesUtil.h"

namespace path
{

   __device__ float DistEstimator( float3 p, float4* color )
   {
      if ( !color )
      {
         return fmin(
            distSphere( ModXZ(p, 1.f) - make_float3(0.5f, 0.5f, 0.5f), 0.3f ),
            //distSierpinski( p ),
            //distMandelbulb(p*2.f + make_float3(0,1,0)),
            distGround( p, -0.6f )
         );
      }
      else
      {
         auto dist1 = distGround( p, -0.6f );
         auto dist2 = distSphere( ModXZ( p, 1.f ) - make_float3( 0.5f, 0.5f, 0.5f ), 0.3f );
         //auto dist2 = distSierpinski( p );
         if ( dist1 < dist2 )
         {
            *color = make_float4( 0.8f, 0.8f, 0.8f, 1 );
            return dist1;
         }
         *color = make_float4( 0.6f, 0.6f, 1, 1 );
         return dist2;
      }
   }

   //! \brief Performs progressive path tracing which continously accumulates samples for each pixel into a floating point texture
   //!
   __global__ void ProgressivePathTracingPass( SurfaceDescription surf,
                                               CudaCamera camera,
                                               cudaTextureObject_t cubemap,
                                               int samplesPerPixel,
                                               bool isInitialPass,
                                               curandState* randStates )
   {
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x + surf.offset.x;
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y + surf.offset.y;

      if ( x >= surf.dimensions.x || y >= surf.dimensions.y )
      {
         return;
      }

      //Fixed camera in [-2;2]² area, positioned at (0,0,-2)
      //const auto CamMatrix = make_mat4f( make_float4( 1, 0, 0, 0 ), make_float4( 0, 1, 0, 0 ), make_float4( 0, 0, 1, 0 ), make_float4( 0, 0.5f, -1.5f, 0 ) );   

      auto rndIdx = ( blockIdx.y * blockDim.y + threadIdx.y ) * blockDim.x * gridDim.x + ( blockIdx.x * blockDim.x + threadIdx.x );
      auto& curRndState = randStates[rndIdx];

      const int SuperSampling = 2;
      auto scaling = 1.f / SuperSampling;
      auto scalingSqr = scaling * scaling;
      auto color = make_float3( 0, 0, 0 );
      for ( int ssy = 0; ssy < SuperSampling; ssy++ )
      {
         for ( int ssx = 0; ssx < SuperSampling; ssx++ )
         {

            for ( auto curSample = 0; curSample < samplesPerPixel; curSample++ )
            {
               auto curColor = sampleAt_improved( surf.dimensions.y - y - 1 + ssy * scaling,
                                                  x + ssx * scaling,
                                                  camera,
                                                  surf.dimensions,
                                                  cubemap,
                                                  curRndState,
                                                  DistEstimator );
               color = color + ( curColor * scalingSqr );
            }

         }
      }

      if ( !isInitialPass )
      {
         //Add the sampled colors to the current color in the texture
         float4 curColorInTexture;
         surf2Dread( &curColorInTexture, surf.surfaceObject, x * sizeof( float4 ), y );
         curColorInTexture = curColorInTexture + color;
         surf2Dwrite( curColorInTexture, surf.surfaceObject, x * sizeof( float4 ), y );
      }
      else
      {
         //In the initial pass, we don't add the values, we just overwrite them, since this is the first pass and we would start with an empty texture anyways
         surf2Dwrite( make_float4( color.x, color.y, color.z, 1.f ), surf.surfaceObject, x * sizeof( float4 ), y );
      }
   }

}
