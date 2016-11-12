#include "ProgressiveTracer.h"
#include "../Util.h"
#include <cuda_runtime_api.h>
#include "ProgressivePathTracing.h"
#include <device_launch_parameters.h>
#include "../CUDA/CudaSurfaceObject.h"
#include "../OpenGL/GlTexture2d.h"
#include <algorithm>
#include "../../packages/glm.0.9.7.1/build/native/include/glm/detail/type_vec3.hpp"

ProgressiveTracer::ProgressiveTracer(int width, int height) :
   _window(width, height),
   _cubeMap(256, 256, MakeGradientCubemap( 256, 256, 0xffffffff, 0xffe2904a ) )
{
   _cam.LookAt( glm::vec3( 0, -1, 0 ), glm::vec3( -0.4f, 1, -1.2f ), glm::vec3( 0, 1, 0 ) );
   _cam.SetFov( DegToRad( 120 ) );
   _cam.SetFocalLength( 0.1f );
}

__global__ void randInit( curandState* randStates, uint32_t seed )
{
   auto x = blockIdx.x * blockDim.x + threadIdx.x;
   auto y = blockIdx.y * blockDim.y + threadIdx.y;
   auto idx = ( y * blockDim.x * gridDim.x ) + x;

   curand_init( seed, idx, 0, &randStates[idx] );
}

__global__ void resolveSamplesToTexture(cudaSurfaceObject_t inputSurfaceObject, dim3 size, dim3 offset, cudaSurfaceObject_t outputSurfaceObject, float invCount)
{
   auto x = blockIdx.x * blockDim.x + threadIdx.x + offset.x;
   auto y = blockIdx.y * blockDim.y + threadIdx.y + offset.y;

   if ( x >= size.x || y >= size.y ) return;

   float4 sample;
   surf2Dread( &sample, inputSurfaceObject, x * sizeof( float4 ), y );

   auto r = static_cast<uint8_t>( sample.x * invCount * 255 );
   auto g = static_cast<uint8_t>( sample.y * invCount * 255 );
   auto b = static_cast<uint8_t>( sample.z * invCount * 255 );

   surf2Dwrite( make_uchar4( r, g, b, 0xff ), outputSurfaceObject, x * sizeof( uchar4 ), y );
}

void ProgressiveTracer::Run()
{
   //Query attributes of device and kernel
   cudaDeviceProp deviceProperties;
   CUDA_VERIFY( cudaGetDeviceProperties( &deviceProperties, 0 ) );

   const auto NumSMs = deviceProperties.multiProcessorCount;

   cudaFuncAttributes attributes;
   CUDA_VERIFY( cudaFuncGetAttributes( &attributes, path::ProgressivePathTracingPass ) );

   auto Split = 1;
   dim3 block( 32, 32 );
   const auto imgWidth = _window.GetWidth();
   const auto imgHeight = _window.GetHeight();
   dim3 grid( imgWidth / block.x, imgHeight / block.y );

   if ( block.x * block.y > attributes.maxThreadsPerBlock ) throw std::exception( "Thread number is too large!" );

   while ( grid.x > 2 &&
           grid.y > 2 &&
           ( grid.x * grid.y ) > 2 * NumSMs * 10 )
   {
      grid.x >>= 1;
      grid.y >>= 1;
      Split <<= 1;
   }

   const auto TotalThreadsX = block.x * grid.x;
   const auto TotalThreadsY = block.y * grid.y;
   const auto TotalThreadsPerDispatch = TotalThreadsX * TotalThreadsY;

   const auto SamplesPerPass = 1;
   auto totalSamples = 0;

   curandState* rndStates;
   CUDA_VERIFY( cudaMalloc( &rndStates, TotalThreadsPerDispatch * sizeof( curandState ) ) );

   randInit <<<grid, block >>>( rndStates, (uint32_t)std::chrono::duration_cast<std::chrono::nanoseconds>( std::chrono::high_resolution_clock::now().time_since_epoch() ).count() );

   auto& camMatrix = _cam.GetTransform();
   auto cudaCamMatrix = make_mat4f( make_float4( camMatrix[0][0], camMatrix[0][1], camMatrix[0][2], camMatrix[0][3] ),
                                    make_float4( camMatrix[1][0], camMatrix[1][1], camMatrix[1][2], camMatrix[1][3] ),
                                    make_float4( camMatrix[2][0], camMatrix[2][1], camMatrix[2][2], camMatrix[2][3] ),
                                    make_float4( camMatrix[3][0], camMatrix[3][1], camMatrix[3][2], camMatrix[3][3] ) );

   auto cudaCam = make_cuda_camera( cudaCamMatrix, DegToRad( 90 ), 1.f );

   auto cudaCubeMap = _cubeMap.GetCudaHandle();

   auto formatDesc = cudaCreateChannelDesc( 32, 32, 32, 32, cudaChannelFormatKindFloat );
   auto cudaArray = MakeCudaArray2D( imgWidth, imgHeight, formatDesc, nullptr, cudaArraySurfaceLoadStore );

   auto cudaSurface = MakeCudaSurfaceObject( *cudaArray );
   
   auto windowTextureCudaArray = MakeCudaArrayTexture2D( *_window.GetTexture() );
   auto drawSurface = MakeCudaSurfaceObject( *windowTextureCudaArray );

   struct Sample { float r, g, b, a; };
   std::vector<Sample> rawPixels( imgWidth * imgHeight );
   std::vector<uint32_t> pixels( imgWidth * imgHeight );


   auto firstPass = true;
   auto iterate = true;

   _window.AddKeyCallback([&](int key, int scancode, int action, int mods)
   {
      if ( key == 256 ) iterate = false; //256 == Escape key
   } );

   while( iterate )
   {
      for ( int y = 0; y < Split; y++ )
      {
         for ( int x = 0; x < Split; x++ )
         {
            dim3 offset( x * ( imgWidth / Split ), y * ( imgHeight / Split ) );
            dim3 texSize( imgWidth, imgHeight );
            auto surfaceDescription = make_surface_description( cudaSurface->GetCudaHandle(), texSize, offset );
            path::ProgressivePathTracingPass <<<grid, block>>> ( surfaceDescription, cudaCam, cudaCubeMap, SamplesPerPass, firstPass, rndStates );

            //CUDA_VERIFY( cudaDeviceSynchronize() );
         }
      }
      totalSamples += SamplesPerPass;

      if ( firstPass ) firstPass = false;

      //Resolve texture data and copy into window texture
      for ( int y = 0; y < Split; y++ )
      {
         for ( int x = 0; x < Split; x++ )
         {
            dim3 offset( x * ( imgWidth / Split ), y * ( imgHeight / Split ) );
            dim3 texSize( imgWidth, imgHeight );
            resolveSamplesToTexture << <grid, block >> > ( cudaSurface->GetCudaHandle(), texSize, offset, drawSurface->GetCudaHandle(), 1.f / totalSamples );
         }
      }

      //cudaArray->MemcpyFromDevice( &rawPixels[0] );
      //std::transform( rawPixels.begin(), rawPixels.end(), pixels.begin(), [&totalSamples]( const Sample& sample )
      //{
      //   auto invSamples = 1.f / totalSamples;
      //   auto r = static_cast<uint32_t>( sample.r * invSamples * 255 );
      //   auto g = static_cast<uint32_t>( sample.g * invSamples * 255 );
      //   auto b = static_cast<uint32_t>( sample.b * invSamples * 255 );
      //   return r | ( g << 8 ) | ( b << 16 ) | ( 0xff << 24 );
      //} );
      //
      //_window.GetTexture()->WriteData( pixels );
      _window.DoSingleFrame();
   }

   CUDA_VERIFY( cudaFree( rndStates ) );
}
