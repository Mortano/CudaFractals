#include "CudaCubeMap.h"

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "../Util.h"
#include <cuda_runtime.h>

CudaCubeMap::CudaCubeMap(int width, int height, const std::vector<std::vector<uint32_t>>& data ) :
   _width(width),
   _height(height),
   _texture(0)
{
   _ASSERT( data.size() == 6 );

   //Linearize data (TODO Maybe just pass linearized memory from the start)
   std::vector<uint32_t> linearizedMemory( 6 * data[0].size() );
   for ( auto idx = 0; idx < 6; idx++ )
   {
      std::copy( data[idx].begin(), data[idx].end(), linearizedMemory.begin() + ( idx * data[idx].size() ) );
   }

   auto channelDesc = cudaCreateChannelDesc( 8, 8, 8, 8, cudaChannelFormatKindUnsigned );
   CUDA_VERIFY( cudaMalloc3DArray( &_textureStorage, &channelDesc, make_cudaExtent(width, height, 6), cudaArrayCubemap ) );

   cudaMemcpy3DParms memcpyParams = { 0 };
   memcpyParams.srcPtr = make_cudaPitchedPtr( &linearizedMemory[0], width * sizeof( uint32_t ), width, height );
   memcpyParams.srcPos = make_cudaPos( 0, 0, 0 );
   memcpyParams.dstArray = _textureStorage;
   memcpyParams.dstPos = make_cudaPos( 0, 0, 0 );
   memcpyParams.extent = make_cudaExtent( width, height, 6 );
   memcpyParams.kind = cudaMemcpyHostToDevice;
   CUDA_VERIFY( cudaMemcpy3D( &memcpyParams ) );

   cudaResourceDesc resourceDesc;
   memset( &resourceDesc, 0, sizeof( resourceDesc ) );
   resourceDesc.resType = cudaResourceTypeArray;
   resourceDesc.res.array.array = _textureStorage;
   
   cudaTextureDesc textureDesc;
   memset( &textureDesc, 0, sizeof( textureDesc ) );
   textureDesc.addressMode[0] = cudaAddressModeWrap;
   textureDesc.addressMode[1] = cudaAddressModeWrap;
   textureDesc.filterMode = cudaFilterModePoint;
   textureDesc.readMode = cudaReadModeElementType;
   textureDesc.normalizedCoords = 1;

   CUDA_VERIFY( cudaCreateTextureObject( &_texture, &resourceDesc, &textureDesc, nullptr ) );
}

CudaCubeMap::~CudaCubeMap()
{
   CUDA_VERIFY( cudaDestroyTextureObject( _texture ) );
   CUDA_VERIFY( cudaFreeArray( _textureStorage ) );
}
