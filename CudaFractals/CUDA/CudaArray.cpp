#include "CudaArray.h"
#include "../Util.h"

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

//! \brief Returns the total size in bytes from the given format descriptor
size_t BytesFromFormat( const cudaChannelFormatDesc& desc )
{
   return static_cast<size_t>( desc.x / 8 + desc.y / 8 + desc.z / 8 + desc.w / 8 );
}

CudaArray::Ptr MakeCudaArray1D( size_t size, cudaChannelFormatDesc formatDesc, const void* data, unsigned int flags )
{
   return std::make_unique<impl::CudaArray_1D>( size, formatDesc, data, flags );
}

CudaArray::Ptr MakeCudaArray2D( size_t sizeX, size_t sizeY, cudaChannelFormatDesc formatDesc, const void* data, unsigned int flags )
{
   return std::make_unique<impl::CudaArray_2D>( sizeX, sizeY, formatDesc, data, flags );
}

CudaArray::Ptr MakeCudaArrayTexture2D( GlTexture2d& texture )
{
   return std::make_unique<impl::CudaArray_Texture2D>( texture );
}


CudaArray::~CudaArray()
{
}

CudaArray::CudaArray() :
   _cudaHandle( nullptr )
{
}

impl::CudaArray_1D::CudaArray_1D( size_t size,
                                  cudaChannelFormatDesc formatDesc,
                                  const void* data,
                                  unsigned int flags ) :
   _size( size ),
   _formatDesc( formatDesc )
{
   CUDA_VERIFY( cudaMallocArray( &_cudaHandle, &_formatDesc, size, 0, flags ) );

   CudaArray_1D::MemcpyToDevice( data );
}

impl::CudaArray_1D::~CudaArray_1D()
{
   CUDA_VERIFY( cudaFreeArray( _cudaHandle ) );
}

cudaExtent impl::CudaArray_1D::GetExtent() const
{
   return make_cudaExtent( _size, 0, 0 );
}

size_t impl::CudaArray_1D::GetBytesPerEntry() const
{
   return BytesFromFormat( _formatDesc );
}

void impl::CudaArray_1D::MemcpyToDevice( const void* mem )
{
   if ( !mem ) return;
   CUDA_VERIFY( cudaMemcpyToArray( _cudaHandle, 0, 0, mem, _size * GetBytesPerEntry(), cudaMemcpyHostToDevice ) );
}

void impl::CudaArray_1D::MemcpyFromDevice( void* mem )
{
   if ( !mem ) return;
   CUDA_VERIFY( cudaMemcpyFromArray( mem, _cudaHandle, 0, 0, _size * GetBytesPerEntry(), cudaMemcpyDeviceToHost ) );
}

impl::CudaArray_2D::CudaArray_2D( size_t sizeX,
                                  size_t sizeY,
                                  cudaChannelFormatDesc formatDesc,
                                  const void* data,
                                  unsigned int flags ) :
   _sizeX( sizeX ),
   _sizeY( sizeY ),
   _formatDesc( formatDesc )
{
   CUDA_VERIFY( cudaMallocArray( &_cudaHandle, &_formatDesc, sizeX, sizeY, flags ) );

   CudaArray_2D::MemcpyToDevice( data );
}

impl::CudaArray_2D::~CudaArray_2D()
{
   CUDA_VERIFY( cudaFreeArray( _cudaHandle ) );
}

cudaExtent impl::CudaArray_2D::GetExtent() const
{
   return make_cudaExtent( _sizeX, _sizeY, 0 );
}

size_t impl::CudaArray_2D::GetBytesPerEntry() const
{
   return BytesFromFormat( _formatDesc );
}

void impl::CudaArray_2D::MemcpyToDevice( const void* mem )
{
   if ( !mem ) return;
   CUDA_VERIFY( cudaMemcpy2DToArray( _cudaHandle, 0, 0, mem, _sizeX * GetBytesPerEntry(), _sizeX, _sizeY, cudaMemcpyHostToDevice ) );
}

void impl::CudaArray_2D::MemcpyFromDevice( void* mem )
{
   if ( !mem ) return;
   auto bpe = GetBytesPerEntry();
   CUDA_VERIFY( cudaMemcpy2DFromArray( mem, _sizeX * GetBytesPerEntry(), _cudaHandle, 0, 0, _sizeX * GetBytesPerEntry(), _sizeY, cudaMemcpyDeviceToHost ) );
}

impl::CudaArray_Texture2D::CudaArray_Texture2D( GlTexture2d& glTexture ) :
   _glTexture( glTexture ),
   _cudaGraphicsResource(nullptr)
{
   auto texHandle = _glTexture.GetGlHandle();
   CUDA_VERIFY( cudaGraphicsGLRegisterImage( &_cudaGraphicsResource, texHandle, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone ) );

   //Map to cuda and fill with some values
   CUDA_VERIFY( cudaGraphicsMapResources( 1, &_cudaGraphicsResource ) );
   CUDA_VERIFY( cudaGraphicsSubResourceGetMappedArray( &_cudaHandle, _cudaGraphicsResource, 0, 0 ) );
}

impl::CudaArray_Texture2D::~CudaArray_Texture2D()
{
   CUDA_VERIFY( cudaGraphicsUnmapResources( 1, &_cudaGraphicsResource ) );
}

cudaExtent impl::CudaArray_Texture2D::GetExtent() const
{
   return make_cudaExtent( _glTexture.GetWidth(), _glTexture.GetHeight(), 0 );
}

size_t impl::CudaArray_Texture2D::GetBytesPerEntry() const
{
   return 4; //TODO Actual size
}

void impl::CudaArray_Texture2D::MemcpyToDevice( const void* mem )
{
}

void impl::CudaArray_Texture2D::MemcpyFromDevice( void* mem )
{
}
