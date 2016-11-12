#pragma once

#include <cuda_runtime_api.h>

#include "../OpenGL/GlTexture2d.h"

//! \brief Encapsulates a cuda array
class CudaArray
{
public:
   using Ptr = std::unique_ptr<CudaArray>;

   CudaArray( const CudaArray& ) = delete;
   CudaArray& operator=( const CudaArray& ) = delete;

   virtual              ~CudaArray();

   auto                 GetCudaHandle() const { return _cudaHandle; }

   //! \brief Returns the extent of the underlying array. This is the number of entries in x, y and z
   virtual cudaExtent   GetExtent() const = 0;
   //! \brief Returns the number of bytes that a single entry requires
   virtual size_t       GetBytesPerEntry() const = 0;
   //! \brief Performs a memcpy from host to device memory, using the given memory block
   //! \param mem Host memory block
   virtual void         MemcpyToDevice( const void* mem ) = 0;
   //! \brief Performs a memcyp from device back to host memory, using the given memory block as destination
   //! \param mem Host memory block
   virtual void         MemcpyFromDevice( void* mem ) = 0;
protected:
   CudaArray();

   cudaArray_t          _cudaHandle;
};

CudaArray::Ptr MakeCudaArray1D( size_t size, cudaChannelFormatDesc formatDesc, const void* data = nullptr, unsigned int flags = 0 );
CudaArray::Ptr MakeCudaArray2D( size_t sizeX, size_t sizeY, cudaChannelFormatDesc formatDesc, const void* data = nullptr, unsigned int flags = 0 );
CudaArray::Ptr MakeCudaArrayTexture2D( GlTexture2d& texture );

namespace impl
{
   class CudaArray_1D : public CudaArray
   {
   public:
      CudaArray_1D( size_t size, cudaChannelFormatDesc formatDesc, const void* data, unsigned int flags );
      ~CudaArray_1D();

      cudaExtent                    GetExtent() const override;
      size_t                        GetBytesPerEntry() const override;
      void                          MemcpyToDevice( const void* mem ) override;
      void                          MemcpyFromDevice( void* mem ) override;
   private:
      const size_t                  _size;
      const cudaChannelFormatDesc   _formatDesc;
   };

   class CudaArray_2D : public CudaArray
   {
   public:
      CudaArray_2D( size_t sizeX, size_t sizeY, cudaChannelFormatDesc formatDesc, const void* data, unsigned int flags );
      ~CudaArray_2D();

      cudaExtent                    GetExtent() const override;
      size_t                        GetBytesPerEntry() const override;
      void                          MemcpyToDevice( const void* mem ) override;
      void                          MemcpyFromDevice( void* mem ) override;
   private:
      const size_t                  _sizeX, _sizeY;
      const cudaChannelFormatDesc   _formatDesc;
   };

   class CudaArray_Texture2D : public CudaArray
   {
   public:
      explicit                      CudaArray_Texture2D( GlTexture2d& glTexture );
      ~CudaArray_Texture2D();

      cudaExtent                    GetExtent() const override;
      size_t                        GetBytesPerEntry() const override;
      void                          MemcpyToDevice( const void* mem ) override;
      void                          MemcpyFromDevice( void* mem ) override;
   private:
      GlTexture2d&                  _glTexture;
      cudaGraphicsResource_t        _cudaGraphicsResource;
   };
}