#pragma once

#include <cuda_runtime_api.h>
#include "CudaArray.h"

class CudaSurfaceObject;

std::unique_ptr<CudaSurfaceObject> MakeCudaSurfaceObject( CudaArray& cudaArray );

class CudaSurfaceObject
{
public:
   using Ptr = std::unique_ptr<CudaSurfaceObject>;

   explicit             CudaSurfaceObject( CudaArray& cudaArray );
                        ~CudaSurfaceObject();

                        CudaSurfaceObject( const CudaSurfaceObject& ) = delete;
                        CudaSurfaceObject& operator=( const CudaSurfaceObject& ) = delete;

   auto                 GetCudaHandle() const { return _cudaHandle; }
private:
   cudaSurfaceObject_t  _cudaHandle;
   CudaArray&           _cudaArray;
};
