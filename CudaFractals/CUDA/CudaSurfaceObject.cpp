#include "CudaSurfaceObject.h"
#include "../Util.h"

CudaSurfaceObject::CudaSurfaceObject(CudaArray& cudaArray) :
   _cudaArray(cudaArray)
{
   cudaResourceDesc dsc;
   dsc.resType = cudaResourceTypeArray;
   dsc.res.array.array = cudaArray.GetCudaHandle();

   CUDA_VERIFY( cudaCreateSurfaceObject( &_cudaHandle, &dsc ) );
}

CudaSurfaceObject::~CudaSurfaceObject()
{
   CUDA_VERIFY( cudaDestroySurfaceObject( _cudaHandle ) );
}

std::unique_ptr<CudaSurfaceObject> MakeCudaSurfaceObject( CudaArray& cudaArray )
{
   return std::make_unique<CudaSurfaceObject>( cudaArray );
}