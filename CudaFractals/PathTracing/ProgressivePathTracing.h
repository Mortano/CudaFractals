#pragma once
#include <host_defines.h>
#include <texture_types.h>
#include <curand_kernel.h>
#include "../CudaTypes.h"

namespace path
{

   //! \brief Performs progressive path tracing which continously accumulates samples for each pixel into a floating point texture
   //!
   __global__ void ProgressivePathTracingPass( SurfaceDescription surf,
                                               CudaCamera camera,
                                               cudaTextureObject_t cubemap,
                                               int samplesPerPixel,
                                               bool isInitialPass,
                                               curandState* randStates );
}
