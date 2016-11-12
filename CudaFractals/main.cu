#include <stdio.h>
#include "VectorTypesUtil.h"

#include <thread>
#include "Util.h"
#include "Camera.h"
#include "OpenGL/GlWindow.h"
#include "OpenGL/GlTexture2d.h"
#include "CUDA/CudaCubeMap.h"
#include "PathTracing/ProgressivePathTracing.h"

//#include <GLFW/glfw3.h>
#include "../packages/glm.0.9.7.1/build/native/include/glm/detail/type_vec3.hpp"
#include <algorithm>

//#include <Windows.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "surface_functions.hpp"
#include "surface_indirect_functions.hpp"
#include "cuComplex.h"
#include <curand_kernel.h>
#include <cfloat>
#include "CUDA/CudaSurfaceObject.h"
#include "PathTracing/ProgressiveTracer.h"

int main(int argc, char** argv)
{

   // Choose which GPU to run on, change this on a multi-GPU system.
   auto cudaStatus = cudaSetDevice( 0 );
   if ( cudaStatus != cudaSuccess )
   {
      fprintf( stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" );
      return;
   }

   ProgressiveTracer tracer( 512, 512 );
   tracer.Run();
   //InteractiveTracer();

   // cudaDeviceReset must be called before exiting in order for profiling and
   // tracing tools such as Nsight and Visual Profiler to show complete traces.
   cudaStatus = cudaDeviceReset();
   if ( cudaStatus != cudaSuccess )
   {
      fprintf( stderr, "cudaDeviceReset failed!" );
   }

   return 0;
}
