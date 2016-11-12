#pragma once
#include "../OpenGL/GlWindow.h"
#include "../CUDA/CudaCubeMap.h"
#include "../Camera.h"

class ProgressiveTracer
{
public:
   ProgressiveTracer( int width, int height );

   void Run();
private:
   GlWindow _window;

   CudaCubeMap _cubeMap;
   Camera _cam;
};
