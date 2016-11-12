#pragma once
#include <texture_types.h>
#include <driver_types.h>

#include <vector>
#include <stdint.h>

class CudaCubeMap
{
public:
   CudaCubeMap(int width, int height, const std::vector<std::vector<uint32_t>>& data );
   ~CudaCubeMap();

   cudaTextureObject_t GetCudaHandle() const { return _texture; }
private:
   cudaTextureObject_t _texture;
   cudaArray_t _textureStorage;

   const int _width, _height;
};
