#pragma once

#include <vector>

#define _STR(x) #x
#define STR(x) _STR(x)

#define CUDA_VERIFY(call) { auto err = call; if(err != cudaSuccess){ fprintf( stderr, "Error %u in CUDA call %s\n", err, STR(call) ); __debugbreak(); return; } }
#define GL_VERIFY(call) call; { auto err = glGetError(); if(err != GL_NO_ERROR) { fprintf( stderr, "Error %u in OpenGL call %s\n", err, STR(call) ); __debugbreak(); } }

constexpr float DegToRad(float deg)
{
   return deg * 0.0174533f;
}

//! \brief Creates a cubemap texture with a color gradient
std::vector<std::vector<uint32_t>> MakeGradientCubemap( int width, int height, uint32_t topColor, uint32_t bottomColor );

//! \brief Creates a dummy cubemap
std::vector<std::vector<uint32_t>> MakeDummyCubemap( int width, int height );

//! \brief Creates a 32-bit RGBA color from the given float color values, all assumed to be in the [0;1] range
uint32_t MakeRGBA( float r, float g, float b, float a );

//! \brief Linear interpolation between two 32-bit RGBA colors
uint32_t LerpRGBA( uint32_t a, uint32_t b, float f );

void DumpToPNG( const std::vector<uint32_t>& data, int width, int height, const std::string& path );