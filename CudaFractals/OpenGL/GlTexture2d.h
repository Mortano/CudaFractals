#pragma once
#include "GlObject.h"
#include <cstdint>
#include <vector>
#include <memory>

enum class TextureFormat
{
   R,
   RGB,
   RGBA,
   RGBA8
};

enum class TextureDataFormat
{
   Byte,
   Float
};

class GlTexture2d : public GlObject
{
public:
   using Ptr = std::unique_ptr<GlTexture2d>;

   GlTexture2d( int width, int height, TextureFormat format = TextureFormat::RGBA, TextureDataFormat dataFormat = TextureDataFormat::Byte );
   ~GlTexture2d() override;

   auto GetWidth() const { return _width; }
   auto GetHeight() const { return _height; }

   void WriteData( const std::vector<uint32_t>& data ) const;
private:
   uint32_t CreateImpl() override;
   const int _width, _height;   
   const TextureFormat _format;
   const TextureDataFormat _dataFormat;
};

class GlTexture2dCube : public GlObject
{
public:
   using Ptr = std::unique_ptr<GlTexture2dCube>;

   GlTexture2dCube( int width, int height, TextureFormat format = TextureFormat::RGBA, TextureDataFormat dataFormat = TextureDataFormat::Byte );
   ~GlTexture2dCube() override;

   auto GetWidth() const { return _width; }
   auto GetHeight() const { return _height; }

   void WriteData( const std::vector<std::vector<uint32_t>>& dataAllFaces ) const;

   static Ptr CreateDummyTexture();
private:
   uint32_t CreateImpl() override;

   const int _width, _height;
   const TextureFormat _format;
   const TextureDataFormat _dataFormat;
};
