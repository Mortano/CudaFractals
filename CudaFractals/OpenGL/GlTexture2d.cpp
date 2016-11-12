#include "GlTexture2d.h"
#include "../Util.h"

#include <GL\glew.h>
#include <GL\freeglut.h>


GLint TextureFormatToGl( TextureFormat format )
{
   switch ( format )
   {
   case TextureFormat::R: return GL_R;
   case TextureFormat::RGB: return GL_RGB;
   case TextureFormat::RGBA: return GL_RGBA;
   case TextureFormat::RGBA8: return GL_RGBA8;
   default: return 0;
   }
}

int FormatToBpp( TextureFormat format )
{
   switch ( format )
   {
   case TextureFormat::R: return 1;
   case TextureFormat::RGB: return 3;
   case TextureFormat::RGBA: return 4;
   default: return -1;
   }
}

GLint TextureDataFormatToGl( TextureDataFormat format )
{
   switch ( format )
   {
   case TextureDataFormat::Byte: return GL_UNSIGNED_BYTE;
   case TextureDataFormat::Float: return GL_FLOAT;
   default: return 0;
   }
}

int BytesPerChannel( TextureDataFormat dataFormat )
{
   switch ( dataFormat )
   {
   case TextureDataFormat::Byte: return sizeof( uint8_t );
   case TextureDataFormat::Float: return sizeof( float );
   default: return -1;
   }
}

GlTexture2d::GlTexture2d( int width, int height, TextureFormat format, TextureDataFormat dataFormat ) :
   _width( width ),
   _height( height ),
   _format( format ),
   _dataFormat( dataFormat )
{
   Create();
}

GlTexture2d::~GlTexture2d()
{
   auto handle = GetGlHandle();
   glDeleteTextures( 1, &handle );
}

void GlTexture2d::WriteData( const std::vector<uint32_t>& data ) const
{
   auto expectedSize = _width * _height * FormatToBpp( _format ) * BytesPerChannel( _dataFormat );
   if ( data.size() * sizeof( uint32_t ) != expectedSize ) throw std::exception( "Data size is wrong!" );

   GL_VERIFY( glBindTexture( GL_TEXTURE_2D, GetGlHandle() ) );

   auto glFormat = TextureFormatToGl( _format );
   auto glDataFormat = TextureDataFormatToGl( _dataFormat );
   GL_VERIFY( glTexImage2D( GL_TEXTURE_2D, 0, glFormat, _width, _height, 0, glFormat, glDataFormat, &data[0] ) );

   GL_VERIFY( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT ) );
   GL_VERIFY( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT ) );
   GL_VERIFY( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR ) );
   GL_VERIFY( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR ) );
}

GLuint GlTexture2d::CreateImpl()
{
   GLuint handle;
   GL_VERIFY( glGenTextures( 1, &handle ) );

   GL_VERIFY( glBindTexture( GL_TEXTURE_2D, handle ) );
   
   auto glFormat = TextureFormatToGl( _format );
   auto glDataFormat = TextureDataFormatToGl( _dataFormat );
   GL_VERIFY( glTexImage2D( GL_TEXTURE_2D, 0, glFormat, _width, _height, 0, glFormat, glDataFormat, nullptr ) );
   
   GL_VERIFY( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT ) );
   GL_VERIFY( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT ) );
   GL_VERIFY( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR ) );
   GL_VERIFY( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR ) );

   return handle;
}

GlTexture2dCube::GlTexture2dCube( int width, int height, TextureFormat format, TextureDataFormat dataFormat ) :
   _width( width ),
   _height( height ),
   _format( format ),
   _dataFormat( dataFormat )
{
   Create();
}

GlTexture2dCube::~GlTexture2dCube()
{
   auto handle = GetGlHandle();
   glDeleteTextures( 1, &handle );
}

void GlTexture2dCube::WriteData( const std::vector<std::vector<uint32_t>>& dataAllFaces ) const
{
   if ( dataAllFaces.size() != 6 ) throw std::exception( "6 data sets required!" );

   const auto expectedSize = _width * _height * FormatToBpp( _format ) * BytesPerChannel( _dataFormat );

   const auto firstFace = GL_TEXTURE_CUBE_MAP_POSITIVE_X;
   const auto glFormat = TextureFormatToGl( _format );
   const auto glDataFormat = TextureDataFormatToGl( _dataFormat );

   GL_VERIFY( glBindTexture( GL_TEXTURE_CUBE_MAP, GetGlHandle() ) );

   for ( auto idx = 0; idx < 6; idx++ )
   {
      const auto& dataCurFace = dataAllFaces[idx];
      if ( dataCurFace.size() * sizeof( uint32_t ) != expectedSize ) throw std::exception( "Data size is wrong!" );
      GL_VERIFY( glTexImage2D( firstFace + idx, 0, glFormat, _width, _height, 0, glFormat, glDataFormat, &dataCurFace[0] ) );
   }

   GL_VERIFY( glTexParameteri( GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_REPEAT ) );
   GL_VERIFY( glTexParameteri( GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_REPEAT ) );
   GL_VERIFY( glTexParameteri( GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR ) );
   GL_VERIFY( glTexParameteri( GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR ) );
}

GlTexture2dCube::Ptr GlTexture2dCube::CreateDummyTexture()
{
   const auto Width = 256;
   const auto Height = 256;
   auto ret = std::make_unique<GlTexture2dCube>( Width, Height );

   const auto TopColor = 0xffffffff;
   const auto BottomColor = 0xff0000ff;

   auto dataPerFace = MakeGradientCubemap( Width, Height, TopColor, BottomColor );

   ret->WriteData( dataPerFace );

   return std::move( ret );
}

GLuint GlTexture2dCube::CreateImpl()
{
   GLuint handle;
   glGenTextures( 1, &handle );
   return handle;
}
