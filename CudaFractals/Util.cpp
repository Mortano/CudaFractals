#include "Util.h"
#include "lodepng.h"

std::vector<std::vector<uint32_t>> MakeGradientCubemap( int width, int height, uint32_t topColor, uint32_t bottomColor )
{
   const auto MakeGradient = []( auto begin, auto end, auto width, auto height, auto topColor, auto bottomColor )
   {
      for ( auto y = 0; y < height; y++ )
      {
         auto ratio = y / static_cast<float>( height );
         auto color = LerpRGBA( topColor, bottomColor, ratio );
         for ( auto x = 0; x < width; x++ )
         {
            *begin++ = color;
         }
      }
   };

   std::vector<std::vector<uint32_t>> dataPerFace;
   dataPerFace.reserve( 6 );
   for ( auto idx = 0; idx < 6; idx++ )
   {
      std::vector<uint32_t> data;
      data.resize( width * height );
      switch ( idx )
      {
      case 0: //POS_X
         MakeGradient( data.begin(), data.end(), width, height, topColor, bottomColor );
         break;
      case 1: //NEG_X
         MakeGradient( data.begin(), data.end(), width, height, topColor, bottomColor );
         break;
      case 2: //POS_Y
         MakeGradient( data.begin(), data.end(), width, height, topColor, topColor );
         break;
      case 3: //NEG_Y
         MakeGradient( data.begin(), data.end(), width, height, bottomColor, bottomColor );
         break;
      case 4: //POS_Z
         MakeGradient( data.begin(), data.end(), width, height, topColor, bottomColor );
         break;
      case 5: //NEG_Z
         MakeGradient( data.begin(), data.end(), width, height, topColor, bottomColor );
         break;
      default: break;
      }
      dataPerFace.push_back( std::move( data ) );
   }
   return dataPerFace;
}

std::vector<std::vector<uint32_t>> MakeDummyCubemap( int width, int height )
{
   const auto MakeSolidColor = []( auto begin, auto end, auto width, auto height, auto color )
   {
      for ( auto y = 0; y < height; y++ )
      {
         for ( auto x = 0; x < width; x++ )
         {
            *begin++ = color;
         }
      }
   };

   std::vector<std::vector<uint32_t>> dataPerFace;
   dataPerFace.reserve( 6 );
   for ( auto idx = 0; idx < 6; idx++ )
   {
      std::vector<uint32_t> data;
      data.resize( width * height );
      switch ( idx )
      {
      case 0: //POS_X
         MakeSolidColor( data.begin(), data.end(), width, height, 0xff0000ff );
         break;
      case 1: //NEG_X
         MakeSolidColor( data.begin(), data.end(), width, height, 0xff00ffff );
         break;
      case 2: //POS_Y
         MakeSolidColor( data.begin(), data.end(), width, height, 0xff00ff00 );
         break;
      case 3: //NEG_Y
         MakeSolidColor( data.begin(), data.end(), width, height, 0xffffff00 );
         break;
      case 4: //POS_Z
         MakeSolidColor( data.begin(), data.end(), width, height, 0xffff0000 );
         break;
      case 5: //NEG_Z
         MakeSolidColor( data.begin(), data.end(), width, height, 0xffffffff );
         break;
      default: break;
      }
      dataPerFace.push_back( std::move( data ) );
   }
   return dataPerFace;
}

uint32_t MakeRGBA( float r, float g, float b, float a )
{
   auto ur = static_cast<uint32_t>( r * 255 );
   auto ug = static_cast<uint32_t>( g * 255 );
   auto ub = static_cast<uint32_t>( b * 255 );
   auto ua = static_cast<uint32_t>( a * 255 );
   return ur | ( ug << 8 ) | ( ub << 16 ) | ( ua << 24 );
}

uint32_t LerpRGBA( uint32_t a, uint32_t b, float f )
{
   auto ar = a & 0xff;
   auto ag = ( a >> 8 ) & 0xff;
   auto ab = ( a >> 16 ) & 0xff;
   auto aa = ( a >> 24 ) & 0xff;

   auto br = b & 0xff;
   auto bg = ( b >> 8 ) & 0xff;
   auto bb = ( b >> 16 ) & 0xff;
   auto ba = ( b >> 24 ) & 0xff;

   auto oneMinusF = 1.f - f;

   auto rr = static_cast<uint32_t>( ( ar * oneMinusF ) + ( br * f ) );
   auto rg = static_cast<uint32_t>( ( ag * oneMinusF ) + ( bg * f ) );
   auto rb = static_cast<uint32_t>( ( ab * oneMinusF ) + ( bb * f ) );
   auto ra = static_cast<uint32_t>( ( aa * oneMinusF ) + ( ba * f ) );

   return rr | ( rg << 8 ) | ( rb << 16 ) | ( ra << 24 );
}

void DumpToPNG( const std::vector<uint32_t>& data, int width, int height, const std::string& path )
{
   std::vector<std::uint8_t> PngBuffer( width * height * 4 );

   for ( auto y = 0; y < height; y++ )
   {
      for ( auto x = 0; x < width; x++ )
      {
         auto curPixel = data[y*width + x];
         auto newPos = ( y * width + x ) * 4;
         PngBuffer[newPos + 0] = static_cast<uint8_t>( ( curPixel >> 16 ) & 0xff ); //B is offset 2
         PngBuffer[newPos + 1] = static_cast<uint8_t>( ( curPixel >> 8 ) & 0xff ); //G is offset 1
         PngBuffer[newPos + 2] = static_cast<uint8_t>( ( curPixel >> 0 ) & 0xff ); //R is offset 0
         PngBuffer[newPos + 3] = static_cast<uint8_t>( ( curPixel >> 24 ) & 0xff ); //A is offset 3
      }
   }

   std::vector<std::uint8_t> ImageBuffer;
   auto err = lodepng::encode( ImageBuffer, PngBuffer, width, height );
   if( err != 0 )
   {
      fprintf( stderr, "Error encoding PNG data: %u\n", err );
      getchar();
      return;
   }
   err = lodepng::save_file( ImageBuffer, path );
   if ( err != 0 )
   {
      fprintf( stderr, "Error saving PNG data: %u\n", err );
      getchar();
      return;
   }
}
