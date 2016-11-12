#include "GlWindow.h"

#include <GL\glew.h>
#include <GLFW/glfw3.h>

#include <stdio.h>

#include <vector>
#include <stdint.h>
#include <algorithm>
#include "../Util.h"

std::vector<GlWindow*> GlWindow::s_instances;

GlWindow::GlWindow( int width, int height ) :
   _width( width ),
   _height( height )
{
   s_instances.push_back( this );

   if( !glfwInit() )
   {
      fprintf( stderr, "Failed to initialize GLFW!" );
      return;
   }
   
   glfwWindowHint( GLFW_SAMPLES, 1 ); 
   glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 4 ); // We want OpenGL 4.0
   glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 0 );
   glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE ); // To make MacOS happy; should not be needed
   glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE );

   _window = glfwCreateWindow( width, height, "Fractals", nullptr, nullptr );
   if ( !_window )
   {
      fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
      glfwTerminate();
      return;
   }   

   glfwMakeContextCurrent( _window ); // Initialize GLEW
   glewExperimental = true; // Needed in core profile
   if ( glewInit() != GLEW_OK )
   {
      fprintf( stderr, "Failed to initialize GLEW\n" );
      return;
   }

   glfwSetKeyCallback( _window, GlobalKeyCallback );
   glfwSetCursorPosCallback( _window, GlobalMouseMoveCallback );

   _windowTexture = std::make_unique<GlTexture2d>( width, height );
}

GlWindow::~GlWindow()
{
   auto inst = std::find( s_instances.begin(), s_instances.end(), this );
   s_instances.erase( inst );
}

void GlWindow::RenderLoop()
{
   _lastTick = std::chrono::high_resolution_clock::now();
   while( true )
   {
      auto curTick = std::chrono::high_resolution_clock::now();
      auto delta = std::chrono::duration_cast<std::chrono::microseconds>( curTick - _lastTick );
      auto deltaMs = delta.count() / 1000.f;
      _lastTick = curTick;

      for ( auto& callback : _renderCallbacks ) callback( deltaMs );

      GL_VERIFY( glEnable( GL_TEXTURE_2D ) );
      GL_VERIFY( glActiveTexture( GL_TEXTURE0 ) );
      GL_VERIFY( glBindTexture( GL_TEXTURE_2D, _windowTexture->GetGlHandle() ) );
      
      glBegin( GL_QUADS );
      
      glVertex2f( -1, -1 );
      glTexCoord2f( 0, 0 );

      glVertex2f( 1, -1 );
      glTexCoord2f( 1, 0 );

      glVertex2f( 1, 1 );
      glTexCoord2f( 1, 1 );

      glVertex2f( -1, 1 );
      glTexCoord2f( 0, 1 );

      glEnd();

      glfwSwapBuffers( _window );
      glfwPollEvents();
   }
}

void GlWindow::DoSingleFrame()
{
   GL_VERIFY( glEnable( GL_TEXTURE_2D ) );
   GL_VERIFY( glActiveTexture( GL_TEXTURE0 ) );
   GL_VERIFY( glBindTexture( GL_TEXTURE_2D, _windowTexture->GetGlHandle() ) );

   glBegin( GL_QUADS );

   glVertex2f( -1, -1 );
   glTexCoord2f( 0, 0 );

   glVertex2f( 1, -1 );
   glTexCoord2f( 1, 0 );

   glVertex2f( 1, 1 );
   glTexCoord2f( 1, 1 );

   glVertex2f( -1, 1 );
   glTexCoord2f( 0, 1 );

   glEnd();

   glfwSwapBuffers( _window );
   glfwPollEvents();
}

void GlWindow::AddRenderCallback(RenderCallback_t callback)
{
   _renderCallbacks.push_back( std::move( callback ) );
}

void GlWindow::AddKeyCallback(KeyCallback_t callback)
{
   _keyCallbacks.push_back( std::move( callback ) );
}

void GlWindow::AddMouseMoveCallback(MouseMoveCallback_t callback)
{
   _mouseMoveCallbacks.push_back( std::move( callback ) );
}

void GlWindow::GlobalKeyCallback(GLFWwindow* glWnd, int key, int scancode, int action, int mods)
{
   auto glWindow = std::find_if( s_instances.begin(), s_instances.end(), [glWnd]( GlWindow* wnd ) { return wnd->_window == glWnd; } );
   if ( glWindow == s_instances.end() ) return;

   (*glWindow)->RaiseKeyCallbacks( key, scancode, action, mods );
}

void GlWindow::GlobalMouseMoveCallback(GLFWwindow* glWnd, double px, double py)
{
   auto glWindow = std::find_if( s_instances.begin(), s_instances.end(), [glWnd]( GlWindow* wnd ) { return wnd->_window == glWnd; } );
   if ( glWindow == s_instances.end() ) return;

   ( *glWindow )->RaiseMouseMoveCallbacks( px, py );
}

void GlWindow::RaiseKeyCallbacks(int a, int b, int c, int d)
{
   for ( auto& keyCallback : _keyCallbacks ) keyCallback( a, b, c, d );
}

void GlWindow::RaiseMouseMoveCallbacks(double px, double py)
{
   for ( auto& mouseCallback : _mouseMoveCallbacks ) mouseCallback( px, py );
}
