#pragma once

#include <functional>
#include <vector>

#include <chrono>
#include "GlTexture2d.h"

struct GLFWwindow;

class GlWindow
{
public:
   using RenderCallback_t = std::function<void( float )>;
   using KeyCallback_t = std::function<void( int, int, int, int )>;
   using MouseMoveCallback_t = std::function<void( double, double )>;

   GlWindow(int width, int height);
   ~GlWindow();

   void RenderLoop();
   void DoSingleFrame();

   auto& GetTexture() const { return _windowTexture; }
   auto GetWidth() const { return _width; }
   auto GetHeight() const { return _height; }

   void AddRenderCallback( RenderCallback_t callback );
   void AddKeyCallback( KeyCallback_t callback );
   void AddMouseMoveCallback( MouseMoveCallback_t callback );
private:
   static void GlobalKeyCallback( GLFWwindow* wnd, int key, int scancode, int action, int mods );
   static void GlobalMouseMoveCallback( GLFWwindow* wnd, double px, double py );

   void RaiseKeyCallbacks( int key, int scancode, int action, int mods );
   void RaiseMouseMoveCallbacks( double, double );

   int _width, _height;
   GLFWwindow* _window;
   GlTexture2d::Ptr _windowTexture;

   std::chrono::high_resolution_clock::time_point _lastTick;

   std::vector<RenderCallback_t> _renderCallbacks;
   std::vector<KeyCallback_t> _keyCallbacks;
   std::vector<MouseMoveCallback_t> _mouseMoveCallbacks;

   static std::vector<GlWindow*> s_instances;
};