#pragma once
#include <cstdint>

class GlObject
{
public:
   virtual ~GlObject() {}

   auto GetGlHandle() const { return _glHandle; }
protected:
   void Create();   
private:
   virtual uint32_t CreateImpl() = 0;
   uint32_t  _glHandle;
};
