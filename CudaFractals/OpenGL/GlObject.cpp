#include "GlObject.h"

void GlObject::Create()
{
   _glHandle = CreateImpl();
}
