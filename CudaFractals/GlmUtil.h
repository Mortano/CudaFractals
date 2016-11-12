#pragma once
#include "../packages/glm.0.9.7.1/build/native/include/glm/detail/type_vec4.hpp"
#include "../packages/glm.0.9.7.1/build/native/include/glm/detail/type_vec3.hpp"

inline glm::vec3 XYZ(const glm::vec4& v)
{
   return glm::vec3( v.x, v.y, v.z );
}
