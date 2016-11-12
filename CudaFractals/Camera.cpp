#include "Camera.h"
#include "GlmUtil.h"

Camera::Camera() :
   _fov( 3.14159f / 2 ),
   _focalLength( 1.f )
{
   _transform = glm::mat4( 1, 0, 0, 0,
                           0, 1, 0, 0,
                           0, 0, 1, 0,
                           0, 0, -2, 1 );
}

glm::vec3 Camera::GetForward() const
{
   return XYZ( _transform[2] );
}

glm::vec3 Camera::GetUp() const
{
   return XYZ( _transform[1] );
}

glm::vec3 Camera::GetRight() const
{
   return XYZ( _transform[0] );
}

glm::vec3 Camera::GetPosition() const
{
   return XYZ( _transform[3] );
}

void Camera::LookAt(const glm::vec3& at, const glm::vec3& eye, const glm::vec3& up)
{
   auto fwd = glm::normalize( at - eye );
   auto right = glm::cross( fwd, up );
   auto correctUp = glm::cross( fwd, right );

   auto tx = -glm::dot( right, eye );
   auto ty = -glm::dot( up, eye );
   auto tz = -glm::dot( fwd, eye );

   _transform[0] = glm::vec4(right.x, right.y, right.z, 0);
   _transform[1] = glm::vec4( up.x, up.y, up.z, 0 );
   _transform[2] = glm::vec4( fwd.x, fwd.y, fwd.z, 0 );
   _transform[3] = glm::vec4( eye.x, eye.y, eye.z, 1 );
}

void Camera::SetTransform( const glm::mat4& transform )
{
   _transform = transform;
}


void Camera::SetFov( float fov )
{
   _fov = fov;
}

void Camera::SetFocalLength( float focalLength )
{
   _focalLength = focalLength;
}

void Camera::Move( const glm::vec3& dir )
{
   _transform[3] += glm::vec4( dir.x, dir.y, dir.z, 0 );
}
