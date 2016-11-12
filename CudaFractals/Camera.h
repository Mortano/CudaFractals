#pragma once

#include <glm/glm.hpp>

class Camera
{
public:
   Camera();

   const auto& GetTransform() const { return _transform; }
   auto GetFov() const { return _fov; }
   auto GetFocalLength() const { return _focalLength; }

   glm::vec3 GetForward() const;
   glm::vec3 GetUp() const;
   glm::vec3 GetRight() const;
   glm::vec3 GetPosition() const;

   void LookAt( const glm::vec3& at, const glm::vec3& eye, const glm::vec3& up );

   void SetTransform( const glm::mat4& transform );
   void SetFov( float fov );
   void SetFocalLength( float focalLength );

   void Move( const glm::vec3& dir );
private:
   glm::mat4 _transform;
   float _fov, _focalLength;
};