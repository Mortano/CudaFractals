#pragma once

#include "../VectorTypesUtil.h"

#include <vector_functions.h>
#include <cmath>

inline __device__ float fract(float f)
{
   return f - static_cast<long>( f );
}

inline __device__ float2 rand2n(float2& seed)
{
   seed = seed + make_float2( -1, 1 );
   return make_float2( fract( sin( dot( seed, make_float2( 12.9898f, 78.233f ) ) ) * 43758.5453f ),
                       fract( cos( dot( seed, make_float2( 4.898f, 7.23f ) ) ) * 23421.631f ) );
}