//
// (C) Jan de Vaan 2007-2009, all rights reserved. See the accompanying "License.txt" for licensed use.
//


#ifndef CHARLS_UTIL
#define CHARLS_UTIL

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "config.h"
#include "interface.h"


#ifndef MAX
#define MAX(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a,b)            (((a) < (b)) ? (a) : (b))
#endif


const LONG BASIC_RESET  = 64;

inline LONG log_2(LONG n)
{
  LONG x = 0;
  while (n > (LONG(1) << x))
  {
    ++x;
  }
  return x;

}

struct Size
{
  Size(LONG width, LONG height) :
    cx(width),
    cy(height)
  {}
  LONG cx;
  LONG cy;
};



inline LONG Sign(LONG n)
  { return (n >> (LONG_BITCOUNT-1)) | 1;}

inline LONG BitWiseSign(LONG i)
  { return i >> (LONG_BITCOUNT-1); }


template<class SAMPLE>
struct Triplet
{
  Triplet() :
    v1(0),
    v2(0),
    v3(0)
  {}

  Triplet(LONG x1, LONG x2, LONG x3) :
    v1((SAMPLE)x1),
    v2((SAMPLE)x2),
    v3((SAMPLE)x3)
  {}

    union
    {
      SAMPLE v1;
      SAMPLE R;
    };
    union
    {
      SAMPLE v2;
      SAMPLE G;
    };
    union
    {
      SAMPLE v3;
      SAMPLE B;
    };
};

inline bool operator==(const Triplet<BYTE>& lhs, const Triplet<BYTE>& rhs)
  { return lhs.v1 == rhs.v1 && lhs.v2 == rhs.v2 && lhs.v3 == rhs.v3; }

inline bool  operator!=(const Triplet<BYTE>& lhs, const Triplet<BYTE>& rhs)
  { return !(lhs == rhs); }


template<class sample>
struct Quad : public Triplet<sample>
{
  Quad() :
    v4(0)
    {}

  Quad(Triplet<sample> triplet, LONG alpha) : Triplet<sample>(triplet), A((sample)alpha)
    {}

  union
  {
    sample v4;
    sample A;
  };
};



template <int size>
struct FromBigEndian
{
};

template <>
struct FromBigEndian<4>
{
  inlinehint static unsigned int Read(BYTE* pbyte)
  {
    return  (pbyte[0] << 24) + (pbyte[1] << 16) + (pbyte[2] << 8) + (pbyte[3] << 0);
  }
};



template <>
struct FromBigEndian<8>
{
  //typedef unsigned long long UINT64;
  typedef uint64_t UINT64;

  inlinehint static UINT64 Read(BYTE* pbyte)
  {
    return  (UINT64(pbyte[0]) << 56) + (UINT64(pbyte[1]) << 48) + (UINT64(pbyte[2]) << 40) + (UINT64(pbyte[3]) << 32) +
          (UINT64(pbyte[4]) << 24) + (UINT64(pbyte[5]) << 16) + (UINT64(pbyte[6]) <<  8) + (UINT64(pbyte[7]) << 0);
  }
};


class JlsException
{
public:
  JlsException(JLS_ERROR error) : _error(error)
    { }

  JLS_ERROR _error;
};


#endif
