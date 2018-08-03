#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __F16C__
#include <immintrin.h>
#endif
#ifdef __ALTIVEC__
#include <altivec.h>
#endif

typedef unsigned short HALF;

static HALF float2half(float flt)
{
  // software implementation rounds toward nearest even
  unsigned const& s = *reinterpret_cast<unsigned const*>(&flt);
  int sign = ((s >> 31) & 0x1);
  int exp = ((s >> 23) & 0xff) - 127;
  int mantissa = (s & 0x7fffff);
  unsigned short u = 0;

  if ((s & 0x7fffffff) == 0) {
    // sign-preserving zero
    return (sign << 15);
  }

  if (exp > 15) {
    if (exp == 128 && mantissa) {
      // not a number
      u = 0x7fff;
    } else {
      // overflow to infinity
      u = (sign << 15) | 0x7c00;
    }
    return u;
  }

  int sticky_bit = 0;

  if (exp >= -14) {
    // normal fp32 to normal fp16
    exp += 15;
    u = ((exp & 0x1f) << 10);
    u |= (mantissa >> 13);
  } else {
    // normal single-precision to subnormal half_t-precision representation
    int rshift = (-14 - exp);
    if (rshift < 32) {
      mantissa |= (1 << 23);

      sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0);

      mantissa = (mantissa >> rshift);
      u = ((mantissa >> 13) & 0x3ff);
    } else {
      mantissa = 0;
      u = 0;
    }
  }

  // round to nearest even
  int round_bit = ((mantissa >> 12) & 1);
  sticky_bit |= ((mantissa & ((1 << 12) - 1)) != 0);

  if ((round_bit && sticky_bit) || (round_bit && (u & 1))) {
    u += 1;
  }

  u |= (sign << 15);

  return u;
}

static float half2float(HALF h)
{
  int sign = ((h >> 15) & 1);
  int exp = ((h >> 10) & 0x1f);
  int mantissa = (h & 0x3ff);
  unsigned f = 0;

  if (exp > 0 && exp < 31) {
    // normal
    exp += 112;
    f = (sign << 31) | (exp << 23) | (mantissa << 13);
  } else if (exp == 0) {
    if (mantissa) {
      // subnormal
      exp += 113;
      while ((mantissa & (1 << 10)) == 0) {
        mantissa <<= 1;
        exp--;
      }
      mantissa &= 0x3ff;
      f = (sign << 31) | (exp << 23) | (mantissa << 13);
    } else {
      // sign-preserving zero
      f = (sign << 31);
    }
  } else if (exp == 31) {
    if (mantissa) {
      f = 0x7fffffff;  // not a number
    } else {
      f = (0xff << 23) | (sign << 31);  //  inf
    }
  }
  return *reinterpret_cast<float const*>(&f);
}  

void cpu_add(HALF *in, HALF *inout, size_t count)
{
#ifdef __F16C__
  for(size_t i = 0; i < count; i += 4) {
    __m128i h1 = _mm_cvtsi64_si128(*(const long long *)(in + i));
    __m128 f1 = _mm_cvtph_ps(h1);
    __m128i h2 = _mm_cvtsi64_si128(*(const long long *)(inout + i));
    __m128 f2 = _mm_cvtph_ps(h2);
    __m128 f3 = _mm_add_ps(f1, f2);
    //f3 = _mm_set_ps(i+3,i+2,i+1,i);
    __m128i h3 = _mm_cvtps_ph(f3, 0);
    *(long long *)(inout + i) = _mm_cvtsi128_si64(h3);
  }
#else
#ifdef __ALTIVEC__
  vector unsigned char sel_in_high = { 0, 1, 0, 0, 2, 3, 0, 0, 4, 5, 0, 0, 6, 7, 0, 0 };
  vector unsigned char sel_in_low = { 8, 9, 0, 0, 10, 11, 0, 0, 12, 13, 0, 0, 14, 15, 0, 0 };
  vector unsigned char sel = { 0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29 };
  for(size_t i = 0; i < count; i += 8) {
    vector unsigned short ha = vec_ld(0, in + i);
    vector unsigned short hb = vec_ld(0, inout + i);
#if 1
    vector unsigned short hal = vec_perm(ha, ha, sel_in_low);
    vector unsigned short hah = vec_perm(ha, ha, sel_in_high);
    vector unsigned short hbl = vec_perm(hb, hb, sel_in_low);
    vector unsigned short hbh = vec_perm(hb, hb, sel_in_high);
    vector float fal, fah, fbl, fbh;
    asm("xvcvhpsp %x0, %x1\n" : "=v"(fal) : "v"(hal));
    asm("xvcvhpsp %x0, %x1\n" : "=v"(fah) : "v"(hah));
    asm("xvcvhpsp %x0, %x1\n" : "=v"(fbl) : "v"(hbl));
    asm("xvcvhpsp %x0, %x1\n" : "=v"(fbh) : "v"(hbh));
#else
    vector float fal = vec_extract_fp_from_shortl(ha);
    vector float fah = vec_extract_fp_from_shorth(ha);
    vector float fbl = vec_extract_fp_from_shortl(hb);
    vector float fbh = vec_extract_fp_from_shorth(hb);
#endif
    vector float fcl = vec_add(fal, fbl);
    vector float fch = vec_add(fah, fbh);
    vector unsigned short hcl, hch, hc;
    asm("xvcvsphp %x0, %x1\n"
        : "=v"(hcl)
        : "v"(fcl));
    asm("xvcvsphp %x0, %x1\n"
        : "=v"(hch)
        : "v"(fch));
    hc = vec_perm(hch, hcl, sel);
    vec_st(hc, 0, inout + i);
  }
#else
  for(size_t i = 0; i < count; i++)
    //inout[i] = float2half(i);
    inout[i] = float2half(half2float(in[i]) +
  			  half2float(inout[i]));
#endif
#endif
}
