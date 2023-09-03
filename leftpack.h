#pragma once

#include <cstdint>
#include <immintrin.h>

using U8 = uint8_t;
using U16 = uint16_t;
using U32 = uint32_t;
using U64 = uint64_t;

using I8 = int8_t;
using I16 = int16_t;
using I32 = int32_t;
using I64 = int64_t;

// Leftpack 32 8-bit elements from 'p' into 'pOut', keeping only elements whose corresponding bit is set in 32-bit
// bitmask 'm'. Returns number of elements output. May unconditionally write up to 32 elements to output, regardless of
// how many are actually output.
//
// DO NOT USE THIS VERSION on AMD Zen1/Zen2 machines; it's wicked slow there due to pdep/pext. This is fixed in Zen3 and
// after, so this version is preferred for any arch other than Zen1/Zen2 specifically.
static inline U64 Leftpack8(U8* __restrict pOut, const U8* __restrict p, U64 m)
{
    alignas(64) static constexpr U64 k1[] = { 0x0F0F0F0F0F0F0F0FULL, 0x0F0F0F0F0F0F0F0FULL, 0x0F0F0F0F0F0F0F0FULL, 0x0F0F0F0F0F0F0F0FULL };
    const __m256i y1 = _mm256_setr_m128i(
        _mm_cvtsi64_si128(I64(_pext_u64(0xFEDCBA9876543210ULL, 0xFULL * _pdep_u64(m & 0xFFFFU, 0x1111111111111111ULL)))),
        _mm_cvtsi64_si128(I64(_pext_u64(0xFEDCBA9876543210ULL, 0xFULL * _pdep_u64(m >> 16, 0x1111111111111111ULL))))
    );
    const __m256i y2 = _mm256_unpacklo_epi8(y1, _mm256_srli_epi16(y1, 4));
    const __m256i y3 = _mm256_and_si256(y2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k1)));
    const __m256i y4 = _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)), y3);
    _mm256_storeu2_m128i(reinterpret_cast<__m128i*>(pOut + U32(_mm_popcnt_u32(U32(m & 0xFFFFU)))), reinterpret_cast<__m128i*>(pOut), y4);
    return U32(_mm_popcnt_u32(U32(m)));
}

// Leftpack 32 8-bit elements from 'p' into 'pOut', keeping only elements whose corresponding bit is set in 32-bit
// bitmask 'm'. Returns number of elements output. May unconditionally write up to 32 elements to output, regardless of
// how many are actually output.
//
// This version is slower on any arch other than AMD Zen1/Zen2; prefer the other version, Leftpack8, unless you are
// targeting Zen1/Zen2.
static inline U64 Leftpack8_Zen2(U8* __restrict pOut, const U8* __restrict p, U64 m)
{
    alignas(64) static constexpr U32 k1[] = {
        0x00000000, 0x00000000, 0x00000001, 0x00000010, 0x00000002, 0x00000020, 0x00000021, 0x00000210, 0x00000003, 0x00000030, 0x00000031, 0x00000310, 0x00000032, 0x00000320, 0x00000321, 0x00003210,
        0x00000004, 0x00000040, 0x00000041, 0x00000410, 0x00000042, 0x00000420, 0x00000421, 0x00004210, 0x00000043, 0x00000430, 0x00000431, 0x00004310, 0x00000432, 0x00004320, 0x00004321, 0x00043210,
        0x00000005, 0x00000050, 0x00000051, 0x00000510, 0x00000052, 0x00000520, 0x00000521, 0x00005210, 0x00000053, 0x00000530, 0x00000531, 0x00005310, 0x00000532, 0x00005320, 0x00005321, 0x00053210,
        0x00000054, 0x00000540, 0x00000541, 0x00005410, 0x00000542, 0x00005420, 0x00005421, 0x00054210, 0x00000543, 0x00005430, 0x00005431, 0x00054310, 0x00005432, 0x00054320, 0x00054321, 0x00543210,
        0x00000006, 0x00000060, 0x00000061, 0x00000610, 0x00000062, 0x00000620, 0x00000621, 0x00006210, 0x00000063, 0x00000630, 0x00000631, 0x00006310, 0x00000632, 0x00006320, 0x00006321, 0x00063210,
        0x00000064, 0x00000640, 0x00000641, 0x00006410, 0x00000642, 0x00006420, 0x00006421, 0x00064210, 0x00000643, 0x00006430, 0x00006431, 0x00064310, 0x00006432, 0x00064320, 0x00064321, 0x00643210,
        0x00000065, 0x00000650, 0x00000651, 0x00006510, 0x00000652, 0x00006520, 0x00006521, 0x00065210, 0x00000653, 0x00006530, 0x00006531, 0x00065310, 0x00006532, 0x00065320, 0x00065321, 0x00653210,
        0x00000654, 0x00006540, 0x00006541, 0x00065410, 0x00006542, 0x00065420, 0x00065421, 0x00654210, 0x00006543, 0x00065430, 0x00065431, 0x00654310, 0x00065432, 0x00654320, 0x00654321, 0x06543210,
        0x00000007, 0x00000070, 0x00000071, 0x00000710, 0x00000072, 0x00000720, 0x00000721, 0x00007210, 0x00000073, 0x00000730, 0x00000731, 0x00007310, 0x00000732, 0x00007320, 0x00007321, 0x00073210,
        0x00000074, 0x00000740, 0x00000741, 0x00007410, 0x00000742, 0x00007420, 0x00007421, 0x00074210, 0x00000743, 0x00007430, 0x00007431, 0x00074310, 0x00007432, 0x00074320, 0x00074321, 0x00743210,
        0x00000075, 0x00000750, 0x00000751, 0x00007510, 0x00000752, 0x00007520, 0x00007521, 0x00075210, 0x00000753, 0x00007530, 0x00007531, 0x00075310, 0x00007532, 0x00075320, 0x00075321, 0x00753210,
        0x00000754, 0x00007540, 0x00007541, 0x00075410, 0x00007542, 0x00075420, 0x00075421, 0x00754210, 0x00007543, 0x00075430, 0x00075431, 0x00754310, 0x00075432, 0x00754320, 0x00754321, 0x07543210,
        0x00000076, 0x00000760, 0x00000761, 0x00007610, 0x00000762, 0x00007620, 0x00007621, 0x00076210, 0x00000763, 0x00007630, 0x00007631, 0x00076310, 0x00007632, 0x00076320, 0x00076321, 0x00763210,
        0x00000764, 0x00007640, 0x00007641, 0x00076410, 0x00007642, 0x00076420, 0x00076421, 0x00764210, 0x00007643, 0x00076430, 0x00076431, 0x00764310, 0x00076432, 0x00764320, 0x00764321, 0x07643210,
        0x00000765, 0x00007650, 0x00007651, 0x00076510, 0x00007652, 0x00076520, 0x00076521, 0x00765210, 0x00007653, 0x00076530, 0x00076531, 0x00765310, 0x00076532, 0x00765320, 0x00765321, 0x07653210,
        0x00007654, 0x00076540, 0x00076541, 0x00765410, 0x00076542, 0x00765420, 0x00765421, 0x07654210, 0x00076543, 0x00765430, 0x00765431, 0x07654310, 0x00765432, 0x07654320, 0x07654321, 0x76543210,
    };
    alignas(64) static constexpr U64 k2[] = { 0x0F0F0F0F0F0F0F0FULL, 0x0F0F0F0F0F0F0F0FULL, 0x0F0F0F0F0F0F0F0FULL, 0x0F0F0F0F0F0F0F0FULL };
#if defined(__GNUC__) || defined(__clang__)
    U64 r0, r1, r2;
    __m256i y0, y1;
    asm(R"(
movl %k1, %k0
movzx %h0, %k0
movl (%7, %0, 4), %k2
orl $0x88888888, %k2
movzx %b1, %k0
popcntl %k0, %k3
shll $2, %k3
shlxq %3, %2, %2
movl (%7, %0, 4), %k0
orq %2, %0
vmovq %0, %x4
movl %k1, %k0
shrl $24, %k0
movl (%7, %0, 4), %k2
orl $0x88888888, %k2
movl %k1, %k0
shrl $16, %k0
movzx %b0, %k0
popcntl %k0, %k3
shll $2, %k3
shlxq %3, %2, %2
movl (%7, %0, 4), %k0
orq %2, %0
vmovq %0, %x5
vinserti128 $1, %x5, %4, %4
vpsrlw $4, %4, %5
vpunpcklbw %5, %4, %4
vpand %8, %4, %4
vmovdqu (%10), %5
vpshufb %4, %5, %4
popcntl %k1, %k0
movzx %w1, %k1
popcntl %k1, %k1
vmovdqu %x4, (%9)
vextracti128 $1, %4, (%9, %1, 1)
)":"=&Q"(r0), "+q"(m), "=&r"(r1), "=&r"(r2), "=&x"(y0), "=&x"(y1), "=m"(*reinterpret_cast<U8(*)[32]>(pOut)) : "r"(k1), "m"(k2), "r"(pOut), "r"(p));
    return r0;
#else
    const U64 q1 = k1[m & 0xFFU];
    const U64 q2 = k1[(m >> 8) & 0xFFU] | 0x88888888U;
    const U64 h1 = (q2 << (U32(_mm_popcnt_u32(U32(m & 0xFFU))) << 2)) | q1;
    const __m128i x1 = _mm_cvtsi64_si128(I64(h1));
    const U64 q3 = k1[(m >> 16) & 0xFFU];
    const U64 q4 = k1[m >> 24] | 0x88888888U;
    const U64 h2 = (q4 << (U32(_mm_popcnt_u32(U32((m >> 16) & 0xFFU))) << 2)) | q3;
    const __m128i x2 = _mm_cvtsi64_si128(I64(h2));
    const __m256i y1 = _mm256_setr_m128i(x1, x2);
    const __m256i y2 = _mm256_and_si256(_mm256_unpacklo_epi8(y1, _mm256_srli_epi16(y1, 4)), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k2)));
    const __m256i y3 = _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)), y2);
    _mm256_storeu2_m128i(reinterpret_cast<__m128i*>(pOut + U32(_mm_popcnt_u32(U32(m & 0xFFFFU)))), reinterpret_cast<__m128i*>(pOut), y3);
    return U32(_mm_popcnt_u32(U32(m)));
#endif
}

// Leftpack 16 16-bit elements from 'p' into 'pOut', keeping only elements whose corresponding bit is set in 16-bit
// bitmask 'm'. Returns number of elements output. May unconditionally write up to 16 elements to output, regardless of
// how many are actually output.
static inline U64 Leftpack16(U16* __restrict pOut, const U16* __restrict p, U64 m)
{
    alignas(64) static constexpr U32 k1[] = {
        0x00000000, 0x00000001, 0x00000003, 0x00000301, 0x00000005, 0x00000501, 0x00000503, 0x00050301, 0x00000007, 0x00000701, 0x00000703, 0x00070301, 0x00000705, 0x00070501, 0x00070503, 0x07050301,
        0x00000009, 0x00000901, 0x00000903, 0x00090301, 0x00000905, 0x00090501, 0x00090503, 0x09050301, 0x00000907, 0x00090701, 0x00090703, 0x09070301, 0x00090705, 0x09070501, 0x09070503, 0x07050391,
        0x0000000b, 0x00000b01, 0x00000b03, 0x000b0301, 0x00000b05, 0x000b0501, 0x000b0503, 0x0b050301, 0x00000b07, 0x000b0701, 0x000b0703, 0x0b070301, 0x000b0705, 0x0b070501, 0x0b070503, 0x070503b1,
        0x00000b09, 0x000b0901, 0x000b0903, 0x0b090301, 0x000b0905, 0x0b090501, 0x0b090503, 0x090503b1, 0x000b0907, 0x0b090701, 0x0b090703, 0x090703b1, 0x0b090705, 0x090705b1, 0x090705b3, 0x0705b391,
        0x0000000d, 0x00000d01, 0x00000d03, 0x000d0301, 0x00000d05, 0x000d0501, 0x000d0503, 0x0d050301, 0x00000d07, 0x000d0701, 0x000d0703, 0x0d070301, 0x000d0705, 0x0d070501, 0x0d070503, 0x070503d1,
        0x00000d09, 0x000d0901, 0x000d0903, 0x0d090301, 0x000d0905, 0x0d090501, 0x0d090503, 0x090503d1, 0x000d0907, 0x0d090701, 0x0d090703, 0x090703d1, 0x0d090705, 0x090705d1, 0x090705d3, 0x0705d391,
        0x00000d0b, 0x000d0b01, 0x000d0b03, 0x0d0b0301, 0x000d0b05, 0x0d0b0501, 0x0d0b0503, 0x0b0503d1, 0x000d0b07, 0x0d0b0701, 0x0d0b0703, 0x0b0703d1, 0x0d0b0705, 0x0b0705d1, 0x0b0705d3, 0x0705d3b1,
        0x000d0b09, 0x0d0b0901, 0x0d0b0903, 0x0b0903d1, 0x0d0b0905, 0x0b0905d1, 0x0b0905d3, 0x0905d3b1, 0x0d0b0907, 0x0b0907d1, 0x0b0907d3, 0x0907d3b1, 0x0b0907d5, 0x0907d5b1, 0x0907d5b3, 0x07d5b391,
        0x0000000f, 0x00000f01, 0x00000f03, 0x000f0301, 0x00000f05, 0x000f0501, 0x000f0503, 0x0f050301, 0x00000f07, 0x000f0701, 0x000f0703, 0x0f070301, 0x000f0705, 0x0f070501, 0x0f070503, 0x070503f1,
        0x00000f09, 0x000f0901, 0x000f0903, 0x0f090301, 0x000f0905, 0x0f090501, 0x0f090503, 0x090503f1, 0x000f0907, 0x0f090701, 0x0f090703, 0x090703f1, 0x0f090705, 0x090705f1, 0x090705f3, 0x0705f391,
        0x00000f0b, 0x000f0b01, 0x000f0b03, 0x0f0b0301, 0x000f0b05, 0x0f0b0501, 0x0f0b0503, 0x0b0503f1, 0x000f0b07, 0x0f0b0701, 0x0f0b0703, 0x0b0703f1, 0x0f0b0705, 0x0b0705f1, 0x0b0705f3, 0x0705f3b1,
        0x000f0b09, 0x0f0b0901, 0x0f0b0903, 0x0b0903f1, 0x0f0b0905, 0x0b0905f1, 0x0b0905f3, 0x0905f3b1, 0x0f0b0907, 0x0b0907f1, 0x0b0907f3, 0x0907f3b1, 0x0b0907f5, 0x0907f5b1, 0x0907f5b3, 0x07f5b391,
        0x00000f0d, 0x000f0d01, 0x000f0d03, 0x0f0d0301, 0x000f0d05, 0x0f0d0501, 0x0f0d0503, 0x0d0503f1, 0x000f0d07, 0x0f0d0701, 0x0f0d0703, 0x0d0703f1, 0x0f0d0705, 0x0d0705f1, 0x0d0705f3, 0x0705f3d1,
        0x000f0d09, 0x0f0d0901, 0x0f0d0903, 0x0d0903f1, 0x0f0d0905, 0x0d0905f1, 0x0d0905f3, 0x0905f3d1, 0x0f0d0907, 0x0d0907f1, 0x0d0907f3, 0x0907f3d1, 0x0d0907f5, 0x0907f5d1, 0x0907f5d3, 0x07f5d391,
        0x000f0d0b, 0x0f0d0b01, 0x0f0d0b03, 0x0d0b03f1, 0x0f0d0b05, 0x0d0b05f1, 0x0d0b05f3, 0x0b05f3d1, 0x0f0d0b07, 0x0d0b07f1, 0x0d0b07f3, 0x0b07f3d1, 0x0d0b07f5, 0x0b07f5d1, 0x0b07f5d3, 0x07f5d3b1,
        0x0f0d0b09, 0x0d0b09f1, 0x0d0b09f3, 0x0b09f3d1, 0x0d0b09f5, 0x0b09f5d1, 0x0b09f5d3, 0x09f5d3b1, 0x0d0b09f7, 0x0b09f7d1, 0x0b09f7d3, 0x09f7d3b1, 0x0b09f7d5, 0x09f7d5b1, 0x09f7d5b3, 0xf7d5b391,
    };
    alignas(64) static constexpr U64 k2[] = { 0, 0x400000004, 0, 0x400000004 };
    alignas(64) static constexpr U64 k3[] = { 0x0F0E0F0E0F0E0F0EULL, 0x0F0E0F0E0F0E0F0EULL, 0x0F0E0F0E0F0E0F0EULL, 0x0F0E0F0E0F0E0F0EULL };
    const U32 n1 = U32(_mm_popcnt_u32(U32(m & 0xFFU)));
    const U32 n2 = U32(_mm_popcnt_u32(U32(m)));
    const __m256i y1 = _mm256_setr_m128i(_mm_set1_epi32(I32(k1[m & 0xFFU])), _mm_set1_epi32(I32(k1[m >> 8])));
    const __m256i y2 = _mm256_unpacklo_epi8(y1, y1);
    const __m256i y3 = _mm256_srlv_epi32(y2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k2)));
    const __m256i y4 = _mm256_and_si256(y3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k3)));
    const __m256i y5 = _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)), y4);
    _mm256_storeu2_m128i(reinterpret_cast<__m128i*>(pOut + n1), reinterpret_cast<__m128i*>(pOut), y5);
    return n2;
}

// Leftpack 8 32-bit elements from 'p' into 'pOut', keeping only elements whose corresponding bit is set in 8-bit
// bitmask 'm'. Returns number of elements output. May unconditionally write up to 8 elements to output, regardless of
// how many are actually output.
//
// DO NOT USE THIS VERSION on AMD Zen1/Zen2 machines; it's wicked slow there due to pdep/pext. This is fixed in Zen3 and
// after, so this version is preferred for any arch other than Zen1/Zen2 specifically.
static inline U64 Leftpack32(U32* __restrict pOut, const U32* __restrict p, U64 m)
{
    const __m256i y1 = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(I64(_pext_u64(0x0706050403020100ULL, 0xFFULL * _pdep_u64(m, 0x0101010101010101ULL)))));
    const __m256i y2 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)), y1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(pOut), y2);
    return U32(_mm_popcnt_u32(U32(m)));
}

// Leftpack 8 32-bit elements from 'p' into 'pOut', keeping only elements whose corresponding bit is set in 8-bit
// bitmask 'm'. Returns number of elements output. May unconditionally write up to 8 elements to output, regardless of
// how many are actually output.
//
// This version is slower on any arch other than AMD Zen1/Zen2; prefer the other version, Leftpack8, unless you are
// targeting Zen1/Zen2.
static inline U64 Leftpack32_Zen2(U32* __restrict pOut, const U32* __restrict p, U64 m)
{
    alignas(64) static constexpr U32 k1[] = {
        0x00000000, 0x00000000, 0x00000001, 0x00000010, 0x00000002, 0x00000020, 0x00000021, 0x00000210, 0x00000003, 0x00000030, 0x00000031, 0x00000310, 0x00000032, 0x00000320, 0x00000321, 0x00003210,
        0x00000004, 0x00000040, 0x00000041, 0x00000410, 0x00000042, 0x00000420, 0x00000421, 0x00004210, 0x00000043, 0x00000430, 0x00000431, 0x00004310, 0x00000432, 0x00004320, 0x00004321, 0x00043210,
        0x00000005, 0x00000050, 0x00000051, 0x00000510, 0x00000052, 0x00000520, 0x00000521, 0x00005210, 0x00000053, 0x00000530, 0x00000531, 0x00005310, 0x00000532, 0x00005320, 0x00005321, 0x00053210,
        0x00000054, 0x00000540, 0x00000541, 0x00005410, 0x00000542, 0x00005420, 0x00005421, 0x00054210, 0x00000543, 0x00005430, 0x00005431, 0x00054310, 0x00005432, 0x00054320, 0x00054321, 0x00543210,
        0x00000006, 0x00000060, 0x00000061, 0x00000610, 0x00000062, 0x00000620, 0x00000621, 0x00006210, 0x00000063, 0x00000630, 0x00000631, 0x00006310, 0x00000632, 0x00006320, 0x00006321, 0x00063210,
        0x00000064, 0x00000640, 0x00000641, 0x00006410, 0x00000642, 0x00006420, 0x00006421, 0x00064210, 0x00000643, 0x00006430, 0x00006431, 0x00064310, 0x00006432, 0x00064320, 0x00064321, 0x00643210,
        0x00000065, 0x00000650, 0x00000651, 0x00006510, 0x00000652, 0x00006520, 0x00006521, 0x00065210, 0x00000653, 0x00006530, 0x00006531, 0x00065310, 0x00006532, 0x00065320, 0x00065321, 0x00653210,
        0x00000654, 0x00006540, 0x00006541, 0x00065410, 0x00006542, 0x00065420, 0x00065421, 0x00654210, 0x00006543, 0x00065430, 0x00065431, 0x00654310, 0x00065432, 0x00654320, 0x00654321, 0x06543210,
        0x00000007, 0x00000070, 0x00000071, 0x00000710, 0x00000072, 0x00000720, 0x00000721, 0x00007210, 0x00000073, 0x00000730, 0x00000731, 0x00007310, 0x00000732, 0x00007320, 0x00007321, 0x00073210,
        0x00000074, 0x00000740, 0x00000741, 0x00007410, 0x00000742, 0x00007420, 0x00007421, 0x00074210, 0x00000743, 0x00007430, 0x00007431, 0x00074310, 0x00007432, 0x00074320, 0x00074321, 0x00743210,
        0x00000075, 0x00000750, 0x00000751, 0x00007510, 0x00000752, 0x00007520, 0x00007521, 0x00075210, 0x00000753, 0x00007530, 0x00007531, 0x00075310, 0x00007532, 0x00075320, 0x00075321, 0x00753210,
        0x00000754, 0x00007540, 0x00007541, 0x00075410, 0x00007542, 0x00075420, 0x00075421, 0x00754210, 0x00007543, 0x00075430, 0x00075431, 0x00754310, 0x00075432, 0x00754320, 0x00754321, 0x07543210,
        0x00000076, 0x00000760, 0x00000761, 0x00007610, 0x00000762, 0x00007620, 0x00007621, 0x00076210, 0x00000763, 0x00007630, 0x00007631, 0x00076310, 0x00007632, 0x00076320, 0x00076321, 0x00763210,
        0x00000764, 0x00007640, 0x00007641, 0x00076410, 0x00007642, 0x00076420, 0x00076421, 0x00764210, 0x00007643, 0x00076430, 0x00076431, 0x00764310, 0x00076432, 0x00764320, 0x00764321, 0x07643210,
        0x00000765, 0x00007650, 0x00007651, 0x00076510, 0x00007652, 0x00076520, 0x00076521, 0x00765210, 0x00007653, 0x00076530, 0x00076531, 0x00765310, 0x00076532, 0x00765320, 0x00765321, 0x07653210,
        0x00007654, 0x00076540, 0x00076541, 0x00765410, 0x00076542, 0x00765420, 0x00765421, 0x07654210, 0x00076543, 0x00765430, 0x00765431, 0x07654310, 0x00765432, 0x07654320, 0x07654321, 0x76543210,
    };
    alignas(64) static constexpr U32 k2[] = { 0, 4, 8, 12, 16, 20, 24, 28 };
    const __m256i y1 = _mm256_srlv_epi32(_mm256_set1_epi32(I32(k1[m])), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k2)));
    const __m256i y2 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)), y1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(pOut), y2);
    return U32(_mm_popcnt_u32(U32(m)));
}

// Leftpack 4 64-bit elements from 'p' into 'pOut', keeping only elements whose corresponding bit is set in 4-bit
// bitmask 'm'. Returns number of elements output. May unconditionally write up to 4 elements to output, regardless of
// how many are actually output.
static inline U64 Leftpack64(U64* __restrict pOut, const U64* __restrict p, U64 m)
{
    alignas(64) static constexpr U32 k1[] = {
        0x00000000, 0x00000010, 0x00000032, 0x00003210, 0x00000054, 0x00005410, 0x00005432, 0x00543210, 0x00000076, 0x00007610, 0x00007632, 0x00763210, 0x00007654, 0x00765410, 0x00765432, 0x76543210,
    };
    alignas(64) static constexpr U32 k2[] = { 0, 4, 8, 12, 16, 20, 24, 28 };
    const __m256i y1 = _mm256_srlv_epi32(_mm256_set1_epi32(I32(k1[m])), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k2)));
    const __m256i y2 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)), y1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(pOut), y2);
    return U32(_mm_popcnt_u32(U32(m)));
}
