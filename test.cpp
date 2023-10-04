#include "leftpack.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

template <class T>
static U64 Ref(T* pOut, const T* p, U64 m)
{
    constexpr U32 k = 32 / sizeof(T);
    U64 c = 0;
    for (U64 i = 0; i < k; ++i)
    {
        pOut[c] = p[i];
        c += (m >> i) & 1;
    }
    return c;
}

template <class T>
static void TestBitmask(U64 f(void*, const void*, U64))
{
    constexpr U32 k = 32 / sizeof(T);
    for (U64 i = 0; i < (1ULL << k); ++i)
    {
        T in[k];
        for (U64 j = 0; j < k; ++j)
            in[j] = T(j + 1);
        T a[k];
        const U64 na = Ref<T>(a, in, i);
        T b[k];
        const U64 nb = f(b, in, i);
        if (na != nb || memcmp(a, b, sizeof(T) * na) != 0)
        {
            printf("FAIL: i == %llu, na == %llu, nb == %llu\n", static_cast<unsigned long long>(i), static_cast<unsigned long long>(na), static_cast<unsigned long long>(nb));
            exit(1);
        }
        if constexpr (k == 32)
        {
            if (i > 1ULL << 22)
            {
                i += 1008;
            }
        }
    }
}

template <class T, bool kSignBitOnly>
static void TestVecmask(U64 f(void*, const void*, __m256i))
{
    constexpr U32 k = 32 / sizeof(T);
    for (U64 i = 0; i < (1ULL << k); ++i)
    {
        T in[k];
        for (U64 j = 0; j < k; ++j)
            in[j] = T(j + 1);
        T a[k];
        const U64 na = Ref<T>(a, in, i);
        T b[k];
        __m256i m;
        if constexpr (k == 32)
        {
            m = _mm256_shuffle_epi8(_mm256_set1_epi32(I32(i)), _mm256_setr_epi64x(0, 0x0101010101010101, 0x0202020202020202, 0x0303030303030303));
            m = _mm256_andnot_si256(m, _mm256_broadcastsi128_si256(_mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128)));
            m = _mm256_cmpeq_epi8(m, _mm256_setzero_si256());
        }
        if constexpr (k == 16)
        {
            m = _mm256_set1_epi16(I16(i));
            m = _mm256_andnot_si256(m, _mm256_setr_epi16(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, -32768));
            m = _mm256_cmpeq_epi16(m, _mm256_setzero_si256());
        }
        if constexpr (k == 8)
        {
            m = _mm256_set1_epi8(I8(i));
            m = _mm256_andnot_si256(m, _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128));
            m = _mm256_cmpeq_epi32(m, _mm256_setzero_si256());
        }
        if constexpr (k == 4)
        {
            m = _mm256_set1_epi8(I8(i));
            m = _mm256_andnot_si256(m, _mm256_setr_epi64x(1, 2, 4, 8));
            m = _mm256_cmpeq_epi64(m, _mm256_setzero_si256());
        }
        if constexpr (kSignBitOnly)
        {
            m = _mm256_and_si256(m, _mm256_setr_epi64x(0x8181818181818181LL, 0x8282828282828282LL, 0x8383838383838383LL, 0x9494949494949494LL));
        }
        const U64 nb = f(b, in, m);
        if (na != nb || memcmp(a, b, sizeof(T) * na) != 0)
        {
            printf("FAIL: i == %llu, na == %llu, nb == %llu\n", static_cast<unsigned long long>(i), static_cast<unsigned long long>(na), static_cast<unsigned long long>(nb));
            exit(1);
        }
        if constexpr (k == 32)
        {
            if (i > 1ULL << 22)
            {
                i += 1008;
            }
        }
    }
}

int main()
{
    TestBitmask<U8>(Leftpack8);
    TestBitmask<U8>(Leftpack8_Zen2);
    TestBitmask<U16>(Leftpack16);
    TestBitmask<U32>(Leftpack32);
    TestBitmask<U32>(Leftpack32_Zen2);
    TestBitmask<U64>(Leftpack64);

    TestVecmask<U8, false>(Leftpack8);
    TestVecmask<U8, false>(Leftpack8_Zen2);
    TestVecmask<U16, false>(Leftpack16);
    TestVecmask<U32, false>(Leftpack32);
    TestVecmask<U64, false>(Leftpack64);

    TestVecmask<U16, true>(Leftpack16);
    TestVecmask<U32, true>(Leftpack32);
    TestVecmask<U64, true>(Leftpack64);
}
