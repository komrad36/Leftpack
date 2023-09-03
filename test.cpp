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

template <class T, class U>
static void Test(U f)
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

int main()
{
    Test<U8>(Leftpack8);
    Test<U8>(Leftpack8_Zen2);

    Test<U16>(Leftpack16);

    Test<U32>(Leftpack32);
    Test<U32>(Leftpack32_Zen2);

    Test<U64>(Leftpack64);
}
