# Leftpack
Extremely fast (up to 39x faster than naive) AVX2 leftpack/compress implementations (keep and contiguously pack a subset of elements)


Sometimes you have an array of elements and a corresponding bitmask/bitarray or corresponding elements of the same size indicating a subset of elements to keep; that is, to output contiguously while discarding the rest, often called a left-pack or compress operation.

The naive way to do this is:

```cpp
// given pointers p and pOut, bitmask m of elements to keep, and k elements to check
U64 n = 0;
for (U64 i = 0; i < k; ++i)
    if ((m >> i) & 1)
        pOut[n++] = p[i];
return n;
```

and a slightly better way to do it, avoiding massive branch mispredicts, is:

```cpp
// given pointers p and pOut, bitmask m of elements to keep, and k elements to check
U64 n = 0;
for (U64 i = 0; i < k; ++i)
{
    pOut[n] = p[i];
    n += (m >> i) & 1;
}
return n;
```

But we can still do much better. If you have AVX512F and, for some word sizes, AVX512_VBMI2, there are the VCOMPRESS instructions, but they're kinda slow; in particular, for the common case of having a bitmask/bitarray of which elements to keep, we can beat them. And lots of environments still don't have AVX512 or the required subextensions anyway, so it's proven useful to have these highly efficient AVX2 implementations.


## Approx. speedup vs. naive methods ##  

Wordsize        | Super naive         | Branchless naive    
--------------- |---------------------|---------------------
 8-bit elements | 39x                 | 5.3x
16-bit elements | 31x                 | 3.8x
32-bit elements | 14x                 | 2.2x
64-bit elements | 7.8x                | 1.6x


The 8-bit and 32-bit versions make use of the PDEP and PEXT instructions from the BMI2 instruction set, which are SLOW AS HELL on AMD Zen1 and Zen2 (subsequent Zens are fine), so alternative implementations (that are only slightly slower) are provided if you need to target Zen1/Zen2.

Supports clang, gcc, and MSVC. Simple tests are provided and have been exercised on all three.
