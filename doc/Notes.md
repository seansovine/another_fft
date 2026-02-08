# Notes on convolution performance

We've added `scripts/perf_test.sh` and `scripts/conv_test_perf.sh`. These scripts will run
different implementations of an algorithm and then re-run them with `perf stat`.

Perf will output something like

```text
 Performance counter stats for 'target/release/image_processing ... conv-test test1':

          4,448.17 msec task-clock                       #    1.013 CPUs utilized
               961      context-switches                 #  216.044 /sec
                20      cpu-migrations                   #    4.496 /sec
         3,476,556      page-faults                      #  781.569 K/sec
    13,105,718,106      cpu_atom/cycles/                 #    2.946 GHz                         (0.29%)
    18,924,267,903      cpu_core/cycles/                 #    4.254 GHz                         (89.54%)
    29,553,009,231      cpu_atom/instructions/           #    2.25  insn per cycle              (0.31%)
    38,470,184,624      cpu_core/instructions/           #    2.03  insn per cycle              (89.54%)
     2,007,472,479      cpu_atom/branches/               #  451.303 M/sec                       (0.32%)
     4,609,246,461      cpu_core/branches/               #    1.036 G/sec                       (89.44%)
         9,256,946      cpu_atom/branch-misses/          #    0.46% of all branches             (0.31%)
        15,219,623      cpu_core/branch-misses/          #    0.33% of all branches             (79.57%)
             TopdownL1 (cpu_core)                 #     27.2 %  tma_backend_bound
                                                  #     13.1 %  tma_bad_speculation
                                                  #     17.2 %  tma_frontend_bound
                                                  #     42.5 %  tma_retiring             (89.51%)
             TopdownL1 (cpu_atom)                 #     18.1 %  tma_bad_speculation
                                                  #     33.3 %  tma_retiring             (0.31%)
                                                  #     34.7 %  tma_backend_bound
                                                  #     34.7 %  tma_backend_bound_aux
                                                  #     13.9 %  tma_frontend_bound       (0.34%)
     6,881,572,129      L1-dcache-loads                  #    1.547 G/sec                       (0.27%)
     7,575,874,615      L1-dcache-loads                  #    1.703 G/sec                       (89.55%)
   <not supported>      L1-dcache-load-misses
        26,898,988      L1-dcache-load-misses            #    0.36% of all L1-dcache accesses   (89.48%)
        18,537,217      LLC-loads                        #    4.167 M/sec                       (0.27%)
         4,445,579      LLC-loads                        #  999.416 K/sec                       (89.51%)
           207,619      LLC-load-misses                  #    1.12% of all LL-cache accesses    (0.24%)
         2,237,590      LLC-load-misses                  #   50.33% of all LL-cache accesses    (89.53%)

       4.393057403 seconds time elapsed

       1.792180000 seconds user
       2.658268000 seconds sys

---------------------------------

 Performance counter stats for 'target/release/image_processing ... conv-test naive1':

          2,651.56 msec task-clock                       #    1.022 CPUs utilized
               868      context-switches                 #  327.354 /sec
                22      cpu-migrations                   #    8.297 /sec
           383,238      page-faults                      #  144.533 K/sec
     8,477,259,023      cpu_atom/cycles/                 #    3.197 GHz                         (0.37%)
    11,463,835,913      cpu_core/cycles/                 #    4.323 GHz                         (89.09%)
    10,296,043,894      cpu_atom/instructions/           #    1.21  insn per cycle              (0.47%)
    30,164,248,177      cpu_core/instructions/           #    2.63  insn per cycle              (89.03%)
     1,247,493,728      cpu_atom/branches/               #  470.475 M/sec                       (0.57%)
     1,752,333,860      cpu_core/branches/               #  660.868 M/sec                       (89.11%)
         4,743,251      cpu_atom/branch-misses/          #    0.38% of all branches             (0.67%)
        12,483,048      cpu_core/branch-misses/          #    0.71% of all branches             (79.23%)
             TopdownL1 (cpu_core)                 #     25.7 %  tma_backend_bound
                                                  #     13.2 %  tma_bad_speculation
                                                  #     16.6 %  tma_frontend_bound
                                                  #     44.5 %  tma_retiring             (89.10%)
             TopdownL1 (cpu_atom)                 #      7.3 %  tma_bad_speculation
                                                  #     37.5 %  tma_retiring             (0.71%)
                                                  #     41.2 %  tma_backend_bound
                                                  #     41.2 %  tma_backend_bound_aux
                                                  #     14.0 %  tma_frontend_bound       (0.75%)
     3,060,129,373      L1-dcache-loads                  #    1.154 G/sec                       (0.44%)
     3,139,078,422      L1-dcache-loads                  #    1.184 G/sec                       (89.13%)
   <not supported>      L1-dcache-load-misses
        16,302,223      L1-dcache-load-misses            #    0.52% of all L1-dcache accesses   (89.20%)
         5,361,559      LLC-loads                        #    2.022 M/sec                       (0.34%)
         1,704,339      LLC-loads                        #  642.768 K/sec                       (89.13%)
           108,216      LLC-load-misses                  #    2.02% of all LL-cache accesses    (0.30%)
           883,474      LLC-load-misses                  #   51.84% of all LL-cache accesses    (89.17%)

       2.593313357 seconds time elapsed

       2.248494000 seconds user
       0.405449000 seconds sys
```

for each run.

I'm just getting started with this kind of low-level performance profiling. Up to now my performance
practices have been more informed by algorithmic complexity and some intuitive pictures I've developed
over time about how hardware and compilers work. I'm just getting a feel for the measurement process
with perf and other profilers and am doing research to develop a good methodology.

## First thoughts on perf results for convolution

The overall number of data cache loads is significantly higher for our test method vs. the naive method,
and the percentage of LLC-load-misses that make it all the way to RAM is three times as high.
This may be due to the fact that we're adding the convolution data for each pixel into the
output buffer in three separate operations in the test method, which results in that data being loaded
three separate times.

The test method also executes significantly more instructions than the naive method, but seems to have
slightly better instruction throughput. This is a bit surprising to me, because both methods are
performing the same arithmetic operations, just in a different order. Their equivalence can be verified
by setting the test to write out the convolved images and then comparing them. The instruction difference
could be a result of compiler optimizations; I can follow up with a look at the generated assembly for both
versions. It does seems possible or likely that the naive version is written in a way that's easier for the
compiler to optimize.

The test version is writing into a 64-bit floating point buffer, whereas the "naive" implementation (maybe
not so naive after all) does arithmetic in 64-bit floating point and then casts to unsigned bytes before
saving to the output buffer. This isn't really necessary for these kernels, which have integral entries,
but I wanted to make the method adaptable to more general kernels. However, it seems that this choice
could also affect the efficiency of the method, which could be a tradeoff in some situations if we need
to combine multiple convolutions in several steps with higher precision.

At this point I can only draw tentative conclusions. But it's clear the "naive" implementation performs
much better: It executes significantly fewer instructions; has signficantly fewer cache misses; it
takes about half as long to run; and it doesn't require converting the image pixel type when writing
the final output image.

## Next steps

I plan to keep looking into performance profiling and testing methodology and will keep working on more
hardware-aware performance optimization strategies. For this particular convolution use case, I'm not
sure what more can be done to optimize without using vectorized operations explicitly (the compiler may
be inserting them already; we can check the assembly) or doing the work on a GPU. But I will certainly
keep researching and studying the problem.

The particular case we're addressing is convolution with a single abitrarily-sized image using a small
(3 x 3) kernel. More involved methods like im2col can be used for batch convolutions, but many such methods
involve an initial transformation that might outweight and gains for our use case. I'll keep researching
available approaches to this problem.

We have a parallelized version of the operation using Rayon. It is also not giving as much of a performance
improvement as we might expect. Each thread is writing to a disjoint segment of data, but there is some
overlap in the constant data that is being read. [This](https://gendignoux.com/blog/2024/11/18/rust-rayon-optimized.html)
post contains some ideas and an open source Rust parallization library with a simlar API to Rayon, that
may perform better for our use case. I plan to look into that too.
