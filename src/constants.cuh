#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "../utils/utils.cuh"
#include "stdio.h"
#include <stdint.h>
#include <sys/time.h>
#include <time.h>

#define DEBUG_INFO false

#define lgWARP 5
#define WARP (1 << lgWARP)

#define RUNS_GPU 500
#define RUNS_CPU 5

#ifndef ELEMS_PER_THREAD
#define ELEMS_PER_THREAD 24
#endif

int timeval_subtract(struct timeval *result, struct timeval *t2,
                     struct timeval *t1) {
  uint32_t resolution = 1000000;
  int64_t diff = (t2->tv_usec + resolution * t2->tv_sec) -
                 (t1->tv_usec + resolution * t1->tv_sec);
  result->tv_sec = diff / resolution;
  result->tv_usec = diff % resolution;
  return (diff < 0);
}

#endif // CONSTANTS_H
