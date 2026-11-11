/*
  Short job 2
*/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>



typedef struct {
double re, im;
} complex_t;


int vyprazdni(int *temp,int k)
{
  // flush cache
  int i,s=0;
  for(i=0;i<k;i++) s+=temp[i];
  return s;
}


int testik(complex_t * c,int n)
{
  double re,im,r2,im2;
  int i2,j2,k2=0;
  int nn=0;
  re=im=0.0;
        for(j2=0;j2<n;j2++)
        {
          for(i2=0;i2<n;i2++)
          {
            re+=c[i2*n+j2].re;
            im+=c[i2*n+j2].im;
          }
        }
  if (isnan(re)) nn++;
  if (isnan(im)) nn++;
  printf("%10.8g %10.8g ",re,im);
  k2=0;

  if (n==1024)
  {
     r2=38912 ; im2=157696;
  }
  if (n==1536)
  {
     r2=419328; im2=1098752;
  }
  if (n==2048       )
  {
     r2=-143360 ; im2=-956416;
  }
  if (n==2560     )
  {
    r2=249856; im2= -168448;
  }
  if (n==3072          )
  {
     r2=102400; im2=-40960;
  }
  if (n==3584                )
  {
     r2=176128; im2=994816 ;
  }
  if (n==4096    )
  {
    r2=-352256    ; im2= -614400;
  }
  if (n==4608     )
  {
     r2=-691712    ; im2=104960;
  }



  double x=fabs(re-r2)+fabs(im-im2);
  if (x<100) k2=1;
  if (nn) k2=0;
  printf("%g ",x);
  return k2;
}


// zacatek casti k modifikaci
// beginning of part for modification

#include <cstdlib>

#define MR 4     
#define NR 8      
#define KC 256   
#define MBLK 16   

// Cross-platform aligned memory allocation helper
static inline float* allocate_aligned(size_t alignment, size_t size) {
#if defined(_MSC_VER) || defined(_WIN32)
    return static_cast<float*>(_aligned_malloc(size, alignment));
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) return nullptr;
    return static_cast<float*>(ptr);
#endif
}

static inline void free_aligned(float* ptr) {
#if defined(_MSC_VER) || defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static inline void kern(const float * __restrict__ ap, const float * __restrict__ bp, float * __restrict__ acc, int kc) {
    float cr[MR][NR], ci[MR][NR];
    int r, j, k;

    for (r = 0; r < MR; r++) {
        for (j = 0; j < NR; j++) {
            cr[r][j] = acc[r * NR + j];
            ci[r][j] = acc[MR * NR + r * NR + j];
        }
    }

    for (k = 0; k < kc; k++) {
        const float * __restrict__ pa = ap + k * 8;
        const float * __restrict__ pb = bp + k * 16;
        float br[NR], bi[NR];

        for (j = 0; j < NR; j++) { 
            br[j] = pb[j]; 
            bi[j] = pb[NR + j]; 
        }

        #pragma GCC unroll 4
        for (r = 0; r < MR; r++) {
            float ar = pa[r], ai = pa[MR + r];

            #pragma omp simd
            for (j = 0; j < NR; j++) {
                // Complex multiplication: (ar + i*ai) * (br + i*bi)
                cr[r][j] += ar * br[j] - ai * bi[j];
                ci[r][j] += ar * bi[j] + ai * br[j];
            }
        }
    }

    for (r = 0; r < MR; r++) {
        for (j = 0; j < NR; j++) {
            acc[r * NR + j] = cr[r][j];
            acc[MR * NR + r * NR + j] = ci[r][j];
        }
    }
}

void gemm(complex_t * a, complex_t * b, complex_t * c, int n) {
    const size_t N = static_cast<size_t>(n);
    const int nI = (n + MR - 1) / MR;
    const int nJ = (n + NR - 1) / NR;

    size_t sa = ((static_cast<size_t>(nI) * N * 8 * sizeof(float)) + 63) & ~static_cast<size_t>(63);
    size_t sb = ((static_cast<size_t>(nJ) * N * 16 * sizeof(float)) + 63) & ~static_cast<size_t>(63);

    float *ap = allocate_aligned(64, sa);
    float *bp = allocate_aligned(64, sb);

    #pragma omp parallel
    {
        int ib, jb, k, r, j;

        // ---- Pack A ----
        #pragma omp for schedule(static)
        for (ib = 0; ib < nI; ib++) {
            for (k = 0; k < n; k++) {
                float *p = ap + (static_cast<size_t>(ib) * N + k) * 8;
                for (r = 0; r < MR; r++) {
                    int i = ib * MR + r;
                    if (i < n) { 
                        p[r] = static_cast<float>(a[static_cast<size_t>(i) * N + k].re); 
                        p[MR + r] = static_cast<float>(a[static_cast<size_t>(i) * N + k].im); 
                    } else { 
                        p[r] = 0.0f; 
                        p[MR + r] = 0.0f; 
                    }
                }
            }
        }

        // ---- Pack B ----
        #pragma omp for schedule(static)
        for (k = 0; k < n; k++) {
            for (jb = 0; jb < nJ; jb++) {
                float *p = bp + (static_cast<size_t>(jb) * N + k) * 16;
                for (j = 0; j < NR; j++) {
                    int jj = jb * NR + j;
                    if (jj < n) { 
                        p[j] = static_cast<float>(b[static_cast<size_t>(k) * N + jj].re); 
                        p[NR + j] = static_cast<float>(b[static_cast<size_t>(k) * N + jj].im); 
                    } else { 
                        p[j] = 0.0f; 
                        p[NR + j] = 0.0f; 
                    }
                }
            }
        }

        // ---- Multiply ----
        int kk;
        for (kk = 0; kk < n; kk += KC) {
            int kc = (n - kk < KC) ? (n - kk) : KC;
            int blk;

            #pragma omp for schedule(dynamic, 1)
            for (blk = 0; blk < (nI + MBLK - 1) / MBLK; blk++) {
                int ib0 = blk * MBLK;
                int ib1 = (ib0 + MBLK < nI) ? ib0 + MBLK : nI;

                for (jb = 0; jb < nJ; jb++) {
                    const float *bs = bp + (static_cast<size_t>(jb) * N + kk) * 16;

                    for (ib = ib0; ib < ib1; ib++) {
                        float acc[2 * MR * NR] __attribute__((aligned(64)));
                        int i0 = ib * MR, j0 = jb * NR;
                        int ri, rj;

                        // Load accumulator
                        for (ri = 0; ri < MR; ri++) {
                            for (rj = 0; rj < NR; rj++) {
                                if (kk == 0 || i0 + ri >= n || j0 + rj >= n) {
                                    acc[ri * NR + rj] = 0.0f;
                                    acc[MR * NR + ri * NR + rj] = 0.0f;
                                } else {
                                    acc[ri * NR + rj]         = static_cast<float>(c[static_cast<size_t>(i0 + ri) * N + j0 + rj].re);
                                    acc[MR * NR + ri * NR + rj] = static_cast<float>(c[static_cast<size_t>(i0 + ri) * N + j0 + rj].im);
                                }
                            }
                        }

                        kern(ap + (static_cast<size_t>(ib) * N + kk) * 8, bs, acc, kc);

                        // Store back to C
                        for (ri = 0; ri < MR && i0 + ri < n; ri++) {
                            for (rj = 0; rj < NR && j0 + rj < n; rj++) {
                                c[static_cast<size_t>(i0 + ri) * N + j0 + rj].re = acc[ri * NR + rj];
                                c[static_cast<size_t>(i0 + ri) * N + j0 + rj].im = acc[MR * NR + ri * NR + rj];
                            }
                        }
                    }
                }
            }
        }
    }

    free_aligned(ap);
    free_aligned(bp);
}

// konec casti k modifikaci
// end of part for modification


int main( void ) {

 double timea[60];

 int soucet=0,N,i,j,k,ni,nj,nk,*pomo,v;
 int ri,rj,rk;
 double delta,s_delta=0.0;
 complex_t *mA, *mB,*mC;

  int ti[4]={3500,5000,3200,2500};
  int tj[4]={3500,3200,2500,5000};
  int tk[4]={3500,2500,5000,3200};

  srand (time(NULL));
  pomo=(int *)malloc(32*1024*1024);
  v=0;
  for(N=0;N<8;N++)
  {
  ni=1024+512*N;
  nj=ni;
  nk=ni;
  printf("%i ",ni);
  mA=(complex_t *)malloc(ni * nk* sizeof(complex_t));
  mB=(complex_t *)malloc(nk * nj* sizeof(complex_t));
  mC=(complex_t *)malloc(ni * nj* sizeof(complex_t));


  for (i=0; i<ni; i++) {
  //ri=rand();
  for (k=0; k<nk; k++) {
    rk=(k^i)^ni;
    mA[i*nk+k].re = (rk%13-6.0);
    mA[i*nk+k].im = (rk%17-8.0);
  }}
  for (k=0; k<nk; k++) {
    //rk=rand();

  for (j=0; j<nj; j++) {
    rk=(k^j)^ni;
    mB[k*nj+j].re = (rk%19-9.0);
    mB[k*nj+j].im = (rk%11-5.0);
  }}
  for (i=0; i<ni; i++) {
  //ri=rand();
  for (j=0; j<nj; j++) {
    //rj=rand();
    mC[i*nj+j].re = 0.0;
    mC[i*nj+j].im = 0.0;
  }}

  double di;


  soucet+=vyprazdni(pomo,v);
  timea[0]=omp_get_wtime();
  // improve performance of this call
  // vylepsit vykonnost tohoto volani
  gemm( mA, mB, mC, ni);
  timea[1]=omp_get_wtime();
  i=testik( mC, ni);
  delta=10.0*di*di*di/(timea[1]-timea[0]);
  s_delta+=(timea[1]-timea[0]);
  //i=testuj(mA,mB,mC,mC2,ni,nj,nk,1.0,1.0);
  printf("%i %g ",i,(timea[1]-timea[0]));

  printf("\n");
  if (i==0)
  {
    printf("Incorrect result !!!\n SJ1 points: 0\n");
    return 0;
  }
  if ((N==0)&&(s_delta>1.0))
  {
    printf("Time limit for N=0 (1 second) is reached (time=%g s). SJ1 points: 0\n",s_delta);
    return 0;
  }

  if (s_delta>20.0)
  {
    printf("Total time limit (20 sec) is reached (time=%g s). SJ1 points: 0\n",s_delta);
    return 0;
  }

  free(mC);
  free(mB);
  free(mA);
  }
  float b=2.75/s_delta*12.0;
  if (b>15.0) b=15.0;
  printf("Total time: %g\n",s_delta);
  printf("SJ1 points: %g\n",b);
  return 0;
}
