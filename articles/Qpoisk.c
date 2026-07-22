#include <math.h>
#include <stdio.h>                                                                           
#define ADIM 3    //max cislo parametrov modeli    (n<=ADIM)

double Urand (long *iy);
double fi(double* x); 
double vbrus(long n,double*  gl,double* gg);
void q2poisk(long n,double* x,double* xl,double* xg,
            double* r1,double* r2,double* r3,long max_prob,double eps_brus); 

long iurand=1;  //servis generatora sluch cisel
double pi=3.1415926535;

double fi(double* x) 
{ double d;
// MODEL
d=sin(x[1]-pi)*sin(x[1]-pi)+cos(x[2]+0.5*pi)*cos(x[2]+0.5*pi);
return(d);}//pfi


int main(void)
{ long n,nprob; double arec[ADIM+1],al[ADIM+1],ag[ADIM+1],r1[ADIM+1],r2[ADIM+1],r3[ADIM+1],eps_brus;
// MODEL
n=2;                 //cislo optimiziruemix peremennix 
arec[1]=1.0; al[1]=0.0; ag[1]= 3.0; // granici parametra x[1]
arec[2]=2.0; al[2]=0.0; ag[2]=10.0; // granici parametra x[2]

//ALG PARAMETRI
nprob=200;           //max cislo prob
eps_brus=1e-4;       //min dopustimii brus    

q2poisk  (n,arec,al,ag,r1,r2,r3,nprob,eps_brus);
printf("\narec= %24.12e %24.12e",arec[1],arec[2]);

return(0);}//main



////
void q2poisk(long n,double* x,double* xl,double* xg,
            double* r1,double* r2,double* r3,long max_prob,double eps_brus) 
{ //  Uproshenii algoritm Qpoisk
  //  Mnogomernoe mnogokratnoi bezotvetstvenoe sjatie dopustimaga brusa
  //  Goditsa dla lubih razmernostei (n>=1).
  //  No luchse svego rabotaet dlia  n=2,3,4
  //  Kriterii okonchania - cislo prob (=iteracii) previshaet max_prob
  //  Alg parametr eps_brus zadaet tochost odnogo sjatia. Recomeduetsa 1e-4 - skolko nado znachashi cifr v rezultate
  
long iter=0,i,ifsum; double ftek,fnew,dvbrus0,dvbrus;    

dvbrus0=vbrus(n,xl,xg)/(double)n;
ifsum=0; ftek=fi(x); 

mobn_qpoisk:;
for (i=1; i<=n; i++)     {
r1[i]=xl[i]; r2[i]=xg[i];}

mkqpoisk: iter++; 
if (ifsum>max_prob) goto kon_k2poisk;

dvbrus=vbrus(n,r1,r2)/(double)n;
printf("\n%7ld FZ(QPOISK  )=%16.8e [%9.1e]",iter,ftek,dvbrus/dvbrus0);

//// gener trial point
for (i=1; i<=n; i++) r3[i]=r1[i]+Urand(&iurand)*(r2[i]-r1[i]);

fnew=fi(r3);   ifsum++;
if (fnew<ftek)    { ftek=fnew;  
for (i=1; i<=n; i++) 
if (r3[i]<x[i]) r2[i]= x[i]; 
else            r1[i]= x[i];

for (i=1; i<=n; i++) x[i]=r3[i];
                  }//if fnew
else              {
for (i=1; i<=n; i++) 
if (r3[i]<x[i]) r1[i]=r3[i];
else            r2[i]=r3[i]; 
                  }//else

if (dvbrus/dvbrus0<eps_brus) goto mobn_qpoisk;
                             goto mkqpoisk;
kon_k2poisk:; 
}//q2poisk

double Urand (long *iy)
{ // Generator sluh cisel
  char   shift;
  long   ia, ic, mic, m2 = 1;
  double halm, s;

  shift = (sizeof(m2) * 8) - 2;
  m2 = m2 << shift;

  halm = (double) m2;
  ia = 8 * (long) (halm * atan (1.E0) / 8.E0) + 5;
  ic = 2 * (long) (halm * (0.5E0 - sqrt (3.E0) / 6.E0)) + 1;
  mic = (m2 - ic) + m2;
  s = 0.5 / halm;

  (*iy) = (*iy) * ia;
  if ((*iy) > mic)
    (*iy) = ((*iy) - m2) - m2;
  (*iy) = (*iy) + ic;
  if ((*iy) / 2 > m2)
    (*iy) = ((*iy) - m2) - m2;
  if ((*iy) < 0)
    (*iy) = ((*iy) + m2) + m2;
  return ((double) (*iy) * s);
} //Urand


double vbrus(long n,double*  gl,double* gg)
{  // L1 razmer brusa
long i; double d; d=0.0;
for (i=1; i<=n; i++)
d+=gg[i]-gl[i];
return(d);}//vbrus

