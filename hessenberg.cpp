#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include "../linearalgebra.h"

/* ----------------------- hessenberg ----------------------- */
/*  Given a matrix A of dimension m by m and arrays v_i of 
    dimension m-i, for i = 1, ..., m - 1, respectively, this 
    algorithm computes m reflection vectors and the factor H 
    of a Hessenberg decomposition of A = Q H Q^*. The n 
    reflection vectors are stored in the arrays v_1, ..., v_n 
    and the columns of A are overwritten by the columns of H.
    
    Input variables:
        a: pointer to array of arrays, the ith array of
            which should correspond to the ith column of the 
            matrix A. During the algorithm, the columns of R 
            will overwrite the columns of A.
        v: pointer to array of arrays in which the ith 
            reflection vector of dimension m - i will be 
            stored.
        m: number of rows and number of columns of A.

    Features: The number of flops for this implementation is
    ~ (10/3) * m^3 and requires O(1) additional 
    memory.                                                    */

void hessenberg (double ** a, double ** v, int m) {
    int i, j;
    double vnorm, vTa, vpartdot;

    for(i = 0; i < m - 1; i++) {
        /* set v[i] equal to subvector a[i][i : m] */
        partialvec_copy(a[i], v[i], m - i - 1, i + 1);

        /* vpartdot = ||v[i]||^2 - v[i][0] * v[i][0]; since vpartdot 
           is unaffected by the change in v[i][0], storing this value 
           prevents the need to recalculate the entire norm of v[i] 
           after updating v[i][0] in the following step              */
        vpartdot = partialdot_product(v[i], v[i], m - i - 1, 1);

        /* set v[i][0] = v[i][0] + sign(v[i][0]) * ||v[i]|| */
        if(v[i][0] < 0) {
            v[i][0] -= sqrt(v[i][0] * v[i][0] + vpartdot);
        }
        else {
            v[i][0] += sqrt(v[i][0] * v[i][0] + vpartdot);
        }

        /* normalize v[i] */
        vnorm = sqrt(v[i][0] * v[i][0] + vpartdot);
        scalar_div(v[i], vnorm, m - i - 1, v[i]);
    
        for(j = i; j < m; j++) {
            /* a[j][i+1:m] = a[j][i+1:m] - 2 * (v[i]^T a[j][i+1:m]) * v[i] */
            vTa = 2 * subdot_product(a[j], v[i], m - i - 1, i + 1);
            partialscalar_sub(v[i], vTa, m - i - 1, i + 1, a[j]);
        }

        for(j = i; j < m; j++) {
            /* a[i+1:m][j] = a[i+1:m][j] - 2 * (a[i+1:m][j] v[i]) * v[i]^T */
            vTa = 2 * submatrow_product(a, v[i], m - i - 1, i + 1, j);
            matrixrow_sub(v[i], vTa, m - i - 1, i + 1, j, a);
        }
    }
}


int main () {
    int i, j, m, sym;
    double x;

    /* let user set the dimension of matrix A */
    printf("Enter the dimension m (where A is a m by m matrix): ");
    scanf("%i", &m);
    printf("Enter either 0 to test a nonsymmetric matrix\n"
           "          or 1 to test a symmetric matrix: ");
    scanf("%i", &sym);

    /* allocate memory for A and vectors v */
    double ** a = new double*[m];
    double ** v = new double*[m - 1];
    a[0] = new double[m];
    for(i = 1; i < m; i++) {
        a[i] = new double[m];
        v[i - 1] = new double[m - i];
    }

    /* initialize the values in matrix A */
    for(i = 0; i < m; i++) {
        for(j = 0; j < m; j++) {
            if(j < i) {
                if(sym) {
                    a[i][j] = i - j + 1;
                }
                else {
                    a[i][j] = i * j + 1;
                }
            }
            else {
                a[i][j] = j - i + 1; // this choice of values was arbitrary
            }
        }
    }

    /* print the matrix A before calling hessenberg */
    printf("A = \n");
    for(i = 0; i < m; i++) {
        for(j = 0; j < m; j++) {

            printf("%9.6g ", a[j][i]);
        }
        printf("\n");
    }
    printf("\n");

    /* execute householder recudtion to hessenberg form */
    hessenberg(a, v, m);

    /* print the matrix R (stored in A) after calling houheholder */
    printf("R = \n");
    for(i = 0; i < m; i++) {
        for(j = 0; j < m; j++) {
            printf("%9.6g ", a[j][i]);
        }
        printf("\n");
    }
    printf("\n");

    /* print the vectors v after calling hessenberg */
    for(i = 0; i < m - 1; i++) {
        printf("v[%i] = ", i);
        for(j = 0; j < m - i - 1; j++) {
            printf("%9.6g ", v[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    /* print numerical evidence that v's are normalized */
    printf("Numerical verification that v_1, ..., v_%i are "
           "normalized:\n", m - 1);
    for(i = 1; i < m - 1; i++) {
        x = dot_product(v[i - 1], v[i - 1], m - i);
        printf("||v[%i]|| = %lg, ", i, x);
        if(i % 5 == 0) {
            printf("\n");
        }
    }
    x = dot_product(v[m - 2], v[m - 2], 1);
        printf("||v[%i]|| = %lg.", m - 1, x);
    if(m % 5 != 0) printf("\n");
    printf("\n");

    /* free memory */
    for(i = 0; i < m - 1; i++) {
        delete[] a[i];
        delete[] v[i];
    }
    delete[] a[i];
    delete[] a;
    delete[] v;
    return 0;
}
