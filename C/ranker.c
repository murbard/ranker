#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_spblas.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define M_α 1.2
#define M_β 2.0
#define M_lnβ 0.693147180559945309
#define M_lnΓ_β 0.0
#define M_lnΓ_α -0.0853740900033158


typedef struct {
    double v;
    gsl_vector *z;
} instance;

typedef struct {
    int n;
    double ln_α;
    double ln_β;
    gsl_vector *μ; // dimension n
    gsl_vector *ln_σ; // dimension n
} ranker;

typedef struct {
    gsl_spmatrix *X;
} observations;

typedef struct {
    gsl_spmatrix *obs;
    gsl_vector *diag;
    double αβ;
    gsl_vector *μσαβ; // 2n vector representing a 2n by 2 matrix where columns are identical
} hessian;


hessian * hessian_alloc(int n) {
    hessian *h = (hessian*)malloc(sizeof(hessian));
    h->obs = gsl_spmatrix_alloc(2 * n, 2 * n);
    h->diag = gsl_vector_alloc(2 * n + 2);
    h->μσαβ = gsl_vector_alloc(2 * n);
    return h;
}

void hessian_free(hessian *h) {
    gsl_spmatrix_free(h->obs);
    gsl_vector_free(h->diag);
    gsl_vector_free(h->μσαβ);
    free(h);
}


void hessian_dot(hessian h, gsl_vector* x, gsl_vector* out) {
    int n = (x->size - 2)/2;
    // the dimensions are 2 * n + 2

    // view of diag, x, and out, excluding last two elements, called diag_μσ, x_μσ, and out_μσ
    gsl_vector_view diag_μσ = gsl_vector_subvector(h.diag, 0, 2 * n);
    gsl_vector_view x_μσ = gsl_vector_subvector(x, 0, 2 * n);
    gsl_vector_view out_μσ = gsl_vector_subvector(out, 0, 2 * n);

    // out_μσ = diag_μσ * x_μσ + μσαβ * (x[2 * n] + x[2 * n + 1])
    gsl_vector_memcpy(&out_μσ.vector, &diag_μσ.vector);
    gsl_vector_mul(&out_μσ.vector, &x_μσ.vector);
    gsl_blas_daxpy(x->data[2 * n] + x->data[2 * n + 1], h.μσαβ, &out_μσ.vector);

    // out[2 * n] = μσαβ . x_μσ + diag[2 * n] * x[2 * n] + αβ * x[2 * n + 1]
    // out[2 * n + 1] = μσαβ . x_μσ + diag[2 * n + 1] * x[2 * n + 1] + αβ * x[2 * n]
    double μσαβx;
    gsl_blas_ddot(h.μσαβ, &x_μσ.vector, &μσαβx);
    out->data[2 * n] = μσαβx + h.diag->data[2 * n] * x->data[2 * n] + h.αβ * x->data[2 * n + 1];
    out->data[2 * n + 1] = μσαβx + h.αβ * x->data[2 * n] + h.diag->data[2 * n + 1] * x->data[2 * n + 1];

    // out[:-2] += h.obs @ x[:-2] + h.obs.T @ x[:-2]
    // remember that out is a sparse upper triangular matrix representing a symmetric matrix
    gsl_spblas_dgemv(CblasNoTrans, 1.0, h.obs, &x_μσ.vector, 1.0, &out_μσ.vector);
    gsl_spblas_dgemv(CblasTrans, 1.0, h.obs, &x_μσ.vector, 1.0, &out_μσ.vector);
    return ;
}

double evaluate(ranker ranker, observations obs, gsl_vector *gradient, hessian *hessian, gsl_matrix *dobs) {
    const int n = ranker.n;


    // Set some constants for easy access
    const double α = exp(ranker.ln_α);
    const double β = exp(ranker.ln_β);
    const double ln_β = ranker.ln_β;
    const double αβ = α * β;
    const double psi_α = gsl_sf_psi(α);
    const double psi1_α = gsl_sf_psi_1(α);
    const double psi2_α = gsl_sf_psi_n(2, α);

    gsl_vector *σ2 = gsl_vector_calloc(n);

    // Compute sum of μi² + σi², and sum of ln(σi)
    double Σ_μ2_σ2 = 0.0;
    double Σ_lnσ = 0.0;
    for(int i = 0; i < n; ++i) {
        double μi = ranker.μ->data[i];
        double lnσi = ranker.ln_σ->data[i];
        double σi2 = exp(2 * lnσi);
        σ2->data[i] = σi2;
        double μi2 = μi * μi;
        Σ_μ2_σ2 += μi2 + σi2;
        Σ_lnσ += lnσi;
    }

    const double Mβ = 2.0;
    const double Mα = 1.2;


    // First the terms that do not depend on observations

    double v = - α + αβ / Mβ + Mα * log(Mβ) - 0.5 * n * (1 + log(2 * M_PI)) +
     ln_β + gsl_sf_lngamma(Mα) - gsl_sf_lngamma(α) + (α - Mα) * psi_α - (1 + Mα) * ln_β
     - Σ_lnσ + 0.5 * (αβ * Σ_μ2_σ2  - n * (ln_β + psi_α - log(2 * M_PI)));

    if (gradient != NULL) {
        // Compute gradient
        // dv/dα
        gradient->data[2 * n] = 0.5 * α *( -2 + 2 * β / Mβ - (2 * Mα + n - 2 * α) * psi1_α + β * Σ_μ2_σ2);
        // dv/dβ
        gradient->data[2 * n + 1] = - Mα - 0.5 * n + αβ / Mβ + 0.5 * αβ * Σ_μ2_σ2;
        // dv/dμi and dv/dlnσi
        // set the first n elements of gh.g to αβ * ranker.μ using daxpy
        // get a view of the first n elements of gh.g
        gsl_vector_view g_μ = gsl_vector_subvector(gradient, 0, n);
        gsl_blas_daxpy(αβ, ranker.μ, &g_μ.vector);
        // set the elements n to 2n of gh.g to -1 + αβ * σ2
        gsl_vector_view g_σ = gsl_vector_subvector(gradient, n, n);
        gsl_vector_memcpy(&g_σ.vector, σ2);
        gsl_vector_scale(&g_σ.vector, αβ);
        gsl_vector_add_constant(&g_σ.vector, -1.0);
    }

    if (hessian != NULL) {
        // Compute hessian
        // d²v/dμi² and d²v/dlnσi²
        // set the first n elements of gh.h.diag to 1 and the next n elements to 2 * σ2
        gsl_vector_view diag_μ = gsl_vector_subvector(hessian->diag, 0, n);
        gsl_vector_set_all(&diag_μ.vector, 1.0);
        gsl_vector_view diag_σ = gsl_vector_subvector(hessian->diag, n, n);
        gsl_vector_memcpy(&diag_σ.vector, σ2);
        gsl_vector_scale(&diag_σ.vector, 2.0);
        // scale the whole thing by αβ
        gsl_vector_scale(hessian->diag, αβ);

        // d²v/dα²
        hessian->diag->data[2 * n] =
            0.5 * α *(-2 + 2 * β / Mβ - (2 * Mα + n - 4 * α) * psi1_α + β * Σ_μ2_σ2 - (2 * Mα + n - 2 * α) * α * psi2_α);

        // d²v/dαdβ
        hessian->αβ = 0.5 * αβ * (2 / Mβ + Σ_μ2_σ2);

        // d²v/dβ²
        hessian->diag->data[2 * n + 1] = hessian->αβ; // same value as the cross term

        // d²v/d{μ|σ}id{α|β}
        // place vector ranker.μ as the first half of the gh.h.μσαβ vector
        gsl_vector_view αβ_μ = gsl_vector_subvector(hessian->μσαβ, 0, n);
        gsl_vector_memcpy(&αβ_μ.vector, ranker.μ);
        // place vector σ2 as the second half of the gh.h.μσαβ vector
        gsl_vector_view αβ_σ = gsl_vector_subvector(hessian->μσαβ, n, n);
        gsl_vector_memcpy(&αβ_σ.vector, σ2);
        // scale gh.h.μσαβ by αβ
        gsl_vector_scale(hessian->μσαβ, αβ);
    }

    // Now we add observations
    // We loop over all non zero elements of the sparse observation matrix
    for(int k = 0; k < obs.X->nz ; ++k) {
        // Get the row and column of the current element
        int i = obs.X->i[k];
        int j = obs.X->p[k];
        if (i == j)
            continue; // this should not happen
        double count = obs.X->data[k];

        double σδ = sqrt(σ2->data[i] + σ2->data[j]);
        double h = sqrt(16.0 / M_PI + 2 * σδ * σδ);
        double sqrt_π_h = sqrt(M_PI) * h;
        double μδ = ranker.μ->data[i] - ranker.μ->data[j];
        double μδ_over_h = μδ / h;
        double exp_minus_μδ_over_h_square = exp(-μδ_over_h * μδ_over_h);
        double gμδ = - 0.5 * (1 - erf(μδ_over_h));
        double gσδ = - exp_minus_μδ_over_h_square * σδ / sqrt_π_h;

        v += count * (exp_minus_μδ_over_h_square * h / (2 * sqrt(M_PI)) - 0.5 * μδ * gμδ);

        if (gradient != NULL) {
            // Compute gradient
            // dv/dμi, dv/dμj
            gradient->data[i] += count * gμδ;
            gradient->data[j] -= count * gμδ;

            // dv/dlnσi, dv/dlnσj
            gradient->data[n + i] -= count * (gσδ * σ2->data[i] / σδ);
            gradient->data[n + j] -= count * (gσδ * σ2->data[j] / σδ);
        }

        if (hessian != NULL) {
            /*
                hμμδ = - exp(-(μδ/h)**2) / (sqrt(π) * h)
                hμσδ = 2 * μδ * σδ * exp(-(μδ/h)**2) / (sqrt(π) * h**3)
                hσσδ = - exp(-(μδ/h)**2) * (256 + 4 * π * σδ**2 * (8 + μδ**2)) / (π**(5/2) * h**5)
            */
            double hμμδ = - exp_minus_μδ_over_h_square / sqrt_π_h;
            double hμσδ = 2 * μδ * σδ * exp_minus_μδ_over_h_square / (sqrt(M_PI) * h * h * h);
            double sqrt_π_h5 = sqrt_π_h * sqrt_π_h; sqrt_π_h5 *= sqrt_π_h5; sqrt_π_h5 *= sqrt_π_h;
            double hσσδ = - exp_minus_μδ_over_h_square * (256 + 4 * M_PI * σδ * σδ * (8 + μδ * μδ)) / sqrt_π_h5;

            // diagonal hessian terms
            hessian->diag->data[i] -= count * hμμδ;
            hessian->diag->data[j] -= count * hμμδ;
            hessian->diag->data[n+i] -= count * σ2->data[i] / σδ * (gσδ * σ2->data[j] / (σδ * σδ) + hσσδ * σ2->data[i] / σδ + gσδ);
            hessian->diag->data[n+j] -= count * σ2->data[j] / σδ * (gσδ * σ2->data[i] / (σδ * σδ) + hσσδ * σ2->data[j] / σδ + gσδ);


            int row = GSL_MIN(i, j);
            int col = GSL_MAX(i, j);
            // there are 6 entries in the sparse matrix, hμiμj, hμiσi, hμiσj, hμjσi, hμjσj, hσiσj

            // hμiμj
            gsl_spmatrix_set(hessian->obs, row, col, count * hμμδ);
            // hμiσi, hμjσi
            double chσ = count * hμσδ * σ2->data[i] / σδ;
            gsl_spmatrix_set(hessian->obs, i, n + i, - chσ);
            gsl_spmatrix_set(hessian->obs, j, n + i, chσ);

            // hμiσj, hμjσj
            chσ = count * hμσδ * σ2->data[j] / σδ;
            gsl_spmatrix_set(hessian->obs, i, n + j, - chσ);
            gsl_spmatrix_set(hessian->obs, j, n + j, chσ);

            // hσiσj
            gsl_spmatrix_set(hessian->obs, n + row, n + col, - count * (σ2->data[i] * σ2->data[j] / (σδ * σδ) * (hσσδ - gσδ / σδ)));
        }
    }
    return v;
}

int main() {
    int n = 3;
    ranker ranker;
    ranker.n = n;
    ranker.μ = gsl_vector_calloc(n);
    ranker.ln_σ = gsl_vector_calloc(n);
    ranker.ln_α = log(0.5);
    ranker.ln_β = log(1.0);

    // initialize ranker.μ->data to {0.1, -0.2, 0.3}
    ranker.μ->data[0] = 0.1;
    ranker.μ->data[1] = -0.2;
    ranker.μ->data[2] = 0.3;
    // initialize ranker.ln_σ->data to log({1.1, 2.2, 3.3})
    ranker.ln_σ->data[0] = log(1.1);
    ranker.ln_σ->data[1] = log(2.2);
    ranker.ln_σ->data[2] = log(3.3);
    // initialize ranker.ln_α to log(1.5) and ranker.ln_β to log(3.0)
    ranker.ln_α = log(1.5);
    ranker.ln_β = log(3.0);

    observations obs;
    obs.X = gsl_spmatrix_alloc(1, 1);

    gsl_vector * g = gsl_vector_calloc(2 * n + 2);
    hessian * h = hessian_alloc(n);

    double v = evaluate(ranker, obs, g, h, NULL);
    printf("%lf\n", v);
    for(int i = 0; i < 2 * n + 2; ++i) {
        printf("g: %lf\n", g->data[i]);
    }
    // print all parts of the hessian
    for(int i = 0; i < 2 * n + 2; ++i) {
        printf("diag: %lf\n", h->diag->data[i]);
    }
    printf("αβ: %lf\n", h->αβ);
    for(int i = 0; i < 2 * n; ++i) {
        printf("μσαβ: %lf\n", h->μσαβ->data[i]);
    }


    return 0;
}
