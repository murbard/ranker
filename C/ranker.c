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

const double Mβ = 2.0;
const double Mα = 1.2;


// Comments for doc generation

/// @brief Intance of a ranker problem
/// @details This structure contains all the hidden information describing a ranker problem
typedef struct {
    double v;
    gsl_vector *z;
} instance;

/// @brief Solution of a ranker problem
/// @details This structure contains all the information describing a solution of a ranker problem
typedef struct {
    int n;
    gsl_vector * params ; // dimension 2 * n + 1
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

/// @brief Allocate a hessian structure
/// @param n Dimension of the problem
/// @return A pointer to the allocated structure
hessian * hessian_alloc(const int n) {
    hessian *h = (hessian*)malloc(sizeof(hessian));
    h->obs = gsl_spmatrix_alloc(2 * n, 2 * n);
    h->diag = gsl_vector_alloc(2 * n + 2);
    h->μσαβ = gsl_vector_alloc(2 * n);
    return h;
}

/// @brief Free a hessian structure
/// @param h A pointer to the structure to free
void hessian_free(hessian * const h) {
    gsl_spmatrix_free(h->obs);
    gsl_vector_free(h->diag);
    gsl_vector_free(h->μσαβ);
    free(h);
}

ranker * ranker_alloc(const int n) {
    ranker *r = (ranker*)malloc(sizeof(ranker));
    r->n = n;
    r->params = gsl_vector_calloc(2 * n + 2);

    /*
        Given that the variance is picked from an inverse gamma distribution, the
        posterior should be a student t distribution with 2 a degrees of freedom
        and scaling factor √(b/a). We minimize the KL divergence between a normal
        and that distribution by picking a standard deviation approximately equal to
         √(b/a) (1 + 3 / (6 a + 2 a²))
    */
    double σ = sqrt(Mβ / Mα) * (1 + 3 / (6 * Mα + 2 * Mα * Mα));

    gsl_vector_view σs = gsl_vector_subvector(r->params, n, n);
    gsl_vector_set_all(&σs.vector, log(σ));
    gsl_vector_set(r->params, 2 * n, log(Mα));
    gsl_vector_set(r->params, 2 * n + 1, log(Mβ));

    return r;
}

void ranker_free(ranker * const r) {
    gsl_vector_free(r->params);
    free(r);
}

/// @brief Computes the matrix product of a hessian structure with a vector
/// @param h A pointer to the hessian structure
/// @param x A pointer to the vector
/// @param out A pointer to the vector where the result will be stored
void hessian_dot(const hessian* const h, const gsl_vector* const x, gsl_vector* const out) {
    int n = (x->size - 2)/2;
    // the dimensions are 2 * n + 2

    // view of diag, x, and out, excluding last two elements, called diag_μσ, x_μσ, and out_μσ
    gsl_vector_const_view diag_μσ = gsl_vector_const_subvector(h->diag, 0, 2 * n);
    gsl_vector_const_view x_μσ = gsl_vector_const_subvector(x, 0, 2 * n);
    gsl_vector_view out_μσ = gsl_vector_subvector(out, 0, 2 * n);

    // out_μσ = diag_μσ * x_μσ + μσαβ * (x[2 * n] + x[2 * n + 1])
    gsl_vector_memcpy(&out_μσ.vector, &diag_μσ.vector);
    gsl_vector_mul(&out_μσ.vector, &x_μσ.vector);
    gsl_blas_daxpy(x->data[2 * n] + x->data[2 * n + 1], h->μσαβ, &out_μσ.vector);

    // out[2 * n] = μσαβ . x_μσ + diag[2 * n] * x[2 * n] + αβ * x[2 * n + 1]
    // out[2 * n + 1] = μσαβ . x_μσ + diag[2 * n + 1] * x[2 * n + 1] + αβ * x[2 * n]
    double μσαβx;
    gsl_blas_ddot(h->μσαβ, &x_μσ.vector, &μσαβx);
    out->data[2 * n] = μσαβx + h->diag->data[2 * n] * x->data[2 * n] + h->αβ * x->data[2 * n + 1];
    out->data[2 * n + 1] = μσαβx + h->αβ * x->data[2 * n] + h->diag->data[2 * n + 1] * x->data[2 * n + 1];

    // out[:-2] += h.obs @ x[:-2] + h.obs.T @ x[:-2]
    // remember that out is a sparse upper triangular matrix representing a symmetric matrix
    gsl_spblas_dgemv(CblasNoTrans, 1.0, h->obs, &x_μσ.vector, 1.0, &out_μσ.vector);
    gsl_spblas_dgemv(CblasTrans, 1.0, h->obs, &x_μσ.vector, 1.0, &out_μσ.vector);
    return ;
}


void hessian_to_dense(const hessian* const h, const gsl_matrix* dense) {
    int n = (h->diag->size - 2)/2;
    // fill the diagonal
    for (int i = 0; i < 2 * n + 2; i++) {
        dense->data[i * (2 * n + 2) + i] = h->diag->data[i];

    }
    // fill the off-diagonal
    for (int i = 0; i < 2 * n; i++) {
        for (int j = i + 1; j < 2  * n; j++) {
            dense->data[i * (2 * n + 2) + j] += gsl_spmatrix_get(h->obs, i, j);
            dense->data[j * (2 * n + 2) + i] += gsl_spmatrix_get(h->obs, i, j);
        }
    }
    // fill in the αβ and βα terms
    dense->data[2 * n * (2 * n + 2) + 2 * n + 1] += h->αβ;
    dense->data[(2 * n + 1) * (2 * n + 2) + 2 * n] += h->αβ;

    // fill in the μσαβ and αβμσ terms
    for (int i = 0; i < 2 * n; i++) {
        dense->data[i * (2 * n + 2) + 2 * n] += h->μσαβ->data[i];
        dense->data[i * (2 * n + 2) + 2 * n + 1] += h->μσαβ->data[i];
        dense->data[2 * n * (2 * n + 2) + i] += h->μσαβ->data[i];
        dense->data[(2 * n + 1) * (2 * n + 2) + i] += h->μσαβ->data[i];
    }
}

void print_hessian(const hessian* const h) {
    int n = (h->diag->size - 2)/2;
    gsl_matrix *dense = gsl_matrix_calloc(2 * n + 2, 2 * n + 2);
    hessian_to_dense(h, dense);
    //gsl_matrix_fprintf(stdout, dense, "%g");
    // print in the Mathematica format on one line, i.e. {{...}, {...}, ...}
    printf("{");
    for (int i = 0; i < 2 * n + 2; i++) {
        printf("{");
        for (int j = 0; j < 2 * n + 2; j++) {
            printf("%g", dense->data[i * (2 * n + 2) + j]);
            if (j < 2 * n + 1) {
                printf(", ");
            }
        }
        printf("}");
        if (i < 2 * n + 1) {
            printf(", ");
        }
    }
    printf("}\n");
    gsl_matrix_free(dense);
}

// Solve H * x = y for x using the conjugate gradient method with Jacobi preconditioning
void cg(const hessian * const hessian, const gsl_vector * const y, gsl_vector * const x,  double λ, double tol, int max_iter) {

    // initialize x to 0
    gsl_vector_set_zero(x);

    gsl_vector *r0 = gsl_vector_alloc(y->size);
    // r0 = y - H * x0 but x0 = 0
    gsl_vector_memcpy(r0, y);

    //hessian_dot(hessian, x, r0);
    //gsl_vector_sub(r0, y);
    //gsl_vector_scale(r0, -1.0);

    gsl_vector *z0 = gsl_vector_alloc(y->size);
    // g0 = M^-1 * r0 with Jacobi conditionning (i.e. M is the diagonal of H)
    gsl_vector_memcpy(z0, r0);
    gsl_vector_div(z0, hessian->diag);
    gsl_vector_scale(z0, 1.0 / (1.0 + λ));

    // p0 = z0
    gsl_vector *p0 = gsl_vector_alloc(y->size);
    gsl_vector_memcpy(p0, z0);

    int k = 0;
    gsl_vector *Hp = gsl_vector_alloc(y->size);
    gsl_vector *x1 = gsl_vector_alloc(y->size);
    gsl_vector *r1 = gsl_vector_alloc(y->size);
    gsl_vector *p1 = gsl_vector_alloc(y->size);
    gsl_vector *z1 = gsl_vector_alloc(y->size);


    gsl_vector* r_[] = {r0, r1};
    gsl_vector* p_[] = {p0, p1};
    gsl_vector* z_[] = {z0, z1};
    gsl_vector* tmp = gsl_vector_alloc(y->size);

    while (k < max_iter) {

        double rkTzk, pkTHpk;
        gsl_blas_ddot(r_[k&1], z_[k&1], &rkTzk);
        // add (1 + λ) * hessian->diag . p to Hp

        gsl_vector_memcpy(tmp, p_[k&1]);
        gsl_vector_mul(tmp, hessian->diag);
        gsl_vector_scale(tmp, λ);
        hessian_dot(hessian, p_[k&1], Hp);
        gsl_vector_add(Hp, tmp);

        gsl_blas_ddot(p_[k&1], Hp, &pkTHpk);
        // αk = (rk . zk) / (pk . H . pk);
        double αk = rkTzk / pkTHpk;
        // xk+1 = xk + αk * pk
        gsl_blas_daxpy(αk, p_[k&1], x);
        // rk+1 = rk - αk * H * pk
        gsl_vector_memcpy(r_[(k+1)&1], r_[k&1]);
        gsl_blas_daxpy(-αk, Hp, r_[(k+1)&1]);
        // if r1 is sufficiently small, we are done, return x[k+1]
        // for debug print the norm of r[k+1]
        double norm_rk1 = gsl_blas_dnrm2(r_[(k+1)&1]);
        //printf("%d: norm of r = %lf\n", k, norm_rk1);
        // flush stdout
        fflush(stdout);

        if (gsl_blas_dnrm2(r_[(k+1)&1]) < tol)
            break;

        // zk1 = M^-1 * rk1 with Jacobi conditionning (i.e. M is the diagonal of H)
        gsl_vector_memcpy(z_[(k+1)&1], r_[(k+1)&1]);
        gsl_vector_div(z_[(k+1)&1], hessian->diag);
        gsl_vector_scale(z_[(k+1)&1], 1.0 / (1.0 + λ));

        // βk = (r1 . z1) / (r0 . z0)
        double rk1Tzk1;
        gsl_blas_ddot(r_[(k+1)&1], z_[(k+1)&1], &rk1Tzk1);
        double βk = rk1Tzk1 / rkTzk;
        // p1 = z1 + β0 * p0
        gsl_vector_memcpy(p_[(k+1)&1], z_[(k+1)&1]);
        gsl_blas_daxpy(βk, p_[k&1], p_[(k+1)&1]);
        k = k + 1;
    }
    free(r0); free(r1); free(p0); free(p1); free(z0); free(z1); free(Hp);
}

/// @brief Evaluate the ELBO objective function
/// @details Evaluate the ELBO objective function for a given ranker problem and observations.
/// If gradient or hessian are not NULL, compute the gradient and/or the hessian. If dobs is not NULL
/// compute the derivative of the parameters with respect to the observations.
/// @param ranker A pointer to the ranker structure
/// @param obs A pointer to the observations structure
/// @param gradient A pointer to the vector where the gradient will be stored
/// @param hessian A pointer to the hessian structure where the hessian will be stored
/// @param dobs A pointer to the vector where the derivative of the parameters with respect to the observations will be stored
/// @return The value of the ELBO objective function
double evaluate(const ranker ranker, const observations obs, gsl_vector * const gradient, hessian * const hessian, gsl_matrix * const dobs) {

    // Set some constants for easy access
    const int n = ranker.n;
    const double ln_α = gsl_vector_get(ranker.params, 2 * n);
    const double ln_β = gsl_vector_get(ranker.params, 2 * n + 1);
    const double α = exp(ln_α);
    const double β = exp(ln_β);
    const double αβ = α * β;
    const double psi_α = gsl_sf_psi(α);
    const double psi1_α = gsl_sf_psi_1(α);
    const double psi2_α = gsl_sf_psi_n(2, α);
    const gsl_vector_const_view μ = gsl_vector_const_subvector(ranker.params, 0, n);
    const gsl_vector_const_view ln_σ = gsl_vector_const_subvector(ranker.params, n, n);

    gsl_vector *σ2 = gsl_vector_calloc(n);

    // Compute sum of μi² + σi², and sum of ln(σi)
    double Σ_μ2_σ2 = 0.0;
    double Σ_lnσ = 0.0;
    for(int i = 0; i < n; ++i) {
        double μi = μ.vector.data[i];
        double lnσi = ln_σ.vector.data[i];
        double σi2 = exp(2 * lnσi);
        σ2->data[i] = σi2;
        double μi2 = μi * μi;
        Σ_μ2_σ2 += μi2 + σi2;
        Σ_lnσ += lnσi;
    }



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
        gsl_blas_daxpy(αβ, &μ.vector, &g_μ.vector);
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
        gsl_vector_memcpy(&αβ_μ.vector, &μ.vector);
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
        double μδ = μ.vector.data[i] - μ.vector.data[j];
        double μδ_over_h = μδ / h;
        double exp_minus_μδ_over_h_square = exp(-μδ_over_h * μδ_over_h);
        double gμδ = - 0.5 * (1 - erf(μδ_over_h));
        double gσδ = - exp_minus_μδ_over_h_square * σδ / sqrt_π_h;

        v += count * (exp_minus_μδ_over_h_square * h / (2 * sqrt(M_PI)) + μδ * gμδ);

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

            // TODO: do not represent obs with a sparse matrix, we do not need to perform
            // algebra on it. That way we can iterate over all matches and now both the number
            // of wins and losses, so we can set the hessian terms once without having to
            // query them to add or subtract. Alternatively, we could use a complex sparse
            // matrix to store the number of wins and losses in the real and imaginary part

            // hμiμj
            double before = gsl_spmatrix_get(hessian->obs, row, col);
            gsl_spmatrix_set(hessian->obs, row, col, before + count * hμμδ);

            // hμiσi, hμjσi
            double chσ = count * hμσδ * σ2->data[i] / σδ;
            before = gsl_spmatrix_get(hessian->obs, i,  n + i);
            gsl_spmatrix_set(hessian->obs, i, n + i, before - chσ);
            before = gsl_spmatrix_get(hessian->obs, j,  n + i);
            gsl_spmatrix_set(hessian->obs, j, n + i, before + chσ);

            // hμiσj, hμjσj
            chσ = count * hμσδ * σ2->data[j] / σδ;
            before = gsl_spmatrix_get(hessian->obs, i,  n + j);
            gsl_spmatrix_set(hessian->obs, i, n + j, before - chσ);
            before = gsl_spmatrix_get(hessian->obs, j, n + j);
            gsl_spmatrix_set(hessian->obs, j, n + j, before + chσ);

            // hσiσj
            before =  gsl_spmatrix_get(hessian->obs, n + row, n + col);
            gsl_spmatrix_set(hessian->obs, n + row, n + col, before - count * (σ2->data[i] * σ2->data[j] / (σδ * σδ) * (hσσδ - gσδ / σδ)));
        }
    }
    return v;
}

double fit(ranker ranker, const observations obs, const double tol, const int max_iter) {

    int k = 0;
    double v = 0;
    double v_prev = 0;
    double v_diff = 0;
    gsl_vector * gradient = gsl_vector_alloc(2 * ranker.n + 2);
    hessian * hessian = hessian_alloc(ranker.n);
    gsl_vector * dx = gsl_vector_alloc(2 * ranker.n + 2);
    gsl_vector * out = gsl_vector_alloc(2 * ranker.n + 2);
    gsl_vector * save_params = gsl_vector_alloc(2 * ranker.n + 2);

    while (k < max_iter) {
        double λ = 1e-10;
        while (true) {
            v = evaluate(ranker, obs, gradient, hessian, NULL);
            // test if the diagonal of the hessian is positive
            bool positive = true;
            for (int i = 0; i < 2 * ranker.n + 2; i++) {
                if (hessian->diag->data[i] <= 0) {
                    positive = false;
                    break;
                }
            }
            if (!positive) {
                printf("ruhroh\n");
            }

            cg(hessian, gradient, dx,  λ, tol, 1000 * ranker.n );
            gsl_vector_memcpy(save_params, ranker.params);
            //gsl_vector_scale(dx, 0.1);
            gsl_vector_sub(ranker.params, dx);
            double v_new = evaluate(ranker, obs, NULL, NULL, NULL);

            printf("k = %d, v_new = %e < v = %e, λ = %e\n", k, v_new, v, λ);
            // print params
            for (int i = 0; i < ranker.n; i++) {
                printf("μ%d = %e, σ%d = %e, ", i, ranker.params->data[i], i, exp(ranker.params->data[ranker.n + i]));
            }
            printf("α = %e, β = %e\n", exp(ranker.params->data[2 * ranker.n]), exp(ranker.params->data[2 * ranker.n + 1]));
            // print what the gradient was
            for (int i = 0; i < ranker.n; i++) {
                printf("gμ%d = %e, gσ%d = %e, ", i, gradient->data[i], i, gradient->data[ranker.n + i]);
            }
            printf("gα = %e, gβ = %e\n\n", gradient->data[2 * ranker.n], gradient->data[2 * ranker.n + 1]);
            if (v_new < v) {

                break;
            } else {

                gsl_vector_memcpy(ranker.params, save_params);
                λ *= 10;
                if (λ > 1e300) {
                    printf("giving up. gradient norm = %e\n", gsl_blas_dnrm2(gradient) );
                    // print params
                    for (int i = 0; i < 2 * ranker.n + 2; i++) {
                        printf("params[%d] = %e\n", i, ranker.params->data[i]);
                    }
                    // pritn hessian using the function we wrote
                    printf("hessian:\n");
                    print_hessian(hessian);


                    exit(-1);
                }
            }
        }
        k += 1;



        // check gradient norm and stop if it is small enough
        if (gsl_blas_dnrm2(gradient) < tol) {
            break;
        }
        // subtract dx[0:n] from ranker.μ, subtract dx[n:2n] from ranker.ln_σ, subtract dx[2n] from ranker.ln_α, subtract dx[2n+1] from ranker.ln_β

    }
    gsl_vector_free(gradient);
    hessian_free(hessian);
    gsl_vector_free(dx);
    return v;
}

int main() {
    int n = 3;
    ranker * ranker = ranker_alloc(n);

    // initialize ranker.μ->data to {0.1, -0.2, 0.3}
    // μ = ranker->params[0:n], create a view, ditto for σ
    gsl_vector_view μ = gsl_vector_subvector(ranker->params, 0, n);
    gsl_vector_view ln_σ = gsl_vector_subvector(ranker->params, n, n);

    // set the first element of μ to 0.1 using the data pointer  and the view μ
    μ.vector.data[0] = 0.1;
    μ.vector.data[1] = -0.2;
    μ.vector.data[2] = 0.3;

    // initialize ranker.ln_σ->data to log({1.1, 2.2, 3.3})
    ln_σ.vector.data[0] = log(1.1);
    ln_σ.vector.data[1] = log(2.2);
    ln_σ.vector.data[2] = log(3.3);

    // initialize ranker.ln_α to log(1.5) and ranker.ln_β to log(3.0)
    gsl_vector_set(ranker->params, 2 * n, log(1.5));
    gsl_vector_set(ranker->params, 2 * n + 1, log(3.0));

    observations obs;
    obs.X = gsl_spmatrix_alloc(1, 1);

    gsl_spmatrix_set(obs.X, 0, 2, 2.0);
    gsl_spmatrix_set(obs.X, 1, 2, 2.0);
    gsl_spmatrix_set(obs.X, 2, 0, 3.0);
    gsl_spmatrix_set(obs.X, 1, 0, 1.0);


    gsl_vector * g = gsl_vector_calloc(2 * n + 2);
    hessian * h = hessian_alloc(n);

    double v = evaluate(*ranker, obs, g, h, NULL);
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
    for(int k = 0; k < h->obs->nz; ++k) {
        printf("obs: (%d, %d) -> %lf\n", h->obs->i[k], h->obs->p[k], h->obs->data[k]);
    }

    // print hessian with dedicated function
    print_hessian(h);



    // cg with Hessian and gradient
    gsl_vector * dx = gsl_vector_calloc(2 * n + 2);
    cg(h, g, dx, 1e-10, 1e-3, 1000);

    // h dot dx
    gsl_vector * hdx = gsl_vector_calloc(2 * n + 2);
    hessian_dot(h, dx, hdx);
    for(int i = 0; i < 2 * n + 2; ++i) {
        printf("h dot dx, d: %lf, %lf\n", hdx->data[i], g->data[i]);
    }

    // print dx
    for(int i = 0; i < 2 * n + 2; ++i) {
        printf("dx: %lf\n", dx->data[i]);
    }

    fit(*ranker, obs, 1e-8, 1000000);

    // print ranker
    for(int i = 0; i < n; ++i) {
        printf("μ: %lf\n", ranker->params->data[i]);
    }
    for(int i = 0; i < n; ++i) {
        printf("σ: %lf\n", exp(ranker->params->data[n+i]));
    }
    printf("α: %lf\n", exp(ranker->params->data[2*n]));
    printf("β: %lf\n", exp(ranker->params->data[2*n+1]));


    gsl_vector_free(hdx);
    gsl_vector_free(dx);
    gsl_spmatrix_free(obs.X);
    gsl_vector_free(g);
    hessian_free(h);
    ranker_free(ranker);



    return 0;

}
