/**
 * @brief Implements all the basic optimizations while still doing iterations
 **/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <mpi.h>

#include <immintrin.h>

/// The number of vertices in the graph.
#define GRAPH_ORDER 1000
/// Parameters used in pagerank convergence, do not change.
#define DAMPING_FACTOR 0.85
/// The number of seconds to not exceed forthe calculation loop.
#define MAX_TIME 10

// For comparison with others
#define MAX_ITERATIONS 999999999999

#ifdef AVX512
    #define VECCALL(fn) _mm512_ ## fn
    #define VECDTYPE __m512d
#else
    #define VECCALL(fn) _mm256_ ## fn
    #define VECDTYPE __m256d
#endif

#define VECWIDTH (sizeof( VECDTYPE ) / sizeof(double))

/**
 * @brief Indicates which vertices are connected.
 * @details If an edge links vertex A to vertex B, then adjacency_matrix[A][B]
 * will be 1.0. The absence of edge is represented with value 0.0.
 * Redundant edges are still represented with value 1.0.
 */
double adjacency_matrix[GRAPH_ORDER][GRAPH_ORDER];
double max_diff = 0.0;
double min_diff = 1.0;
double total_diff = 0.0;

void initialize_graph(void) {
    for (int i = 0; i < GRAPH_ORDER; i++) {
        for (int j = 0; j < GRAPH_ORDER; j++) {
            adjacency_matrix[i][j] = 0.0;
        }
    }
}

/**
 * @brief Calculates the pagerank of all vertices in the graph.
 * @param pagerank The array in which store the final pageranks.
 */
void calculate_pagerank(double pagerank[]) {
    // Initialise all vertices to 1/n.
    VECDTYPE initial_rank = VECCALL(set1_pd)(1.0 / GRAPH_ORDER);
#ifdef AVX512
    VECDTYPE damping_value = _mm512_set_pd(0, 0, 0, 0, 0, 0, 0, (1.0 - DAMPING_FACTOR) / GRAPH_ORDER);
#else
    VECDTYPE damping_value = _mm256_set_pd(0, 0, 0, (1.0 - DAMPING_FACTOR) / GRAPH_ORDER);
#endif

    for (int i = 0; i < GRAPH_ORDER; i += VECWIDTH) {
        VECCALL(store_pd)(&pagerank[i], initial_rank);
    }
    
    double diff = 1.0;
    size_t iteration = 0;
    double time_per_iteration = 0;

    double *new_pagerank;
    posix_memalign((void**)&new_pagerank, 32, GRAPH_ORDER * sizeof(double));

    // Convert the graph representation to a transition matrix
    // This removes a memory access, a division and a branch from the main loop, and a whole second loop to apply damping
    VECDTYPE (*transition_matrix)[GRAPH_ORDER][GRAPH_ORDER/VECWIDTH];
    posix_memalign((void**)&transition_matrix, 32, GRAPH_ORDER * GRAPH_ORDER * sizeof(double));

    for (int i = 0; i < GRAPH_ORDER; i++) {
        for (int j = 0; j < GRAPH_ORDER; j+=VECWIDTH) {
            // Glue 4 values together for vectorization
            // This isn't strictly necessary as transition_matrix is still GRAPH_ORDER * GRAPH_ORDER * sizeof(double) but makes it a bit easier to work with
            double vector[VECWIDTH];
            for (int offset = 0; offset < VECWIDTH; offset++) {
                if (adjacency_matrix[j + offset][i]) {
                    // This loop can be lifted
                    int outdegree = 0;
                    for (int k = 0; k < GRAPH_ORDER; k++) {
                        if (adjacency_matrix[j+offset][k])
                            outdegree++;
                    }
                    vector[offset] = DAMPING_FACTOR / outdegree;
                } else {
                    vector[offset] = 0;
                }
            }

            (*transition_matrix)[i][j/VECWIDTH] = VECCALL(loadu_pd)(vector);
        }
    }

    double start = omp_get_wtime();
    double elapsed = 0;

    // If we exceeded the MAX_TIME seconds, we stop. If we typically spend X seconds on an iteration, and we are less than X seconds away from MAX_TIME, we stop.
    while (elapsed < MAX_TIME && (elapsed + time_per_iteration) < MAX_TIME && iteration < MAX_ITERATIONS) {
        double iteration_start = omp_get_wtime();

#pragma omp parallel for schedule(static)
        for (int i = 0; i < GRAPH_ORDER; i++) {
            VECDTYPE sum = damping_value;

            for (int j = 0; j < GRAPH_ORDER; j+=VECWIDTH) {
                VECDTYPE pr = VECCALL(loadu_pd)(&pagerank[j]); // reads pagerank[j,...j+3]

                // adj0 can be stored directly in the transition matrix
                VECDTYPE adj0 = (*transition_matrix)[i][j/VECWIDTH];
                VECDTYPE mul = VECCALL(mul_pd)(pr, adj0);
                sum = VECCALL(add_pd)(sum, mul);
            }

            // Horizontally sum the 4 elements in the AVX register
            // Ew but idk a better way, can probably be optimized
            double temp[VECWIDTH];
            VECCALL(storeu_pd)(temp, sum);
#ifdef AVX512
            new_pagerank[i] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
#else
            new_pagerank[i] = temp[0] + temp[1] + temp[2] + temp[3];
#endif
        }

        // TODO vectorize or remove
        diff = 0.0;
        for (int i = 0; i < GRAPH_ORDER; i++) {
            diff += fabs(new_pagerank[i] - pagerank[i]);
        }
        max_diff = (max_diff < diff) ? diff : max_diff;
        total_diff += diff;
        min_diff = (min_diff > diff) ? diff : min_diff;

        // Swap the buffers
        {
            double *tmp = pagerank;
            pagerank = new_pagerank;
            new_pagerank = tmp;
        }

        // TODO vectorize or just remove
        double pagerank_total = 0.0;
        for (int i = 0; i < GRAPH_ORDER; i++) {
            pagerank_total += pagerank[i];
        }
        if (fabs(pagerank_total - 1.0) >= 1E-12) {
            printf("[ERROR] Iteration %zu: sum of all pageranks is not 1 but %.12f.\n", iteration, pagerank_total);
        }

        double iteration_end = omp_get_wtime();
        elapsed = omp_get_wtime() - start;
        iteration++;
        time_per_iteration = iteration_end - iteration_start;
    }

    printf("%zu iterations achieved in %.2f seconds\n", iteration, elapsed);
}

/**
 * @brief Populates the edges in the graph for testing.
 **/
void generate_nice_graph(void) {
    printf("Generate a graph for testing purposes (i.e.: a nice and conveniently designed graph :) )\n");
    double start = omp_get_wtime();
    initialize_graph();
    for (int i = 0; i < GRAPH_ORDER; i++) {
        for (int j = 0; j < GRAPH_ORDER; j++) {
            int source = i;
            int destination = j;
            if (i != j) {
                adjacency_matrix[source][destination] = 1.0;
            }
        }
    }
    printf("%.2f seconds to generate the graph.\n", omp_get_wtime() - start);
}

/**
 * @brief Populates the edges in the graph for the challenge.
 **/
void generate_sneaky_graph(void) {
    printf("Generate a graph for the challenge (i.e.: a sneaky graph :P )\n");
    double start = omp_get_wtime();
    initialize_graph();
    for (int i = 0; i < GRAPH_ORDER; i++) {
        for (int j = 0; j < GRAPH_ORDER - i; j++) {
            int source = i;
            int destination = j;
            if (i != j) {
                adjacency_matrix[source][destination] = 1.0;
            }
        }
    }
    printf("%.2f seconds to generate the graph.\n", omp_get_wtime() - start);
}

int main(int argc, char *argv[]) {
    // We do not need argc, this line silences potential compilation warnings.
    (void) argc;
    // We do not need argv, this line silences potential compilation warnings.
    (void) argv;

    printf("This program has two graph generators: generate_nice_graph and generate_sneaky_graph. If you intend to submit, your code will be timed on the sneaky graph, remember to try both.\n");

    // Get the time at the very start.
    double start = omp_get_wtime();

    generate_sneaky_graph();

    /// The array in which each vertex pagerank is stored. Make sure its aligned for simd
    double *pagerank;
    posix_memalign((void**)&pagerank, 32, GRAPH_ORDER * sizeof(double));

    calculate_pagerank(pagerank);

    // Calculates the sum of all pageranks. It should be 1.0, so it can be used as a quick verification.
    double sum_ranks = 0.0;
    for (int i = 0; i < GRAPH_ORDER; i++) {
        if (i % 100 == 0) {
            printf("PageRank of vertex %d: %.6f\n", i, pagerank[i]);
        }
        sum_ranks += pagerank[i];
    }
    printf("Sum of all pageranks = %.12f, total diff = %.12f, max diff = %.12f and min diff = %.12f.\n", sum_ranks,
           total_diff, max_diff, min_diff);
    double end = omp_get_wtime();

    printf("Total time taken: %.2f seconds.\n", end - start);

    return 0;
}
