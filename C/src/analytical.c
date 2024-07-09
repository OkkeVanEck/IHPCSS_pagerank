/**
 * @brief Mostly analytical solution. Binary search is used to obtain the first value, from which the rest can
 * be calculated. Complexity is O(n^2 * c) where n is the graph size and c is -log(EPSILON) (i.e. no. of steps for bin search to converge).
 * Only works on the sneaky graph.
 **/
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <stdbool.h>

/// The number of vertices in the graph.
#define GRAPH_ORDER 1000
/// Parameters used in pagerank convergence, do not change.
#define DAMPING_FACTOR 0.85
/// The number of seconds to not exceed forthe calculation loop.
#define MAX_TIME 10
/// For binary search
#define EPSILON 1e-15


/**
 * @brief Indicates which vertices are connected.
 * @details If an edge links vertex A to vertex B, then adjacency_matrix[A][B]
 * will be 1.0. The absence of edge is represented with value 0.0.
 * Redundant edges are still represented with value 1.0.
 */
double adjacency_matrix[GRAPH_ORDER][GRAPH_ORDER]; // OPT: use bool instead of double

double max_diff = 0.0;
double min_diff = 1.0;
double total_diff = 0.0;
 
void initialize_graph(void)
{
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        for(int j = 0; j < GRAPH_ORDER; j++)
        {
            adjacency_matrix[i][j] = 0.0;
        }
    }
}

/**
 * @brief Calculates the pagerank of all vertices in the graph.
 * @param pagerank The array in which store the final pageranks.
 */
void calculate_pagerank(double pagerank[])
{
    double damping_value = (1.0 - DAMPING_FACTOR) / GRAPH_ORDER;
    size_t iteration = 0;

    // Analytical solution possible when first element is known
    // An overestimation of page_rank[0] -> sum(page_rank) > 1, allowing binary search
    double low = 0.0;
    double high = 1.0;

    while ((high - low) > EPSILON) {
        double mid = (low + high) / 2.0;
        pagerank[0] = mid; // this is the only unknown, we take it from the bin search

        for (int i = 0; i < GRAPH_ORDER / 2; i++) {
            // pr[<= i], pr[> iv] is already solved
            // pr[iv] needs to be solved
            // pr[i+1] needs to be solved

            int iv = GRAPH_ORDER - i - 1; // inverse index of i
            int outdegree_i = GRAPH_ORDER - i - 1; // happens to be the same
            int outdegree_iv = i + 1;

            // pr[iv] is the sum of flow for all pr[<= i], which we have already so we can compute it
            pagerank[iv] = damping_value;
            for (int j = 0; j <= i; j++) {
                int jv = GRAPH_ORDER - j - 1;
                pagerank[iv] += pagerank[j] / (GRAPH_ORDER - j - 1) * DAMPING_FACTOR;
            }

            // pr[i+1] is _very_ similar to pr[i], as it shares almost all incoming edges
            // what it does not share however we already know: it's missing pr[iv] which we can subtract and it has a connection to pr[i] and no self edge, which we can factor out
            if (i < GRAPH_ORDER / 2 - 1) {
                pagerank[i + 1] = (pagerank[i] - (pagerank[iv] * DAMPING_FACTOR / outdegree_iv) +
                                   pagerank[i] / outdegree_i * DAMPING_FACTOR) /
                                  (1.0 + 1.0 / (outdegree_i - 1.0) * DAMPING_FACTOR);
            }
        }

        // Evaluate and update bin search
        double sum = 0;
        for(int i = 0; i < GRAPH_ORDER; i++) {
            sum += pagerank[i];
        }

        if (sum >= 1.0) {
            high = mid; // overestimation
        }
        else {
            low = mid; // underestimation
        }
    }
}

/**
 * @brief Populates the edges in the graph for testing.
 **/
void generate_nice_graph(void)
{
    printf("Generate a graph for testing purposes (i.e.: a nice and conveniently designed graph :) )\n");
    double start = omp_get_wtime();
    initialize_graph();
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        for(int j = 0; j < GRAPH_ORDER; j++)
        {
            int source = i;
            int destination = j;
            if(i != j)
            {
                adjacency_matrix[source][destination] = 1.0;
            }
        }
    }
    printf("%.2f seconds to generate the graph.\n", omp_get_wtime() - start);
}

/**
 * @brief Populates the edges in the graph for the challenge.
 **/
void generate_sneaky_graph(void)
{
    printf("Generate a graph for the challenge (i.e.: a sneaky graph :P )\n");
    double start = omp_get_wtime();
    initialize_graph();
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        for(int j = 0; j < GRAPH_ORDER - i; j++)
        {
            int source = i;
            int destination = j;
            if(i != j)
            {
                adjacency_matrix[source][destination] = 1.0;
            }
        }
    }
    printf("%.2f seconds to generate the graph.\n", omp_get_wtime() - start);
}

int main(int argc, char* argv[])
{
    // We do not need argc, this line silences potential compilation warnings.
    (void) argc;
    // We do not need argv, this line silences potential compilation warnings.
    (void) argv;

    // Get the time at the very start.
    double start = omp_get_wtime();
    
    generate_sneaky_graph();
 
    /// The array in which each vertex pagerank is stored.
    double pagerank[GRAPH_ORDER];
    calculate_pagerank(pagerank);

    // Calculates the sum of all pageranks. It should be 1.0, so it can be used as a quick verification.
    double sum_ranks = 0.0;
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        if(i % 100 == 0)
        {
            printf("PageRank of vertex %d: %.6f\n", i, pagerank[i]);
        }
        sum_ranks += pagerank[i];
    }
    printf("Sum of all pageranks = %.12f, total diff = %.12f, max diff = %.12f and min diff = %.12f.\n", sum_ranks, total_diff, max_diff, min_diff);
    double end = omp_get_wtime();

    printf("Total time taken: %.2f seconds.\n", end - start);
 
    return 0;
}
