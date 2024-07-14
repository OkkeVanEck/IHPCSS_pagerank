/**
 * @brief Implements all the basic optimizations while still doing iterations, without intrinsics
 **/

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cub/block/block_reduce.cuh>
#include <iostream>
#include <memory>

/// The number of vertices in the graph.
constexpr int GRAPH_ORDER = 1000;
/// Parameters used in pagerank convergence, do not change.
constexpr double DAMPING_FACTOR = 0.85;
/// The number of seconds to not exceed forthe calculation loop.
constexpr int MAX_TIME = 10;
// For comparison with others
// constexpr int MAX_ITERATIONS = 1'000'000'000;
constexpr int MAX_ITERATIONS = 1;

// Define a custom deleter for cudaFree
struct CudaFreeDeleter
{
  void operator()(double *ptr) const { cudaFree(ptr); }
};

// Error checking macro
#define CHECK_CUDA(call)                                                                                                \
  do                                                                                                                    \
    {                                                                                                                   \
      cudaError_t err = call;                                                                                           \
      if(err != cudaSuccess)                                                                                            \
        {                                                                                                               \
          std::cerr << "CUDA error in " << __FILE__ << "@" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
          std::exit(EXIT_FAILURE);                                                                                      \
        }                                                                                                               \
    }                                                                                                                   \
  while(0)

#define CHECK_CUBLAS(call)                                                                               \
  do                                                                                                     \
    {                                                                                                    \
      cublasStatus_t status = call;                                                                      \
      if(status != CUBLAS_STATUS_SUCCESS)                                                                \
        {                                                                                                \
          std::cerr << "cuBLAS error in " << __FILE__ << "@" << __LINE__ << ": " << status << std::endl; \
          std::exit(EXIT_FAILURE);                                                                       \
        }                                                                                                \
    }                                                                                                    \
  while(0)

template <typename T>
__global__ void initializeArrayValue(T *arr, const T value, const int size)
{
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  if(idx < size)
    {
      arr[idx] = value;
    }
}

constexpr int THREADS_PER_BLOCK = 256;
/**
 * @brief Indicates which vertices are connected.
 * @details If an edge links vertex A to vertex B, then adjacency_matrix[A][B]
 * will be 1.0. The absence of edge is represented with value 0.0.
 * Redundant edges are still represented with value 1.0.
 */
double adjacency_matrix[GRAPH_ORDER][GRAPH_ORDER];
double max_diff   = 0.0;
double min_diff   = 1.0;
double total_diff = 0.0;

// CUDA kernel for element-wise absolute difference
__global__ void elementWiseFabs(const double *a, const double *b, double *c, const int n)
{
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n)
    {
      c[idx] = fabs(a[idx] - b[idx]);
    }
}

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
  constexpr int blocksPerGrid = (GRAPH_ORDER + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  // Initialise all vertices to 1/n.
  double initial_rank  = 1.0 / GRAPH_ORDER;
  double damping_value = (1.0 - DAMPING_FACTOR) / GRAPH_ORDER;

  double diff               = 1.0;
  size_t iteration          = 0;
  double time_per_iteration = 0;

  // Convert the graph representation to a transition matrix
  // This removes a memory access, a division and a branch from the main loop, and a whole second loop to apply damping
  double transition_matrix[GRAPH_ORDER][GRAPH_ORDER];

  for(int i = 0; i < GRAPH_ORDER; i++)
    {
      for(int j = 0; j < GRAPH_ORDER; j++)
        {
          double val;
          if(adjacency_matrix[i][j])
            {
              int outdegree = 0;
              for(int k = 0; k < GRAPH_ORDER; k++)
                {
                  if(adjacency_matrix[j][k])
                    outdegree++;
                }
              val = DAMPING_FACTOR / outdegree;
            }
          else
            {
              val = 0;
            }
          transition_matrix[i][j] = val;
        }
    }

  int minGridSize = -1;
  int blockSize   = -1;

  // Calculate the best block size and minimum grid size for maximum occupancy
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, initializeArrayValue<double>, 0, GRAPH_ORDER);
  printf("Best block size for initializeArrayValue<double>: %d, minimum grid size: %d\n", blockSize, minGridSize);

  // Calculate grid size to cover the whole data set
  int gridSize = (GRAPH_ORDER + blockSize - 1) / blockSize;

  // Use unique_ptr for device memory management
  std::unique_ptr<double[], CudaFreeDeleter> d_pagerank(nullptr);
  std::unique_ptr<double[], CudaFreeDeleter> d_new_pagerank(nullptr);
  std::unique_ptr<double[], CudaFreeDeleter> d_transition_matrix(nullptr);
  std::unique_ptr<double[], CudaFreeDeleter> d_diff(nullptr);
  {
    double *ptr;
    CHECK_CUDA(cudaMalloc((void **)&ptr, GRAPH_ORDER * sizeof(double)));
    d_pagerank.reset(ptr);

    CHECK_CUDA(cudaMalloc((void **)&ptr, GRAPH_ORDER * sizeof(double)));
    d_new_pagerank.reset(ptr);

    CHECK_CUDA(cudaMalloc((void **)&ptr, GRAPH_ORDER * GRAPH_ORDER * sizeof(double)));
    d_transition_matrix.reset(ptr);

    CHECK_CUDA(cudaMalloc((void **)&ptr, GRAPH_ORDER * sizeof(double)));
    d_diff.reset(ptr);
  }

  CHECK_CUDA(cudaMemcpy(d_pagerank.get(), pagerank, GRAPH_ORDER * sizeof(double), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_new_pagerank.get(), 0, GRAPH_ORDER * sizeof(double)));
  CHECK_CUDA(
      cudaMemcpy(d_transition_matrix.get(), transition_matrix, GRAPH_ORDER * GRAPH_ORDER * sizeof(double), cudaMemcpyHostToDevice));

  initializeArrayValue<double><<<gridSize, blockSize>>>(d_pagerank.get(), initial_rank, GRAPH_ORDER);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  double start   = omp_get_wtime();
  double elapsed = 0;

  // cuBLAS handle
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  // If we exceeded the MAX_TIME seconds, we stop. If we typically spend X seconds on an iteration, and we are less than X seconds away
  // from MAX_TIME, we stop.
  while(elapsed < MAX_TIME && (elapsed + time_per_iteration) < MAX_TIME && iteration < MAX_ITERATIONS)
    {
      double iteration_start = omp_get_wtime();

      initializeArrayValue<double><<<gridSize, blockSize>>>(d_new_pagerank.get(), 1., GRAPH_ORDER);
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_CUDA(cudaGetLastError());

      // Perform the matrix-vector multiplication: y = A * x
      constexpr double alpha = 1.0;
      CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N, GRAPH_ORDER, GRAPH_ORDER, &alpha, d_transition_matrix.get(), GRAPH_ORDER,
                               d_pagerank.get(), 1, &damping_value, d_new_pagerank.get(), 1));

      diff = 0.0;
      elementWiseFabs<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_new_pagerank.get(), d_pagerank.get(), d_diff.get(), GRAPH_ORDER);
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUBLAS(cublasDasum(handle, GRAPH_ORDER, d_diff.get(), 1, &diff));

      max_diff = (max_diff < diff) ? diff : max_diff;
      total_diff += diff;
      min_diff = (min_diff > diff) ? diff : min_diff;

      // Swap the buffers
      d_pagerank.swap(d_new_pagerank);

      double pagerank_total = 0.0;

      CHECK_CUBLAS(cublasDasum(handle, GRAPH_ORDER, d_pagerank.get(), 1, &pagerank_total));
      if(fabs(pagerank_total - 1.0) >= 1E-12)
        {
          printf("[ERROR] Iteration %zu: sum of all pageranks is not 1 but %.12f.\n", iteration, pagerank_total);
        }

      double iteration_end = omp_get_wtime();
      elapsed              = omp_get_wtime() - start;
      iteration++;
      time_per_iteration = iteration_end - iteration_start;
    }

  CHECK_CUDA(cudaMemcpy(pagerank, d_pagerank.get(), GRAPH_ORDER * sizeof(double), cudaMemcpyDeviceToHost));

  CHECK_CUBLAS(cublasDestroy(handle));

  printf("%zu iterations achieved in %.2f seconds\n", iteration, elapsed);
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
          int source      = i;
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
          int source      = i;
          int destination = j;
          if(i != j)
            {
              adjacency_matrix[source][destination] = 1.0;
            }
        }
    }
  printf("%.2f seconds to generate the graph.\n", omp_get_wtime() - start);
}

int main(int argc, char *argv[])
{
  // We do not need argc, this line silences potential compilation warnings.
  (void)argc;
  // We do not need argv, this line silences potential compilation warnings.
  (void)argv;

  printf(
      "This program has two graph generators: generate_nice_graph and generate_sneaky_graph. If you intend to submit, your code will "
      "be timed on the sneaky graph, remember to try both.\n");

  // Get the time at the very start.
  double start = omp_get_wtime();

  generate_sneaky_graph();

  /// The array in which each vertex pagerank is stored. Make sure its aligned for simd
  auto pagerank = std::make_unique<double[]>(GRAPH_ORDER);

  for(int i = 0; i < GRAPH_ORDER; i++)
    {
      pagerank[i] = 0;
    }

  calculate_pagerank(pagerank.get());

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
  printf("Sum of all pageranks = %.12f, total diff = %.12f, max diff = %.12f and min diff = %.12f.\n", sum_ranks, total_diff, max_diff,
         min_diff);
  double end = omp_get_wtime();

  printf("Total time taken: %.2f seconds.\n", end - start);

  return 0;
}
