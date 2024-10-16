#include <mpi.h>
#include <stdio.h>
#include <math.h>

// Function to evaluate the curve (y = g(x))
float g(float x) {
    return x * x; // Example: y = x^2
}

// Function to calculate the area of a segment using the trapezoidal rule
float compute_segment_area(float left, float right, float d) { 
    float segment_area = 0;
    for (float x = left; x < right; x += d) {
        segment_area += g(x) + g(x + d); 
    }
    return segment_area * d / 2.0f;
}

int main(int argc, char** argv) {
    int process_id, total_processes;
    float lower_bound = 0.0f, upper_bound = 1.0f;  // Limits of integration
    int intervals;
    float local_start, local_end, local_segment_area, total_integral;

  
    intervals = 10000000;

    double sequential_start_time, sequential_end_time, sequential_duration;
    double sequential_result = 0;
    float delta_x = (upper_bound - lower_bound) / intervals; // Width of each small interval

    //Parallel
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id); // Get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes); // Get total number of processes

    double parallel_start_time, parallel_end_time, parallel_duration;

    if (process_id == 0) {
        // Output the number of intervals being used
        printf("Using %d intervals for the computation.\n", intervals);

        // Start sequential computation
        sequential_start_time = MPI_Wtime();
        sequential_result = compute_segment_area(lower_bound, upper_bound, delta_x);
        sequential_end_time = MPI_Wtime();
        sequential_duration = sequential_end_time - sequential_start_time;

        // Display sequential results
        printf("Total area under the curve: %f\n", sequential_result);
        printf("Time taken: %f seconds\n", sequential_duration);
    }

    // Start parallel computation
    parallel_start_time = MPI_Wtime();

    // Broadcast the number of intervals to all processes
    MPI_Bcast(&intervals, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the size of the region assigned to each process
    float sub_interval_width = (upper_bound - lower_bound) / total_processes;

    // Compute local bounds for the current process
    local_start = lower_bound + process_id * sub_interval_width;
    local_end = local_start + sub_interval_width;

    // Each process calculates the area of its assigned segment
    local_segment_area = compute_segment_area(local_start, local_end, delta_x);

    // Reduce all local areas to get the total integral
    MPI_Reduce(&local_segment_area, &total_integral, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    parallel_end_time = MPI_Wtime();
    parallel_duration = parallel_end_time - parallel_start_time;

    if (process_id == 0) {
        // Display parallel results
        printf("Total area under the curve: %f\n", total_integral);
        printf("Time taken: %f seconds\n", parallel_duration);

        float speedup = sequential_duration / parallel_duration;
        float efficiency = (speedup / total_processes) * 100;

        printf("Speedup factor: %f\n", speedup);
        printf("Efficiency: %f%%\n", efficiency);
    }

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}
