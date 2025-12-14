#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <mpi.h>

const double Lx = 1.0;
const double Ly = 1.0;
const double Lz = 1.0;
const double c1 = 1.0 / (4.0 * M_PI * M_PI);

struct TimingData {
    double total_start, total_end;
    double init_time, compute_time, comm_time, copy_time, kernel_time, time_loop_total, time_error_calc;

    TimingData() : init_time(0), compute_time(0), comm_time(0), copy_time(0), kernel_time(0),
                   time_loop_total(0), time_error_calc(0) {}
};

__device__ __host__ double exact_solution(double x, double y, double z, double t) {
    double at = 0.5 * sqrt(4.0/(Lx*Lx) + 1.0/(Ly*Ly) + 1.0/(Lz*Lz));
    return sin(2.0*M_PI*x/Lx) * sin(M_PI*y/Ly) * sin(M_PI*z/Lz) * cos(at * t + 2.0*M_PI);
}

__global__ void initialize_kernel(double* field, int nx, int ny, int nz,
                                 double dx, double dy, double dz,
                                 int offset_x, int offset_y, int offset_z,
                                 double t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nz;
    if (idx >= total) return;
    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);
    double x = (offset_x + i) * dx;
    double y = (offset_y + j) * dy;
    double z = (offset_z + k) * dz;
    field[idx] = exact_solution(x, y, z, t);
}

__global__ void compute_step_kernel(const double* u0, const double* u1, double* u2,
                                   int nx, int ny, int nz,
                                   double dt2, double dx2, double dy2, double dz2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nz;
    if (idx >= total) return;
    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) {
        u2[idx] = u1[idx]; 
        return;
    }

    double center = u1[idx];
    double left = u1[(i-1) * ny * nz + j * nz + k];
    double right = u1[(i+1) * ny * nz + j * nz + k];
    double down = u1[i * ny * nz + (j-1) * nz + k];
    double up = u1[i * ny * nz + (j+1) * nz + k];
    double back = u1[i * ny * nz + j * nz + (k-1)];
    double front = u1[i * ny * nz + j * nz + (k+1)];

    double laplace = (left - 2.0*center + right)/dx2 + 
                     (down - 2.0*center + up)/dy2 + 
                     (back - 2.0*center + front)/dz2;

    u2[idx] = 2.0 * u1[idx] - u0[idx] + c1 * dt2 * laplace;
}

__global__ void compute_abs_errors_kernel(const double* numerical, double* abs_errors,
                                         int nx, int ny, int nz,
                                         double dx, double dy, double dz,
                                         int offset_x, int offset_y, int offset_z,
                                         double t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nz;
    if (idx >= total) return;
    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);
    double x = (offset_x + i) * dx;
    double y = (offset_y + j) * dy;
    double z = (offset_z + k) * dz;
    abs_errors[idx] = fabs(numerical[idx] - exact_solution(x, y, z, t));
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    TimingData timing;
    timing.total_start = MPI_Wtime();

    int N = 128;
    int K = 50;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) K = atoi(argv[2]);

    if (rank == 0) std::cout << "Wave solver MPI+CUDA\n";

    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No GPU\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int gpu_id = rank % device_count;
    cudaSetDevice(gpu_id);

    int nx_local = N / size;
    int remainder = N % size;
    int start_x = rank * nx_local;
    if (rank < remainder) { nx_local++; start_x += rank; } 
    else { start_x += remainder; }

    double dx = Lx / (N - 1), dy = Ly / (N - 1), dz = Lz / (N - 1);
    double min_d = fmin(dx, fmin(dy, dz));
    double dt = 0.25 * min_d / sqrt(3.0 * c1);
    double dx2 = dx*dx, dy2 = dy*dy, dz2 = dz*dz, dt2 = dt*dt;

    int local_size = nx_local * N * N;
    size_t bytes = local_size * sizeof(double);
    double *d_u0, *d_u1, *d_u2, *d_abs_errors;
    cudaMalloc(&d_u0, bytes); cudaMalloc(&d_u1, bytes);
    cudaMalloc(&d_u2, bytes); cudaMalloc(&d_abs_errors, bytes);

    int block_size = 256;
    int grid_size = (local_size + block_size - 1) / block_size;

    double init_start = MPI_Wtime();
    initialize_kernel<<<grid_size, block_size>>>(d_u0, nx_local, N, N, dx, dy, dz, start_x,0,0,0.0);
    initialize_kernel<<<grid_size, block_size>>>(d_u1, nx_local, N, N, dx, dy, dz, start_x,0,0,dt);
    cudaDeviceSynchronize();
    timing.init_time = MPI_Wtime() - init_start;

    double max_abs_error = 0.0;
    double loop_start = MPI_Wtime();

    for (int step = 0; step < K; step++) {
        double compute_start = MPI_Wtime();
        compute_step_kernel<<<grid_size, block_size>>>(d_u0, d_u1, d_u2, nx_local, N, N, dt2, dx2, dy2, dz2);
        cudaDeviceSynchronize();
        timing.compute_time += MPI_Wtime() - compute_start;

        std::swap(d_u0, d_u1);
        std::swap(d_u1, d_u2);

        if ((step+1)%10 == 0) {
            double error_start = MPI_Wtime();
            compute_abs_errors_kernel<<<grid_size, block_size>>>(d_u1, d_abs_errors, nx_local, N, N, dx, dy, dz, start_x,0,0,(step+1)*dt);
            cudaDeviceSynchronize();

            double copy_start = MPI_Wtime();
            std::vector<double> h_abs_errors(local_size);
            cudaMemcpy(h_abs_errors.data(), d_abs_errors, bytes, cudaMemcpyDeviceToHost);
            timing.copy_time += MPI_Wtime() - copy_start;

            double local_max = *std::max_element(h_abs_errors.begin(), h_abs_errors.end());
            double global_max;
            double comm_start = MPI_Wtime();
            MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            timing.comm_time += MPI_Wtime() - comm_start;

            if (rank==0) std::cout << "Step " << step+1 << ", max error = " << global_max << std::endl;
            max_abs_error = std::max(max_abs_error, global_max);
            timing.time_error_calc += MPI_Wtime() - error_start;
        }
    }

    timing.time_loop_total = MPI_Wtime() - loop_start;
    double final_time = K*dt;

    cudaFree(d_u0); cudaFree(d_u1); cudaFree(d_u2); cudaFree(d_abs_errors);

    timing.total_end = MPI_Wtime();

    if (rank==0) {
        double total_wall_time = timing.total_end - timing.total_start;
        std::cout << "\n=== SUMMARY ===\n";
        std::cout << "Total wall time: " << total_wall_time << " s\n";
        std::cout << "Simulation time: " << final_time << " s\n";
        std::cout << "Max error: " << max_abs_error << "\n";
        std::cout << "\nTiming:\n";
        std::cout << "Initialization: " << timing.init_time << " s\n";
        std::cout << "Compute loop: " << timing.time_loop_total << " s\n";
        std::cout << "Compute time: " << timing.compute_time << " s\n";
        std::cout << "MPI communication: " << timing.comm_time << " s\n";
        std::cout << "Data copy GPU->CPU: " << timing.copy_time << " s\n";
        std::cout << "Error calc time: " << timing.time_error_calc << " s\n";
    }

    MPI_Finalize();
    return 0;
}
