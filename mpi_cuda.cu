#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <mpi.h>

const double Lx = 1.0;
const double Ly = 1.0;
const double Lz = 1.0;
const double c1 = 1.0 / (4.0 * M_PI * M_PI);
const int HALO = 1;

__constant__ double c_dx, c_dy, c_dz;
__constant__ double c_dx2, c_dy2, c_dz2, c_dt2;
__constant__ double c_c1;
__constant__ int c_nx, c_ny, c_nz;
__constant__ int c_offset_x;

double exact_solution_host(double x, double y, double z, double t) {
    double at = 0.5 * sqrt(4.0/(Lx*Lx) + 1.0/(Ly*Ly) + 1.0/(Lz*Lz));

    return sin(2.0*M_PI*x/Lx) * sin(M_PI*y/Ly) * sin(M_PI*z/Lz) * cos(at * t + 2.0*M_PI);
}

__device__ __host__ double exact_solution(double x, double y, double z, double t) 
{
    double at = 0.5 * sqrt(4.0/(Lx*Lx) + 1.0/(Ly*Ly) + 1.0/(Lz*Lz));

    return sin(2.0*M_PI*x/Lx) * sin(M_PI*y/Ly) * sin(M_PI*z/Lz) * cos(at * t + 2.0*M_PI);
}

__global__ void initialize_kernel(double* field, double t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = c_nx * c_ny * c_nz;
    if (idx >= total) return;
    

    int k = idx % c_nz;
    int j = (idx / c_nz) % c_ny;
    int i = idx / (c_ny * c_nz);
    
    double x = (c_offset_x + i) * c_dx;
    double y = j * c_dy;
    double z = k * c_dz;
    field[idx] = exact_solution(x, y, z, t);
}

__global__ void compute_step_kernel(const double* u0, const double* u1, double* u2) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = c_nx * c_ny * c_nz;
    if (idx >= total) return;
    
    int k = idx % c_nz;
    int j = (idx / c_nz) % c_ny;
    int i = idx / (c_ny * c_nz);


    if (i < HALO || i >= c_nx - HALO || j < HALO || j >= c_ny - HALO || k < HALO || k >= c_nz - HALO) {
        u2[idx] = u1[idx];
        return;
    }

    int idx_center = idx;
    int idx_left   = (i-1) * c_ny * c_nz + j * c_nz + k;
    int idx_right  = (i+1) * c_ny * c_nz + j * c_nz + k;

    int idx_down   = i * c_ny * c_nz + (j-1) * c_nz + k;
    int idx_up     = i * c_ny * c_nz + (j+1) * c_nz + k;

    int idx_back   = i * c_ny * c_nz + j * c_nz + (k-1);
    int idx_front  = i * c_ny * c_nz + j * c_nz + (k+1);

    if (idx_left < 0 || idx_left >= total || idx_right < 0 || idx_right >= total || idx_down < 0 || idx_down >= total || idx_up < 0 || idx_up >= total ||
        idx_back < 0 || idx_back >= total || idx_front < 0 || idx_front >= total) 
        {
        u2[idx] = u1[idx];
        return;
    }

    double center = u1[idx_center];
    double left = u1[idx_left];
    double right = u1[idx_right];
    double down = u1[idx_down];
    double up = u1[idx_up];
    double back = u1[idx_back];
    double front = u1[idx_front];

    double laplace = (left - 2.0*center + right)/c_dx2 + (down - 2.0*center + up)/c_dy2 +  (back - 2.0*center + front)/c_dz2;

    u2[idx] = 2.0 * u1[idx_center] - u0[idx_center] + c_c1 * c_dt2 * laplace;
}

__global__ void compute_abs_errors_kernel(const double* numerical, double* abs_errors, double t) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = c_nx * c_ny * c_nz;
    if (idx >= total) return;
    
    int k = idx % c_nz;

    int j = (idx / c_nz) % c_ny;
    int i = idx / (c_ny * c_nz);

    
    double x = (c_offset_x + i) * c_dx;
    double y = j * c_dy;
    double z = k * c_dz;
    abs_errors[idx] = fabs(numerical[idx] - exact_solution(x, y, z, t));
}

__global__ void pack_x_halo_kernel(const double* field, double* buffer, int side) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (j >= c_ny || k >= c_nz) return;
    
    int src_i = (side == 0) ? HALO : (c_nx - 1 - HALO);

    int src_idx = src_i * c_ny * c_nz + j * c_nz + k;
    int buf_idx = j * c_nz + k;

    
    buffer[buf_idx] = field[src_idx];
}

__global__ void unpack_x_halo_kernel(double* field, const double* buffer, int side) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (j >= c_ny || k >= c_nz) return;
    
    int dst_i = (side == 0) ? 0 : (c_nx - 1);
    int dst_idx = dst_i * c_ny * c_nz + j * c_nz + k;
    int buf_idx = j * c_nz + k;
    
    field[dst_idx] = buffer[buf_idx];
}

__global__ void pairwise_max_kernel(double* array, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= stride) return;

    double val1 = array[idx];
    double val2 = array[stride + idx];
    array[idx] = fmax(val1, val2);
}

double find_max_device(double* d_array, int size) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) 
    {
        int threads = std::min(1024, stride);

           int blocks = (stride + threads - 1) / threads;
        pairwise_max_kernel<<<blocks, threads>>>(d_array, stride);
        cudaDeviceSynchronize();
    }
    double max_value;
    cudaMemcpy(&max_value, d_array, sizeof(double), cudaMemcpyDeviceToHost);
    return max_value;
}

void setup_constants(int nx, int ny, int nz, int offset_x,
                     double dx, double dy, double dz,
                     double dx2, double dy2, double dz2, double dt2) {
    cudaMemcpyToSymbol(c_dx, &dx, sizeof(double));
    cudaMemcpyToSymbol(c_dy, &dy, sizeof(double));
    cudaMemcpyToSymbol(c_dz, &dz, sizeof(double));

    cudaMemcpyToSymbol(c_dx2, &dx2, sizeof(double));
    cudaMemcpyToSymbol(c_dy2, &dy2, sizeof(double));
    cudaMemcpyToSymbol(c_dz2, &dz2, sizeof(double));
    cudaMemcpyToSymbol(c_dt2, &dt2, sizeof(double));

    cudaMemcpyToSymbol(c_c1, &c1, sizeof(double));
       cudaMemcpyToSymbol(c_nx, &nx, sizeof(int));
    cudaMemcpyToSymbol(c_ny, &ny, sizeof(int));
 cudaMemcpyToSymbol(c_nz, &nz, sizeof(int));
    cudaMemcpyToSymbol(c_offset_x, &offset_x, sizeof(int));
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    double total_start = MPI_Wtime();
    double init_start, init_end;
      double loop_start, loop_end;

    double compute_time = 0, comm_time = 0, copy_time = 0, error_time = 0;
    double final_start, final_end;
    
    int N = 512;
    int K = 100;
    
    if (argc > 1) N = atoi(argv[1]);

    if (argc > 2) K = atoi(argv[2]);
    
    cudaSetDevice(rank);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, rank);
    
    int nx_local = N / size;
    int remainder = N % size;
    int start_x = rank * nx_local;
    
    if (rank < remainder) 
    { 
        nx_local++; 
        start_x += rank; 

    } else { 
        start_x += remainder; 
    }
    
    int nx_with_halos = nx_local + 2 * HALO;
    int start_with_halo = start_x - HALO;
    
    if (rank == 0) start_with_halo = 0;
       if (rank == size - 1 && start_x + nx_local >= N) 
       
       {
        nx_with_halos = nx_local + HALO;
    }
    
    double dx = Lx / (N - 1), dy = Ly / (N - 1), dz = Lz / (N - 1);
    double min_d = fmin(dx, fmin(dy, dz));
     double dt = 0.25 * min_d / sqrt(3.0 * c1);
    double dx2 = dx*dx, dy2 = dy*dy, dz2 = dz*dz, dt2 = dt*dt;
    
    setup_constants(nx_with_halos, N, N, start_with_halo, 
                    dx, dy, dz, dx2, dy2, dz2, dt2);
    
    size_t local_size = nx_with_halos * N * N;

    size_t bytes = local_size * sizeof(double);
    
    double *d_u0, *d_u1, *d_u2, *d_abs_errors;
    cudaMalloc(&d_u0, bytes);
      cudaMalloc(&d_u1, bytes);

cudaMalloc(&d_u2, bytes);
    cudaMalloc(&d_abs_errors, bytes);
    
    size_t halo_bytes = N * N * sizeof(double);
     double *h_send_left, *h_recv_left, *h_send_right, *h_recv_right;
    double *d_send_left, *d_recv_left, *d_send_right, *d_recv_right;
    
    cudaMallocHost(&h_send_left, halo_bytes);
    cudaMallocHost(&h_recv_left, halo_bytes);
       cudaMallocHost(&h_send_right, halo_bytes);
    cudaMallocHost(&h_recv_right, halo_bytes);
    
    cudaMalloc(&d_send_left, halo_bytes);
    cudaMalloc(&d_recv_left, halo_bytes);

    
      cudaMalloc(&d_send_right, halo_bytes);
    cudaMalloc(&d_recv_right, halo_bytes);
    
    int block_size = 256;
    int grid_size = (local_size + block_size - 1) / block_size;
    
    dim3 block_2d(16, 16);
    dim3 grid_2d((N + block_2d.x - 1) / block_2d.x,
                 (N + block_2d.y - 1) / block_2d.y);
    
    init_start = MPI_Wtime();
    initialize_kernel<<<grid_size, block_size>>>(d_u0, 0.0);
      initialize_kernel<<<grid_size, block_size>>>(d_u1, dt);
       cudaDeviceSynchronize();
    init_end = MPI_Wtime();
    
    double max_abs_error = 0.0;
    loop_start = MPI_Wtime();
    
    int left_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
       int right_neighbor = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
    
    for (int step = 0; step < K; step++) 
    {
        double compute_start = MPI_Wtime();
     compute_step_kernel<<<grid_size, block_size>>>(d_u0, d_u1, d_u2);
        cudaDeviceSynchronize();

        compute_time += MPI_Wtime() - compute_start;
        
        if (size > 1) {
        
            double comm_start = MPI_Wtime();
            MPI_Request requests[4];
            int count = 0;
            
            if (left_neighbor != MPI_PROC_NULL) 
            {
            pack_x_halo_kernel<<<grid_2d, block_2d>>>(d_u1, d_send_left, 0);
                cudaMemcpy(h_send_left, d_send_left, halo_bytes, cudaMemcpyDeviceToHost);
         MPI_Isend(h_send_left, N*N, MPI_DOUBLE, left_neighbor, 0, MPI_COMM_WORLD, &requests[count++]);
            }
            
      if (right_neighbor != MPI_PROC_NULL) 
         {
                pack_x_halo_kernel<<<grid_2d, block_2d>>>(d_u1, d_send_right, 1);
             cudaMemcpy(h_send_right, d_send_right, halo_bytes, cudaMemcpyDeviceToHost);
                MPI_Isend(h_send_right, N*N, MPI_DOUBLE, right_neighbor, 1, MPI_COMM_WORLD, &requests[count++]);
            }
            
            if (left_neighbor != MPI_PROC_NULL)
             {

                   MPI_Irecv(h_recv_left, N*N, MPI_DOUBLE, left_neighbor, 1, MPI_COMM_WORLD, &requests[count++]);
            }

            
            if (right_neighbor != MPI_PROC_NULL) 
            {
                
                
                MPI_Irecv(h_recv_right, N*N, MPI_DOUBLE, right_neighbor, 0, MPI_COMM_WORLD, &requests[count++]);
            }

            
            MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
            


            if (left_neighbor != MPI_PROC_NULL) 
            {
                cudaMemcpy(d_recv_left, h_recv_left, halo_bytes, cudaMemcpyHostToDevice);
                unpack_x_halo_kernel<<<grid_2d, block_2d>>>(d_u1, d_recv_left, 0);
            }
            
            if (right_neighbor != MPI_PROC_NULL) 
            {
                cudaMemcpy(d_recv_right, h_recv_right, halo_bytes, cudaMemcpyHostToDevice);
                unpack_x_halo_kernel<<<grid_2d, block_2d>>>(d_u1, d_recv_right, 1);
            }
            
            cudaDeviceSynchronize();
            comm_time += MPI_Wtime() - comm_start;
        }
        
        std::swap(d_u0, d_u1);
        std::swap(d_u1, d_u2);
        
        if ((step + 1) % 10 == 0) {
            double error_start = MPI_Wtime();
            compute_abs_errors_kernel<<<grid_size, block_size>>>(d_u1, d_abs_errors, (step + 1) * dt);
            cudaDeviceSynchronize();
            
            double copy_start = MPI_Wtime();
            double local_max = find_max_device(d_abs_errors, local_size);
            copy_time += MPI_Wtime() - copy_start;
            
            double global_max;
            double reduce_start = MPI_Wtime();

            MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
              comm_time += MPI_Wtime() - reduce_start;
            
            if (rank == 0) {
                std::cout << "Step " << step + 1 << "/" << K 
                          << ", error = " << global_max << std::endl;
            }
            max_abs_error = std::max(max_abs_error, global_max);
            error_time += MPI_Wtime() - error_start;
        }
    }
    
    loop_end = MPI_Wtime();
    
    final_start = MPI_Wtime();
    cudaFree(d_u0); cudaFree(d_u1); cudaFree(d_u2); cudaFree(d_abs_errors);
    cudaFree(d_send_left); cudaFree(d_recv_left);
    cudaFree(d_send_right); cudaFree(d_recv_right); 

    cudaFreeHost(h_send_left); cudaFreeHost(h_recv_left); 
       cudaFreeHost(h_send_right); cudaFreeHost(h_recv_right);
    final_end = MPI_Wtime();
    
    double total_end = MPI_Wtime();
    
    if (rank == 0) {
        double total_time = total_end - total_start;
        double init_time = init_end - init_start;
        double loop_time = loop_end - loop_start;
        double final_time = final_end - final_start;
        double compute_in_loop = compute_time;
        
        long long total_points = (long long)K * N * N * N;
        double mpoints_per_sec = total_points / (loop_time * 1e6);
        
        double peak_gflops = prop.clockRate * 1000.0 * prop.multiProcessorCount * 
                            prop.warpSize * 2 / 1e9;
        double achieved_gflops = (total_points * 7 * 2) / (loop_time * 1e9);
        double gpu_efficiency = (achieved_gflops / peak_gflops) * 100;
        
        std::cout << "\n=== RESULTS ===\n";
        std::cout << "MPI процессов: " << size << "\n";
        std::cout << "GPU: " << prop.name << "\n";
          std::cout << "Сетка: " << N << "x" << N << "x" << N << "\n";
        std::cout << "Общее время (сек): " << total_time << "\n";

        std::cout << "Инициализация (сек): " << init_time << "\n";
         std::cout << "Основной цикл (сек): " << loop_time << "\n";

        std::cout << "CUDA ядра (сек): " << compute_time << "\n";
          std::cout << "MPI коммуникации (сек): " << comm_time << "\n";
        std::cout << "Копирование GPU→CPU (сек): " << copy_time << "\n";

        std::cout << "Вычисления в цикле (сек): " << compute_in_loop << "\n";
          std::cout << "Финализация (сек): " << final_time << "\n";
        std::cout << "Физическое время (сек): " << K * dt << "\n";

           std::cout << "Погрешность: " << max_abs_error << "\n";
        std::cout << "Скорость (Мpoints/sec): " << mpoints_per_sec << "\n";

         std::cout << "Эффективность GPU: " << gpu_efficiency << "%\n";
        std::cout << "Ускорение: " << (size > 1 ? "N/A" : "1.0") << "\n";
    }
    
    MPI_Finalize();
    return 0;
}