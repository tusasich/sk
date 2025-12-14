#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <mpi.h>
#include <omp.h>

const double d1 = 1.0;
const double d2 = 1.0;
const double d3 = 1.0;
const double c1 = 1.0 / (4.0 * M_PI * M_PI);

double exact_solution(double x, double y, double z, double t) {
    return sin(2 * M_PI * x) * sin(2 * M_PI * y) * sin(2 * M_PI * z) * cos(2 * M_PI * c1 * t);
}

class Data3D {
private:
    std::vector<double> buf;
    int sx, sy, sz;
    int gx, gy, gz;
    int rk, np;
    int bx, ex, by, ey, bz, ez;
    int dm[3], pd[3], crd[3];
    MPI_Comm cm;

    std::vector<double> send_buf[6];
    std::vector<double> recv_buf[6];
    MPI_Request send_req[6];
    MPI_Request recv_req[6];
    int active_requests;

    static const int GL = 1;

    int pos(int i, int j, int k) const {
        return (i + GL) * (sy + 2*GL) * (sz + 2*GL) +
               (j + GL) * (sz + 2*GL) +
               (k + GL);
    }

public:
    Data3D(int nx, int ny, int nz, int r, int n)
        : gx(nx), gy(ny), gz(nz), rk(r), np(n), active_requests(0) {

        dm[0] = dm[1] = dm[2] = 0;
        MPI_Dims_create(np, 3, dm);

        pd[0] = 1;
        pd[1] = 0;
        pd[2] = 0;
        MPI_Cart_create(MPI_COMM_WORLD, 3, dm, pd, 0, &cm);
        MPI_Cart_coords(cm, rk, 3, crd);

        sx = gx / dm[0];
        sy = gy / dm[1];
        sz = gz / dm[2];

        bx = crd[0] * sx;
        by = crd[1] * sy;
        bz = crd[2] * sz;

        if (crd[0] == dm[0] - 1) sx = gx - bx;
        if (crd[1] == dm[1] - 1) sy = gy - by;
        if (crd[2] == dm[2] - 1) sz = gz - bz;

        ex = bx + sx - 1;
        ey = by + sy - 1;
        ez = bz + sz - 1;

        buf.resize((sx + 2*GL) * (sy + 2*GL) * (sz + 2*GL), 0.0);

        if (dm[0] > 1) {
            send_buf[0].resize(sy * sz);
            recv_buf[0].resize(sy * sz);
            send_buf[1].resize(sy * sz);
            recv_buf[1].resize(sy * sz);
        }
        if (dm[1] > 1) {
            send_buf[2].resize(sx * sz);
            recv_buf[2].resize(sx * sz);
            send_buf[3].resize(sx * sz);
            recv_buf[3].resize(sx * sz);
        }
        if (dm[2] > 1) {
            send_buf[4].resize(sx * sy);
            recv_buf[4].resize(sx * sy);
            send_buf[5].resize(sx * sy);
            recv_buf[5].resize(sx * sy);
        }
        
        for (int i = 0; i < 6; i++) {
            send_req[i] = MPI_REQUEST_NULL;
            recv_req[i] = MPI_REQUEST_NULL;
        }
    }

    double& operator()(int i, int j, int k) {
        return buf[pos(i, j, k)];
    }

    const double& operator()(int i, int j, int k) const {
        return buf[pos(i, j, k)];
    }

    int getX() const { return sx; }
    int getY() const { return sy; }
    int getZ() const { return sz; }
    int getBX() const { return bx; }
    int getBY() const { return by; }
    int getBZ() const { return bz; }
    
    void swapBuffers(Data3D& other) {
        std::swap(buf, other.buf);
    }
    
    void waitAll() {
        if (active_requests > 0) {
            MPI_Waitall(active_requests, send_req, MPI_STATUSES_IGNORE);
            MPI_Waitall(active_requests, recv_req, MPI_STATUSES_IGNORE);
            active_requests = 0;
        }
    }

    void exchange() {
        int neighbors[6];
        MPI_Cart_shift(cm, 0, 1, &neighbors[1], &neighbors[0]);
        MPI_Cart_shift(cm, 1, 1, &neighbors[3], &neighbors[2]);
        MPI_Cart_shift(cm, 2, 1, &neighbors[5], &neighbors[4]);

        active_requests = 0;

        if (dm[0] > 1) {
            #pragma omp parallel for collapse(2)
            for (int j = 0; j < sy; j++) {
                for (int k = 0; k < sz; k++) {
                    send_buf[0][j * sz + k] = (*this)(sx - 1, j, k);
                    send_buf[1][j * sz + k] = (*this)(0, j, k);
                }
            }

            MPI_Isend(send_buf[0].data(), sy * sz, MPI_DOUBLE, neighbors[0], 0, cm, &send_req[active_requests]);
            MPI_Irecv(recv_buf[0].data(), sy * sz, MPI_DOUBLE, neighbors[1], 0, cm, &recv_req[active_requests]);
            active_requests++;

            MPI_Isend(send_buf[1].data(), sy * sz, MPI_DOUBLE, neighbors[1], 1, cm, &send_req[active_requests]);
            MPI_Irecv(recv_buf[1].data(), sy * sz, MPI_DOUBLE, neighbors[0], 1, cm, &recv_req[active_requests]);
            active_requests++;
        }

        if (dm[1] > 1) {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < sx; i++) {
                for (int k = 0; k < sz; k++) {
                    send_buf[2][i * sz + k] = (*this)(i, sy - 1, k);
                    send_buf[3][i * sz + k] = (*this)(i, 0, k);
                }
            }

            MPI_Isend(send_buf[2].data(), sx * sz, MPI_DOUBLE, neighbors[2], 2, cm, &send_req[active_requests]);
            MPI_Irecv(recv_buf[2].data(), sx * sz, MPI_DOUBLE, neighbors[3], 2, cm, &recv_req[active_requests]);
            active_requests++;

            MPI_Isend(send_buf[3].data(), sx * sz, MPI_DOUBLE, neighbors[3], 3, cm, &send_req[active_requests]);
            MPI_Irecv(recv_buf[3].data(), sx * sz, MPI_DOUBLE, neighbors[2], 3, cm, &recv_req[active_requests]);
            active_requests++;
        }

        if (dm[2] > 1) {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < sx; i++) {
                for (int j = 0; j < sy; j++) {
                    send_buf[4][i * sy + j] = (*this)(i, j, sz - 1);
                    send_buf[5][i * sy + j] = (*this)(i, j, 0);
                }
            }

            MPI_Isend(send_buf[4].data(), sx * sy, MPI_DOUBLE, neighbors[4], 4, cm, &send_req[active_requests]);
            MPI_Irecv(recv_buf[4].data(), sx * sy, MPI_DOUBLE, neighbors[5], 4, cm, &recv_req[active_requests]);
            active_requests++;

            MPI_Isend(send_buf[5].data(), sx * sy, MPI_DOUBLE, neighbors[5], 5, cm, &send_req[active_requests]);
            MPI_Irecv(recv_buf[5].data(), sx * sy, MPI_DOUBLE, neighbors[4], 5, cm, &recv_req[active_requests]);
            active_requests++;
        }

        waitAll();

        if (dm[0] > 1) {
            #pragma omp parallel for collapse(2)
            for (int j = 0; j < sy; j++) {
                for (int k = 0; k < sz; k++) {
                    (*this)(-1, j, k) = recv_buf[1][j * sz + k];
                    (*this)(sx, j, k) = recv_buf[0][j * sz + k];
                }
            }
        }

        if (dm[1] > 1) {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < sx; i++) {
                for (int k = 0; k < sz; k++) {
                    (*this)(i, -1, k) = recv_buf[3][i * sz + k];
                    (*this)(i, sy, k) = recv_buf[2][i * sz + k];
                }
            }
        }

        if (dm[2] > 1) {
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < sx; i++) {
                for (int j = 0; j < sy; j++) {
                    (*this)(i, j, -1) = recv_buf[5][i * sy + j];
                    (*this)(i, j, sz) = recv_buf[4][i * sy + j];
                }
            }
        }
    }

    double laplace(int i, int j, int k, double dx2, double dy2, double dz2) const {
        double d2u_dx2 = ((*this)(i+1, j, k) - 2*(*this)(i, j, k) + (*this)(i-1, j, k)) / dx2;
        double d2u_dy2 = ((*this)(i, j+1, k) - 2*(*this)(i, j, k) + (*this)(i, j-1, k)) / dy2;
        double d2u_dz2 = ((*this)(i, j, k+1) - 2*(*this)(i, j, k) + (*this)(i, j, k-1)) / dz2;
        return d2u_dx2 + d2u_dy2 + d2u_dz2;
    }
};

void init_condition(Data3D& u, Data3D& u_old, double dx, double dy, double dz, double dt) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < u.getX(); i++) {
        for (int j = 0; j < u.getY(); j++) {
            for (int k = 0; k < u.getZ(); k++) {
                double x = (u.getBX() + i) * dx;
                double y = (u.getBY() + j) * dy;
                double z = (u.getBZ() + k) * dz;
                
                u(i, j, k) = exact_solution(x, y, z, 0);
                u_old(i, j, k) = exact_solution(x, y, z, -dt);
            }
        }
    }
}

void compute_error(const Data3D& u, double dx, double dy, double dz, double t, 
                   double& local_max_abs_error, double& local_abs_error_norm,
                   double& local_exact_norm) {
    
    local_max_abs_error = 0.0;
    local_abs_error_norm = 0.0;
    local_exact_norm = 0.0;
    
    #pragma omp parallel for collapse(3) reduction(max:local_max_abs_error) \
                                     reduction(+:local_abs_error_norm, local_exact_norm)
    for (int i = 0; i < u.getX(); i++) {
        for (int j = 0; j < u.getY(); j++) {
            for (int k = 0; k < u.getZ(); k++) {
                double x = (u.getBX() + i) * dx;
                double y = (u.getBY() + j) * dy;
                double z = (u.getBZ() + k) * dz;
                
                double exact_val = exact_solution(x, y, z, t);
                double approx_val = u(i, j, k);
                double abs_error = fabs(approx_val - exact_val);
                
                local_max_abs_error = std::max(local_max_abs_error, abs_error);
                local_abs_error_norm += abs_error * abs_error;
                local_exact_norm += exact_val * exact_val;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rk, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rk);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    int N = 128;
    const int K = 50;

    if (argc > 1) N = std::atoi(argv[1]);

    double total_start = MPI_Wtime();

    char* env_threads = std::getenv("OMP_NUM_THREADS");
    int actual_threads = (env_threads != NULL) ? std::atoi(env_threads) : 1;
    omp_set_num_threads(actual_threads);

    const double dx = d1 / (N - 1);
    const double dy = d2 / (N - 1);
    const double dz = d3 / (N - 1);
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;
    double min_d = std::min({dx, dy, dz});
    double dt = 0.1 * min_d / sqrt(3.0 * c1);

    if (rk == 0) {
        std::cout << "Grid: " << N << "x" << N << "x" << N << std::endl;
        std::cout << "MPI: " << np << ", OMP: " << actual_threads << std::endl;
        std::cout << "Steps: " << K << std::endl;
    }

    double init_start = MPI_Wtime();
    Data3D u(N, N, N, rk, np);
    Data3D u_old(N, N, N, rk, np);
    Data3D u_new(N, N, N, rk, np);
    double init_time = MPI_Wtime() - init_start;

    init_condition(u, u_old, dx, dy, dz, dt);
    u.exchange();
    u_old.exchange();

    double comp_start = MPI_Wtime();

    for (int step = 0; step < K; step++) {
        #pragma omp parallel for collapse(3)
        for (int i = 0; i < u.getX(); i++) {
            for (int j = 0; j < u.getY(); j++) {
                for (int k = 0; k < u.getZ(); k++) {
                    double lap = u.laplace(i, j, k, dx2, dy2, dz2);
                    u_new(i, j, k) = 2 * u(i, j, k) - u_old(i, j, k) + c1 * dt * dt * lap;
                }
            }
        }
        
        u_new.exchange();
        
        u_old.swapBuffers(u);
        u.swapBuffers(u_new);
    }

    double comp_time = MPI_Wtime() - comp_start;

    double final_time = K * dt;
    double local_max_abs_error = 0.0;
    double local_abs_error_norm = 0.0;
    double local_exact_norm = 0.0;
    
    compute_error(u, dx, dy, dz, final_time, 
                  local_max_abs_error, local_abs_error_norm, local_exact_norm);

    double global_max_abs_error = 0.0;
    double global_abs_error_norm = 0.0;
    double global_exact_norm = 0.0;
    
    MPI_Reduce(&local_max_abs_error, &global_max_abs_error, 1, MPI_DOUBLE, 
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_abs_error_norm, &global_abs_error_norm, 1, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_exact_norm, &global_exact_norm, 1, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);
    
    double relative_error_L2 = 0.0;
    if (rk == 0 && global_exact_norm > 0) {
        global_abs_error_norm = sqrt(global_abs_error_norm);
        global_exact_norm = sqrt(global_exact_norm);
        relative_error_L2 = global_abs_error_norm / global_exact_norm;
    }

    double total_time = MPI_Wtime() - total_start;

    double max_init_time, max_comp_time, max_total_time;
    MPI_Reduce(&init_time, &max_init_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rk == 0) {
        std::cout << "Время: init=" << max_init_time << " comp=" << max_comp_time << " total=" << max_total_time << std::endl;
        std::cout << "Выполнение: " << (K * N * N * N / max_comp_time / 1e9) << " GDOF/s" << std::endl;
        std::cout << "Максиммальная абсолютная ошибка: " << global_max_abs_error << std::endl;
        std::cout << "Относительная ошибка: " << relative_error_L2 * 100 << "%" << std::endl;
    }

    MPI_Finalize();
    return 0;
}