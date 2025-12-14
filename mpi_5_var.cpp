#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <mpi.h>

const double d1 = 1.0;  
const double d2 = 1.0;
const double d3 = 1.0;
const double c1 = 1.0 / (4.0 * M_PI * M_PI);

class Data3D {
private:
    std::vector<double> buf;
    int sx, sy, sz;
    int gx, gy, gz;
    int rk, np;
    int bx, ex, by, ey, bz, ez;
    int dm[3], pd[3], crd[3];
    MPI_Comm cm;
    
    static const int GL = 1;
    
    int pos(int i, int j, int k) const {
        return (i + GL) * (sy + 2*GL) * (sz + 2*GL) + 
               (j + GL) * (sz + 2*GL) + 
               (k + GL);
    }

public:
    Data3D(int nx, int ny, int nz, int r, int n) 
        : gx(nx), gy(ny), gz(nz), rk(r), np(n) {
        
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
    int getEX() const { return ex; }
    int getEY() const { return ey; }
    int getEZ() const { return ez; }
    MPI_Comm getComm() const { return cm; }
    const int* getDims() const { return dm; }
    const int* getCoords() const { return crd; }
    
    std::vector<double> getData() const {
        std::vector<double> res(sx * sy * sz);
        int idx = 0;
        for (int i = 0; i < sx; i++) {
            for (int j = 0; j < sy; j++) {
                for (int k = 0; k < sz; k++) {
                    res[idx++] = (*this)(i, j, k);
                }
            }
        }
        return res;
    }
    
    void exchange() {
        MPI_Status st;
        
        int lx, rx;
        MPI_Cart_shift(cm, 0, 1, &lx, &rx);
        
        int ly, ry;
        MPI_Cart_shift(cm, 1, 1, &ly, &ry);
        
        int lz, rz;
        MPI_Cart_shift(cm, 2, 1, &lz, &rz);
        
        std::vector<double> sb, rb;
        
        if (dm[0] > 1) {
            sb.resize(sy * sz);
            rb.resize(sy * sz);
            
            int idx = 0;
            for (int j = 0; j < sy; j++) {
                for (int k = 0; k < sz; k++) {
                    sb[idx++] = (*this)(sx - 1, j, k);
                }
            }
            MPI_Sendrecv(sb.data(), sb.size(), MPI_DOUBLE, rx, 0,
                        rb.data(), rb.size(), MPI_DOUBLE, lx, 0,
                        cm, &st);
            
            idx = 0;
            for (int j = 0; j < sy; j++) {
                for (int k = 0; k < sz; k++) {
                    (*this)(-1, j, k) = rb[idx++];
                }
            }
            
            idx = 0;
            for (int j = 0; j < sy; j++) {
                for (int k = 0; k < sz; k++) {
                    sb[idx++] = (*this)(0, j, k);
                }
            }
            MPI_Sendrecv(sb.data(), sb.size(), MPI_DOUBLE, lx, 1,
                        rb.data(), rb.size(), MPI_DOUBLE, rx, 1,
                        cm, &st);
            
            idx = 0;
            for (int j = 0; j < sy; j++) {
                for (int k = 0; k < sz; k++) {
                    (*this)(sx, j, k) = rb[idx++];
                }
            }
        }
        
        if (dm[1] > 1) {
            sb.resize(sx * sz);
            rb.resize(sx * sz);
            
            int idx = 0;
            for (int i = 0; i < sx; i++) {
                for (int k = 0; k < sz; k++) {
                    sb[idx++] = (*this)(i, sy - 1, k);
                }
            }
            MPI_Sendrecv(sb.data(), sb.size(), MPI_DOUBLE, ry, 2,
                        rb.data(), rb.size(), MPI_DOUBLE, ly, 2,
                        cm, &st);
            
            idx = 0;
            for (int i = 0; i < sx; i++) {
                for (int k = 0; k < sz; k++) {
                    (*this)(i, -1, k) = rb[idx++];
                }
            }
            
            idx = 0;
            for (int i = 0; i < sx; i++) {
                for (int k = 0; k < sz; k++) {
                    sb[idx++] = (*this)(i, 0, k);
                }
            }
            MPI_Sendrecv(sb.data(), sb.size(), MPI_DOUBLE, ly, 3,
                        rb.data(), rb.size(), MPI_DOUBLE, ry, 3,
                        cm, &st);
            
            idx = 0;
            for (int i = 0; i < sx; i++) {
                for (int k = 0; k < sz; k++) {
                    (*this)(i, sy, k) = rb[idx++];
                }
            }
        }
        
        if (dm[2] > 1) {
            sb.resize(sx * sy);
            rb.resize(sx * sy);
            
            int idx = 0;
            for (int i = 0; i < sx; i++) {
                for (int j = 0; j < sy; j++) {
                    sb[idx++] = (*this)(i, j, sz - 1);
                }
            }
            MPI_Sendrecv(sb.data(), sb.size(), MPI_DOUBLE, rz, 4,
                        rb.data(), rb.size(), MPI_DOUBLE, lz, 4,
                        cm, &st);
            
            idx = 0;
            for (int i = 0; i < sx; i++) {
                for (int j = 0; j < sy; j++) {
                    (*this)(i, j, -1) = rb[idx++];
                }
            }
            
            idx = 0;
            for (int i = 0; i < sx; i++) {
                for (int j = 0; j < sy; j++) {
                    sb[idx++] = (*this)(i, j, 0);
                }
            }
            MPI_Sendrecv(sb.data(), sb.size(), MPI_DOUBLE, lz, 5,
                        rb.data(), rb.size(), MPI_DOUBLE, rz, 5,
                        cm, &st);
            
            idx = 0;
            for (int i = 0; i < sx; i++) {
                for (int j = 0; j < sy; j++) {
                    (*this)(i, j, sz) = rb[idx++];
                }
            }
        }
    }
};

double calc_exact(double x, double y, double z, double t) {
    double at = 0.5 * sqrt(4.0/(d1*d1) + 1.0/(d2*d2) + 1.0/(d3*d3));
    return sin(2.0*M_PI*x/d1) * sin(M_PI*y/d2) * sin(M_PI*z/d3) * cos(at * t + 2.0*M_PI);
}

double init_func(double x, double y, double z) {
    return calc_exact(x, y, z, 0.0);
}

double laplace(const Data3D& u, int i, int j, int k, double dx, double dy, double dz) {
    return (u(i-1, j, k) - 2.0*u(i, j, k) + u(i+1, j, k)) / (dx*dx) +
           (u(i, j-1, k) - 2.0*u(i, j, k) + u(i, j+1, k)) / (dy*dy) +
           (u(i, j, k-1) - 2.0*u(i, j, k) + u(i, j, k+1)) / (dz*dz);
}

void set_bounds(Data3D& u) {
    const int* dm = u.getDims();
    const int* crd = u.getCoords();
    
    if (crd[1] == 0) {
        for (int i = -1; i <= u.getX(); i++) {
            for (int k = -1; k <= u.getZ(); k++) {
                u(i, 0, k) = 0.0;
            }
        }
    }
    if (crd[1] == dm[1] - 1) {
        for (int i = -1; i <= u.getX(); i++) {
            for (int k = -1; k <= u.getZ(); k++) {
                u(i, u.getY() - 1, k) = 0.0;
            }
        }
    }
    
    if (crd[2] == 0) {
        for (int i = -1; i <= u.getX(); i++) {
            for (int j = -1; j <= u.getY(); j++) {
                u(i, j, 0) = 0.0;
            }
        }
    }
    if (crd[2] == dm[2] - 1) {
        for (int i = -1; i <= u.getX(); i++) {
            for (int j = -1; j <= u.getY(); j++) {
                u(i, j, u.getZ() - 1) = 0.0;
            }
        }
    }
}

void save_node_data(const Data3D& num, 
                   double dx, double dy, double dz, 
                   double ft,
                   int rk, int np, int N) {
    
    std::vector<double> lnum = num.getData();
    std::vector<double> lerr;
    std::vector<double> lx, ly, lz;
    
    int lp = num.getX() * num.getY() * num.getZ();
    lerr.resize(lp);
    lx.resize(lp);
    ly.resize(lp);
    lz.resize(lp);
    
    int idx = 0;
    for (int i = 0; i < num.getX(); i++) {
        for (int j = 0; j < num.getY(); j++) {
            for (int k = 0; k < num.getZ(); k++) {
                double x = (num.getBX() + i) * dx;
                double y = (num.getBY() + j) * dy;
                double z = (num.getBZ() + k) * dz;
                
                double ex = calc_exact(x, y, z, ft);
                double ap = lnum[idx];
                
                lx[idx] = x;
                ly[idx] = y;
                lz[idx] = z;
                lerr[idx] = std::abs(ap - ex);  
                idx++;
            }
        }
    }
    
    std::vector<int> rc(np), disp(np);
    int lc = lp;
    
    MPI_Gather(&lc, 1, MPI_INT, rc.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rk == 0) {
        disp[0] = 0;
        for (int i = 1; i < np; i++) {
            disp[i] = disp[i-1] + rc[i-1];
        }
    }
    
    std::vector<double> gerr, gx, gy, gz;
    if (rk == 0) {
        gerr.resize(N * N * N);
        gx.resize(N * N * N);
        gy.resize(N * N * N);
        gz.resize(N * N * N);
    }
    
    MPI_Gatherv(lerr.data(), lc, MPI_DOUBLE,
                gerr.data(), rc.data(), disp.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    MPI_Gatherv(lx.data(), lc, MPI_DOUBLE,
                gx.data(), rc.data(), disp.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    MPI_Gatherv(ly.data(), lc, MPI_DOUBLE,
                gy.data(), rc.data(), disp.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    MPI_Gatherv(lz.data(), lc, MPI_DOUBLE,
                gz.data(), rc.data(), disp.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    if (rk == 0) {
        std::string filename = "absolute_error_mpi_" + std::to_string(np) + "_" + std::to_string(N) + ".txt";
        std::ofstream file(filename);
        
        file << std::scientific << std::setprecision(15);
        file << "# x y z absolute_error" << std::endl;
        
        for (int i = 0; i < N * N * N; i++) {
            file << gx[i] << " " << gy[i] << " " << gz[i] << " " << gerr[i] << std::endl;
        }
        
        file.close();
        std::cout << "Absolute error data saved: " << filename << std::endl;
    }
}

void save_stats(int np, int N, double time, double abs_err) {
    std::string filename = "stats_mpi_" + std::to_string(np) + "_" + std::to_string(N) + ".txt";
    std::ofstream file(filename);
    
    file << std::scientific << std::setprecision(15);
    file << "# processes grid_size time absolute_error" << std::endl;
    file << np << " " << N << " " << time << " " << abs_err << std::endl;
    
    file.close();
    std::cout << "Statistics saved: " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rk, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rk);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    
    int N = 512;
    if (argc > 1) N = std::atoi(argv[1]);
    
    const int K = 20;
    const double dx = d1 / (N - 1);
    const double dy = d2 / (N - 1);
    const double dz = d3 / (N - 1);
    double min_d = dx;
    if (dy < min_d) min_d = dy;
    if (dz < min_d) min_d = dz;
    double dt = 0.05 * min_d / sqrt(3.0 * c1);
    
    if (rk == 0) {
        std::cout << "MPI simulation - Grid: " << N << "^3, Processes: " << np << std::endl;
        std::cout << "Time steps: " << K << ", dt: " << dt << std::endl;
    }
    
    Data3D v0(N, N, N, rk, np);
    Data3D v1(N, N, N, rk, np);
    Data3D v2(N, N, N, rk, np);
    
    double t1 = MPI_Wtime();

    
    for (int i = 0; i < v0.getX(); i++) {
        for (int j = 0; j < v0.getY(); j++) 
        {
            for (int k = 0; k < v0.getZ(); k++) {
                double x = (v0.getBX() + i) * dx;
                double y = (v0.getBY() + j) * dy;
                double z = (v0.getBZ() + k) * dz;
                v0(i, j, k) = init_func(x, y, z);
            }
        }
    }
    
    v0.exchange();
    set_bounds(v0);
    
    for (int i = 0; i < v0.getX(); i++) 
    {
        for (int j = 0; j < v0.getY(); j++) 
        {
            for (int k = 0; k < v0.getZ(); k++) {
                double l = laplace(v0, i, j, k, dx, dy, dz);
                
                v1(i, j, k) = v0(i, j, k) + c1 * dt * dt / 2.0 * l;

            }
        }
    }
    
    v1.exchange();
    set_bounds(v1);
    
    double merr = 0.0;
    
    for (int n = 1; n < K; n++) {
        double t = n * dt;
        
        for (int i = 0; i < v1.getX(); i++) 
        {
            for (int j = 0; j < v1.getY(); j++) {
                for (int k = 0; k < v1.getZ(); k++) 
                {
                    double l = laplace(v1, i, j, k, dx, dy, dz);
                    v2(i, j, k) = 2.0 * v1(i, j, k) - v0(i, j, k) + c1 * dt * dt * l;
                }
            }
        }
        
        v2.exchange();
        set_bounds(v2);
        
        double lerr = 0.0;
        for (int i = 0; i < v2.getX(); i++) 
        {
            for (int j = 0; j < v2.getY(); j++) 
            {
                for (int k = 0; k < v2.getZ(); k++) 
                {
                    double x = (v2.getBX() + i) * dx;
                    double y = (v2.getBY() + j) * dy;
                    double z = (v2.getBZ() + k) * dz;
                    double ex = calc_exact(x, y, z, t);
                    double ap = v2(i, j, k);
                    double ae = std::abs(ap - ex);
                    
                    lerr = std::max(lerr, ae);
                }
            }
        }
        
        double gerr;
        MPI_Allreduce(&lerr, &gerr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        
        merr = std::max(merr, gerr);
        
        for (int i = 0; i < v0.getX(); i++) 
        {
            for (int j = 0; j < v0.getY(); j++) 
            {
                for (int k = 0; k < v0.getZ(); k++)
                 {
                    v0(i, j, k) = v1(i, j, k);
                    v1(i, j, k) = v2(i, j, k);
                }
            }
        }
        
        if (rk == 0 && n % 5 == 0) {
            std::cout << "Step " << n << "/" << K << ", time: " << t << ", max error: " << gerr << std::endl;
        }
    }
    
    double t2 = MPI_Wtime();
    double tt = t2 - t1;
    
    double ft = K * dt;
    
    double gmerr;
    MPI_Reduce(&merr, &gmerr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rk == 0) {
        std::cout << "\n=== FINAL RESULTS ===" << std::endl;
        std::cout << "Processes: " << np << ", Grid: " << N << "^3" << std::endl;
        std::cout << "Total time: " << tt << " s" << std::endl;
        std::cout << "Max absolute error: " << gmerr << std::endl;
        std::cout << "Final time: " << ft << std::endl;
        
        save_stats(np, N, tt, gmerr);
    }
    
    save_node_data(v2, dx, dy, dz, ft, rk, np, N);
    
    MPI_Finalize();
    return 0;
}