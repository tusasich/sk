#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

double Lx = 1;
double Ly = 1;
double Lz = 1;
double a = 1.0/(4*M_PI*M_PI);

double anal(double x, double y, double z, double t) {
    double temp = 0.5*sqrt(4/(Lx*Lx)+1/(Ly*Ly)+1/(Lz*Lz));
    return sin(2*M_PI*x/Lx)*sin(M_PI*y/Ly)*sin(M_PI*z/Lz)*cos(temp*t+2*M_PI);
}

double phi(double x, double y, double z) 
{
    return anal(x,y,z,0);
}

double lap(std::vector<std::vector<std::vector<double>>>& u, int i, int j, int k, double hx, double hy, double hz)
 {
    return (u[i-1][j][k]-2*u[i][j][k]+u[i+1][j][k])/(hx*hx) 
         + (u[i][j-1][k]-2*u[i][j][k]+u[i][j+1][k])/(hy*hy)
         + (u[i][j][k-1]-2*u[i][j][k]+u[i][j][k+1])/(hz*hz);
}

int main() {
    int N = 128;
    int step = 20;

    double hx = Lx/(N-1);
    double hy = Ly/(N-1);
    double hz = Lz/(N-1);

    double tau = 0.1*std::min(hx,std::min(hy,hz))/sqrt(3*a);

    std::cout << "Сетка: " << N << ", шаг времени: " << tau << ", шагов: " << step << std::endl;

    std::chrono::high_resolution_clock::time_point start_total, end_total;
    double total_time = 0.0;
    double comp_time = 0.0;

    start_total = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<std::vector<double>>> u0(N, std::vector<std::vector<double>>(N, std::vector<double>(N)));
    std::vector<std::vector<std::vector<double>>> u1 = u0;
    std::vector<std::vector<std::vector<double>>> u2 = u0;


    auto start_comp = std::chrono::high_resolution_clock::now();
    for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
    for(int k=0;k<N;k++)
        u0[i][j][k] = phi(i*hx,j*hy,k*hz);
    auto end_comp = std::chrono::high_resolution_clock::now();
    comp_time += std::chrono::duration<double>(end_comp - start_comp).count();


    start_comp = std::chrono::high_resolution_clock::now();
    for(int i=1;i<N-1;i++)
    for(int j=1;j<N-1;j++)
    for(int k=1;k<N-1;k++)
        u1[i][j][k] = u0[i][j][k] + a*tau*tau/2 * lap(u0,i,j,k,hx,hy,hz);
    end_comp = std::chrono::high_resolution_clock::now();
    comp_time += std::chrono::duration<double>(end_comp - start_comp).count();

    for(int i=0;i<N;i++)
    for(int j=0;j<N;j++) 
{
        u1[i][j][0] = u1[i][j][N-1] = 0;
        u1[i][0][j] = u1[i][N-1][j] = 0;
        u1[0][i][j] = u1[N-1][i][j] = 0;
    }

    double max_err = 0;

    for(int n=1;n<step;n++)
     {
        double t = n*tau;

        start_comp = std::chrono::high_resolution_clock::now();
        for(int i=1;i<N-1;i++)
        for(int j=1;j<N-1;j++)
        for(int k=1;k<N-1;k++)
            u2[i][j][k] = 2*u1[i][j][k] - u0[i][j][k] + a*tau*tau*lap(u1,i,j,k,hx,hy,hz);
        end_comp = std::chrono::high_resolution_clock::now();
        comp_time += std::chrono::duration<double>(end_comp - start_comp).count();

        for(int i=0;i<N;i++)
        for(int j=0;j<N;j++) 
    {
            u2[i][j][0] = u2[i][j][N-1] = 0;
            u2[i][0][j] = u2[i][N-1][j] = 0;
            u2[0][i][j] = u2[N-1][i][j] = 0;
        }

     
        start_comp = std::chrono::high_resolution_clock::now();
        double cur_err = 0;
        for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
        for(int k=0;k<N;k++) {
            double diff = fabs(u2[i][j][k]-anal(i*hx,j*hy,k*hz,t));
            if(diff>cur_err) cur_err = diff;
        }
        end_comp = std::chrono::high_resolution_clock::now();
        comp_time += std::chrono::duration<double>(end_comp - start_comp).count();

        if(cur_err>max_err) max_err = cur_err;

        u0 = u1;
        u1 = u2;
    }

    end_total = std::chrono::high_resolution_clock::now();
    total_time = std::chrono::duration<double>(end_total - start_total).count();

    std::cout << "Общее время работы программы: " << total_time << " сек" << std::endl;
    std::cout << "Максимальная погрешность: " << max_err << std::endl;
    std::cout << "Время вычислений: " << comp_time << " сек" << std::endl;

    return 0;
}
