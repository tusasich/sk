#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <omp.h>

double LX = 1.0;
double LY = 1.0;
double LZ = 1.0;
double TT = 0.1;
double A = 1.0 / (4.0 * M_PI * M_PI);

struct Grid {
    double* data;
    int nx, ny, nz;
    
    Grid(int _nx, int _ny, int _nz) : nx(_nx), ny(_ny), nz(_nz) {
        data = new double[nx * ny * nz]();
    }
    
    ~Grid() {
        delete[] data;
    }
    
    double& operator()(int i, int j, int k) {
        return data[i * ny * nz + j * nz + k];
    }
    
    double get(int i, int j, int k) const {
        return data[i * ny * nz + j * nz + k];
    }
    
    void setZero() {
        for(int i = 0; i < nx*ny*nz; i++) data[i] = 0.0;
    }
};

double toch(double x, double y, double z, double t) {
    double w = 0.5 * sqrt(4.0/(LX*LX) + 1.0/(LY*LY) + 1.0/(LZ*LZ));
    return sin(2.0*3.141592653589793*x/LX) * 
           sin(3.141592653589793*y/LY) * 
           sin(3.141592653589793*z/LZ) * 
           cos(w * t + 2.0*3.141592653589793);
}

double nach(double x, double y, double z) {
    return toch(x, y, z, 0.0);
}

double laplace(const Grid& u, int i, int j, int k, double hx, double hy, double hz) {
    double d2x = (u.get(i-1, j, k) - 2.0*u.get(i, j, k) + u.get(i+1, j, k)) / (hx*hx);
    double d2y = (u.get(i, j-1, k) - 2.0*u.get(i, j, k) + u.get(i, j+1, k)) / (hy*hy);
    double d2z = (u.get(i, j, k-1) - 2.0*u.get(i, j, k) + u.get(i, j, k+1)) / (hz*hz);
    return d2x + d2y + d2z;
}

int main(int argc, char** argv) {
    int potoki = 4;
    if(argc > 1) potoki = atoi(argv[1]);
    omp_set_num_threads(potoki);
    
    int NN = 128;
    int shagi = 20;
    
    double hx = LX / (NN - 1);
    double hy = LY / (NN - 1);
    double hz = LZ / (NN - 1);
    
    double tau = 0.1;
    double minh = hx;
    if(hy < minh) minh = hy;
    if(hz < minh) minh = hz;
    tau = tau * minh / sqrt(3.0 * A);
    
    std::cout << "параметры" << std::endl;
    std::cout << "размер " << NN << std::endl;
    std::cout << "шаги " << shagi << std::endl;
    std::cout << "потоки " << potoki << std::endl;
    std::cout << "шаг времени " << tau << std::endl;
    
    Grid u0(NN, NN, NN);
    Grid u1(NN, NN, NN);
    Grid u2(NN, NN, NN);
    #pragma omp parallel for
    for(int i=0;i<NN;i++){
        double x = i*hx;
        for(int j=0;j<NN;j++){
            double y = j*hy;
            for(int k=0;k<NN;k++){
                double z = k*hz;
                u0(i,j,k) = nach(x,y,z);
            }
        }
    }
    #pragma omp parallel for
    for(int i=1;i<NN-1;i++){
        for(int j=1;j<NN-1;j++){
            for(int k=1;k<NN-1;k++){
                double l = laplace(u0,i,j,k,hx,hy,hz);
                u1(i,j,k) = u0(i,j,k) + A*tau*tau/2.0*l;
            }
        }
    }
    
    for(int j=0;j<NN;j++)
    {
        for(int k=0;k<NN;k++)
        {
            u1(0,j,k)=0; u1(NN-1,j,k)=0;
        }
    }
    for(int i=0;i<NN;i++)
    {
        for(int k=0;k<NN;k++)
        {
            u1(i,0,k)=0; u1(i,NN-1,k)=0;
        }
    }
    for(int i=0;i<NN;i++)
    {
        for(int j=0;j<NN;j++)
        {
            u1(i,j,0)=0; u1(i,j,NN-1)=0;
        }
    }
    
    double max_abs = 0;
    double max_rel = 0;

    double t1 = omp_get_wtime();
    
    std::vector<double> vremena;
    std::vector<double>  abs_osh;
    std::vector<double> rel_osh;
    
    vremena.push_back(0);
    abs_osh.push_back(0);
    rel_osh.push_back(0);
    
    for(int n=1;n<shagi;n++)
    {
        double t = n*tau;
        
        #pragma omp parallel for
        for(int i=1;i<NN-1;i++){
            for(int j=1;j<NN-1;j++)
            {
                for(int k=1;k<NN-1;k++)
                {
                    double l = laplace(u1,i,j,k,hx,hy,hz);
                    u2(i,j,k) = 2.0*u1(i,j,k) - u0(i,j,k) + A*tau*tau*l;
                }
            }
        }
        for(int j=0;j<NN;j++)
        {
            for(int k=0;k<NN;k++)
            {
                u2(0,j,k)=0; u2(NN-1,j,k)=0;
            }
        }
        for(int i=0;i<NN;i++)
        {
            for(int k=0;k<NN;k++)
            {
                u2(i,0,k)=0; u2(i,NN-1,k)=0;
            }
        }
        for(int i=0;i<NN;i++)
        
        {
            for(int j=0;j<NN;j++)
            {
                u2(i,j,0)=0; u2(i,j,NN-1)=0;
            }
        }
        
        double tek_abs = 0;
        double tek_rel = 0;
        
        #pragma omp parallel for reduction(max:tek_abs,tek_rel)
        for(int i=0;i<NN;i++)
        {
            double x=i*hx;
            for(int j=0;j<NN;j++)
            
            {
                double y=j*hy;
                for(int k=0;k<NN;k++)
                {
                    double z=k*hz;
                    double exact = toch(x,y,z,t);
                    double pribl = u2(i,j,k);

                    double abs_e = fabs(pribl-exact);
                    double rel_e = 0;
                    
                    if(fabs(exact)>1e-12){
                        rel_e = fabs(pribl-exact)/fabs(exact)*100.0;
                    }else{
                        rel_e = abs_e*100.0;
                    }
                    
                    if(abs_e>tek_abs) tek_abs=abs_e;
                    if(rel_e>tek_rel) tek_rel=rel_e;
                }
            }
        }
        
        vremena.push_back(t);
        abs_osh.push_back(tek_abs);
        rel_osh.push_back(tek_rel);
        
        if(tek_abs>max_abs) max_abs=tek_abs;
        if(tek_rel>max_rel) max_rel=tek_rel;
        
       
        #pragma omp parallel for
        for(int i=0;i<NN;i++)
        {
            for(int j=0;j<NN;j++)
            {
                for(int k=0;k<NN;k++){
                    u0(i,j,k)=u1(i,j,k);
                    u1(i,j,k)=u2(i,j,k);
                }
            }
        }
        
        if(n%5==0){
            std::cout<<"шаг "<<n<<" время "<<t<<" ошибка "<<tek_abs<<" относ "<<tek_rel<<std::endl;
        }
    }
    
    double t2=omp_get_wtime();
    double total=t2-t1;
    
    std::cout<<"всего времени "<<total<<" сек"<<std::endl;
    std::cout<<"макс ошибка "<<max_abs<<std::endl;
    std::cout<<"макс относ "<<max_rel<<"%"<<std::endl;
    
    double fin_t = shagi*tau;
    double fin_abs = 0;
    double fin_rel = 0;
    

    Grid anal(NN,NN,NN);
    Grid abs_err(NN,NN,NN);

    Grid rel_err(NN,NN,NN);
    
    #pragma omp parallel for reduction(max:fin_abs,fin_rel)
    for(int i=0;i<NN;i++){
        double x=i*hx;
        for(int j=0;j<NN;j++){
            double y=j*hy;
            for(int k=0;k<NN;k++)
            {  
                double z=k*hz;
                double exact=toch(x,y,z,fin_t);
                double pribl=u2(i,j,k);
                double ae=fabs(pribl-exact);
                double re=0;
                
                if(fabs(exact)>1e-12){
                    re=fabs(pribl-exact)/fabs(exact)*100.0;
                }else{
                    re=ae*100.0;
                }
                
                anal(i,j,k)=exact;
                abs_err(i,j,k)=ae;
                rel_err(i,j,k)=re;
                
                if(ae>fin_abs) fin_abs=ae;
                if(re>fin_rel) fin_rel=re;
            }
        }
    }
    
    std::cout<<"в конце: abs="<<fin_abs<<" rel="<<fin_rel<<std::endl;
    

    std::ofstream f1("chisl.txt");
    std::ofstream f2("tochno.txt");
    std::ofstream f3("abs_err.txt");
    std::ofstream f4("rel_err.txt");
    
    f1.precision(12);
    f2.precision(12);
    f3.precision(12);
    f4.precision(12);
    
    for(int i=0;i<NN;i++){
        double x=i*hx;
        for(int j=0;j<NN;j++){
            double y=j*hy;
            for(int k=0;k<NN;k++){
                double z=k*hz;
                f1<<x<<" "<<y<<" "<<z<<" "<<u2(i,j,k)<<std::endl;
                f2<<x<<" "<<y<<" "<<z<<" "<<anal(i,j,k)<<std::endl;
                f3<<x<<" "<<y<<" "<<z<<" "<<abs_err(i,j,k)<<std::endl;
                f4<<x<<" "<<y<<" "<<z<<" "<<rel_err(i,j,k)<<std::endl;
            }
        }
    }
    
    f1.close(); f2.close(); f3.close(); f4.close();
    
    std::ofstream f5("evolution.txt");
    for(size_t i=0;i<vremena.size();i++){
        f5<<i<<" "<<vremena[i]<<" "<<abs_osh[i]<<" "<<rel_osh[i]<<std::endl;
    }
    f5.close();
    
    return 0;
}