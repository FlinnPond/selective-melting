#include <cmath>
#include <iostream>
#include <fstream>
#include <gnuplot-iostream.h>
#include <string>
#include <vector>

struct constants{
    double TEMP_MELT;
    double BOLTZ_CONST;
    double R;
    double MOL_MASS;
    double MASS_ATOM;
    double MASS_ATOM_ATM;
    double TEMP_CRIT;
    double TEMP_BOIL;
    double TEMP_LIQ;
    double EVAPOR_HEAT;
    double FUSION_HEAT;
    double HEAT_CAP_LIQ;
    double HEAT_CAP_SOL;
    double press_atm;
    double PRESS_AMB;
    float HEATS_RAT;
    float HEATS_RAT_ATM;
    double TEMP_ATM;
    double LIQ_DENSITY;
    double SOL_DENSITY;
    double HEAT_COND_0_SOL, HEAT_COND_SOL;
    double HEAT_COND_0_LIQ, HEAT_COND_LIQ;
// constants();
    constants(std::string name);
};
enum ST {
    FLUID,
    GAS,
    SOLID,
    WALL,
};
struct Data {
    constants* cts_Al_h, *cts_Ti_h;
    constants* cts_Al_d, *cts_Ti_d;
    double h, hz, tau;
    double tim, time_stop;
    int step, drop_rate;
    double beam_radius, beam_vel, beam_start, beam_power, deltaX; // m\s
    
    double substrate_length, substrate_width, substrate_depth;
    double calc_length, calc_width, calc_depth;

    int Nx_calc, Nz_calc;
    int Nx,Nz;

    double ti_density, al_density, al_dole, al_mass_dole;

    double density, m_al, m_ti;

    double deltam_al, deltam_ti;
    double al_mass_loss, ti_mass_loss;

    double topal, bottomal, leftal, rightal;
    double topti, bottomti, leftti, rightti;
    
    double *al_d, *ti_d, *al_next_d, *ti_next_d;
    double *al_h, *ti_h, *al_next_h, *ti_next_h;
    double *heat_source_h, *heat_source_d;

    double *heat_h, *heat_next_h;
    double *heat_d, *heat_next_d;

    double *D_h, *D_d;

    double lambda_al_0_s, lambda_al_s;
    double lambda_al_0_l, lambda_al_l;
    double lambda_ti_0_s, lambda_ti_s;
    double lambda_ti_0_l, lambda_ti_l;

    ST* state_field_h, *state_field_d;

    double D, temp;
    double al_0, ti_0;

    //__host__ __device__ Data(){}
    __host__ void init(double temp, int drop_rate);
    __host__ void clean();
    __host__ void update();
    __host__ void extract();
};


__device__ inline double _f1  (double mach, constants* cts);
__device__ inline double _f1_d(double mach, constants* cts);
__device__ inline double _f2  (double mach, constants* cts);
__device__ inline double _f2_d(double mach, constants* cts);
__device__ inline double _f3   (double mach, double temp_surf, constants* cts);
__device__ inline double _f3_d (double mach, double temp_surf, constants* cts);
__device__ inline double _f3_dt(double mach, double temp_surf, constants* cts);
__device__ inline double f   (double mach, double temp_surf, constants* cts);
__device__ inline double f_d (double mach, double temp_surf, constants* cts);
__device__ inline double f_dt(double mach, double temp_surf, constants* cts);
__device__ inline double g  (double temp_surf, constants* cts);
__device__ inline double g_d(double temp_surf, constants* cts);
__device__ inline double g_new   (double temp_surf, constants* cts);
__device__ inline double g_new_dt(double temp_surf, constants* cts);
__device__ inline double h_new   (double mach, double t_s, constants* cts);
__device__ inline double h_new_dt(double mach, double t_s, constants* cts);
__device__ inline double h   (double mach, double t_s, constants* cts);
__device__ inline double h_dt(double mach, double t_s, constants* cts);
__device__ inline double activity_Al(double temp_surf, double mole_dole, constants* cts);
__device__ inline double activity_Ti(double temp_surf, double mole_dole, constants* cts);
__device__ inline double ps(double t_s, constants* cts);
__device__ inline double mach_exact(double t_s, constants* cts);
__device__ inline double heat_loss(double temp, constants* cts);
__device__ inline double mass_loss(double temp, constants* cts);
__device__ inline double recoil   (double temp, constants* cts);

constants::constants(std::string name)
{
    if (name == "Al")
    {
        TEMP_MELT = 933;
        BOLTZ_CONST = 1.38e-23;
        R = 8.31;
        MOL_MASS = 0.027;
        MASS_ATOM = 27 / 6e26;
        MASS_ATOM_ATM = 39 / 6e26;
        TEMP_CRIT = 6700;
        TEMP_BOIL = 2726;
        TEMP_LIQ  = 933;
        EVAPOR_HEAT = 11600000;
        FUSION_HEAT = 390000;
        HEAT_CAP_LIQ = 1177;
        HEAT_CAP_SOL = 1039;
        press_atm = 1e5;
        PRESS_AMB = 1e5;
        HEATS_RAT = 1.66;
        HEATS_RAT_ATM = 1.66;
        TEMP_ATM = 300;
        LIQ_DENSITY = 2380;
        SOL_DENSITY = 2700;
        HEAT_COND_0_SOL = 261.1; HEAT_COND_SOL =  -54;
        HEAT_COND_0_LIQ = 63;    HEAT_COND_LIQ =   30;
        }
    else if (name == "Ti")
    {
        TEMP_MELT = 1940;
        BOLTZ_CONST = 1.38e-23;
        R = 8.31;
        MOL_MASS = 0.048;
        MASS_ATOM = 48 / 6e26;
        MASS_ATOM_ATM = 4 / 6e26;
        TEMP_CRIT = 7890;
        TEMP_BOIL = 3558;
        TEMP_LIQ  = 1940;
        EVAPOR_HEAT = 9700000;
        FUSION_HEAT = 305000;
        HEAT_CAP_LIQ = 966;
        HEAT_CAP_SOL = 660;
        press_atm = 1e5;
        PRESS_AMB = 1e5;
        HEATS_RAT = 1.66;
        HEATS_RAT_ATM = 1.66;
        TEMP_ATM = 300;
        LIQ_DENSITY = 4130;
        SOL_DENSITY = 4506;
        HEAT_COND_0_SOL = -0.3; HEAT_COND_SOL = 14.6;
        HEAT_COND_0_LIQ = -6.7; HEAT_COND_LIQ = 18.3;
    }
    else if (name == "Fe")
    {
        TEMP_MELT = 1538;
        BOLTZ_CONST = 1.38e-23;
        R = 8.31;
        MOL_MASS = 0.056;
        MASS_ATOM = 56 / 6e26;
        MASS_ATOM_ATM = 4 / 6e26;
        TEMP_CRIT = 7890;
        TEMP_BOIL = 3558;
        TEMP_LIQ  = 1940;
        EVAPOR_HEAT = 9700000;
        FUSION_HEAT = 305000;
        HEAT_CAP_LIQ = 966;
        HEAT_CAP_SOL = 660;
        press_atm = 1e5;
        PRESS_AMB = 1e5;
        HEATS_RAT = 1.66;
        HEATS_RAT_ATM = 1.66;
        TEMP_ATM = 298;
        LIQ_DENSITY = 4110;
    }
}
__device__ inline double Max(double A, double B){if(A>B) {return A;} else {return B;}}
__device__ inline double Min(double A, double B){if(A<B) {return A;} else {return B;}}

__host__ void Data::init(double _temp, int _drop_rate){
    temp=_temp;
    h = 3e-6;
    hz = 3e-6;
    tau = 1e-9;
    time_stop = 1e-6;
    drop_rate = (_drop_rate==-1) ? (int)(time_stop/tau) : _drop_rate;  // drop once at the end
    beam_vel = 1; // m\s
    beam_power = 300;
    beam_start = 200;
    beam_radius = 5e-5;
    deltaX = 0;
    
    substrate_length = 1e-5;
    substrate_depth  = 1e-5;
    substrate_width  = 1e-5;

    calc_length = 20e-6;
    calc_depth  = 2e-6;  
    calc_width  = 300e-6;

    //Nz = (int)(bed_depth  / h);
    // Nx_total = (int)(substrate_length / h);
    // Nz_total = (int)(substrate_depth / hz);
    // Nz = (int)(calc_depth / hz);
    // Nx = (int)(calc_length / h);
    Nx_calc = 50;
    Nz_calc = 10;
    Nx = 1200;
    Nz = 200;

    lambda_ti_0_s = -0.3;  lambda_ti_s = 14.6;
    lambda_ti_0_l = -6.7;  lambda_ti_l = 18.3;

    std::cout << "Allocating data maps and constants on host...\n"; 
    al_h = (double*)(malloc(Nx*Nz*sizeof(double)));
    ti_h = (double*)(malloc(Nx*Nz*sizeof(double)));
    al_next_h = (double*)(malloc(Nx*Nz*sizeof(double)));
    ti_next_h = (double*)(malloc(Nx*Nz*sizeof(double)));
    heat_h = (double*)(malloc(Nx*Nz*sizeof(double)));
    heat_next_h = (double*)(malloc(Nx*Nz*sizeof(double)));
    state_field_h = (ST*)(malloc(Nx*Nz*sizeof(ST)));
    D_h = (double*)(malloc(Nx*Nz*sizeof(double)));
    heat_source_h = (double*)(malloc(Nx*sizeof(double)));
    cts_Al_h = (constants*)(malloc(sizeof(constants)));
    cts_Ti_h = (constants*)(malloc(sizeof(constants)));
    std::cout << "Initializing constants on host\n";
    *cts_Al_h = constants("Al");
    *cts_Ti_h = constants("Ti");
    cudaMalloc((void**)&cts_Al_d, sizeof(constants)); cudaMemcpy(cts_Al_d, cts_Al_h, sizeof(constants), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&cts_Ti_d, sizeof(constants)); cudaMemcpy(cts_Ti_d, cts_Ti_h, sizeof(constants), cudaMemcpyHostToDevice);

    // cudaMallocHost((void**)&al_h, Nx*Nz*sizeof(double)); cudaMemset(al_h, al_0, Nx*Nz*sizeof(double));
    // cudaMallocHost((void**)&ti_h, Nx*Nz*sizeof(double)); cudaMemset(ti_h, ti_0, Nx*Nz*sizeof(double));
    // cudaMallocHost((void**)&al_next_h, Nx*Nz*sizeof(double)); cudaMemset(al_next_h, al_0, Nx*Nz*sizeof(double));
    // cudaMallocHost((void**)&ti_next_h, Nx*Nz*sizeof(double)); cudaMemset(ti_next_h, ti_0, Nx*Nz*sizeof(double));
    // cudaMallocHost((void**)&state_field_h, Nx*Nz*sizeof(ST));
    
    ti_density = 4506;
    al_density = 2700;
    al_dole = 0.5;
    al_mass_dole = al_dole * cts_Al_h->MOL_MASS / (al_dole * cts_Al_h->MOL_MASS + (1.0 - al_dole) * cts_Ti_h->MOL_MASS);

    density = al_mass_dole * al_density + (1 - al_mass_dole) * ti_density;
    m_al = density * al_mass_dole * h*h*h;
    m_ti = density * (1 - al_mass_dole) * h*h*h;

    tim = 0;
    deltam_al = 0;
    deltam_ti = 0;
    al_mass_loss = 0;
    ti_mass_loss = 0;

    D = 1e-9;
    al_0 = density/0.027 * al_mass_dole;
    ti_0 = density/0.048 * (1 - al_mass_dole);
    step = 0;

    std::cout << "Initializing data on host\n";
    for (int x = 0; x < Nx; x++){
        for (int z = 0; z < Nz; z++){
            al_h[x*Nz+z] = al_0;
            ti_h[x*Nz+z] = ti_0;
            al_next_h[x*Nz+z] = al_0;
            ti_next_h[x*Nz+z] = ti_0;
            state_field_h[x*Nz+z] = SOLID;
            if (x==Nx-1) {state_field_h[x*Nz+z] = WALL;}
            if (x==0)    {state_field_h[x*Nz+z] = WALL;}
            heat_h[x*Nz+z] = 300;
            heat_next_h[x*Nz+z] = 300;
            // std::cout << x << " of " << Nx << ", " << z << " of " << Nz << std::endl;
        }
        state_field_h[x*Nz+0] = GAS;
        //if (x>=Nx - 50 && state_field_h[x*Nz+1] != WALL) {state_field_h[x*Nz+1] = INTERFACE;}
        state_field_h[x*Nz+Nz-1] = WALL;
    }
    std::cout << "### al_0 = " << al_0 << std::endl;
    std::cout << "Defining data maps and constants on device\n";
    cudaMalloc((void**)&al_d, Nx*Nz*sizeof(double));       cudaMemcpy(al_d, al_h, Nx*Nz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&ti_d, Nx*Nz*sizeof(double));       cudaMemcpy(ti_d, ti_h, Nx*Nz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&al_next_d, Nx*Nz*sizeof(double));  cudaMemcpy(al_next_d, al_next_h, Nx*Nz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&ti_next_d, Nx*Nz*sizeof(double));  cudaMemcpy(ti_next_d, ti_next_h, Nx*Nz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&heat_d, Nx*Nz*sizeof(double));     cudaMemcpy(heat_d, heat_h, Nx*Nz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&heat_next_d, Nx*Nz*sizeof(double));cudaMemcpy(heat_next_d, heat_next_h, Nx*Nz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&D_d, Nx*Nz*sizeof(double));        cudaMemcpy(D_d, D_h, Nx*Nz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&state_field_d, Nx*Nz*sizeof(ST));  cudaMemcpy(state_field_d, state_field_h, Nx*Nz*sizeof(ST), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&heat_source_d, Nx*sizeof(double)); cudaMemcpy(heat_source_d, heat_source_h, Nx*sizeof(double), cudaMemcpyHostToDevice);
    //std::cout << "memset " << cts_Al_h->MOL_MASS << ": " << cudaGetErrorString(cudaGetLastError()) << "\n";
}

__host__ void Data::extract(){
    cudaMemcpy(al_h, al_d, Nx*Nz*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ti_h, ti_d, Nx*Nz*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(heat_h, heat_d, Nx*Nz*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(heat_source_h, heat_source_d, Nx*sizeof(double), cudaMemcpyDeviceToHost);
}

__host__ void Data::clean(){
        free(al_h);     free(ti_h);     free(al_next_h);     free(ti_next_h); 
    cudaFree(al_d); cudaFree(ti_d); cudaFree(al_next_d); cudaFree(ti_next_d); 
        free(heat_h);     free(heat_next_h);     free(state_field_h);     free(heat_source_h);
    cudaFree(heat_d); cudaFree(heat_next_d); cudaFree(state_field_d); cudaFree(heat_source_d);
}

__device__ double _f1(double mach, constants* cts){
    return sqrt(1 + M_PI * pow(( cts->HEATS_RAT - 1) / ( cts->HEATS_RAT + 1) * mach / 2, 2)) - sqrt(M_PI) * ( cts->HEATS_RAT - 1) / ( cts->HEATS_RAT + 1) * mach / 2;
}
__device__ double _f1_d(double mach, constants* cts){
    return M_PI * mach / 4 * pow(( cts->HEATS_RAT - 1) / ( cts->HEATS_RAT + 1), 2) / sqrt(1 + M_PI * pow(( cts->HEATS_RAT - 1) / ( cts->HEATS_RAT + 1) * mach / 2, 2)) - 
    sqrt(M_PI) * ( cts->HEATS_RAT - 1) / ( cts->HEATS_RAT + 1) * 0.5;
}
__device__ double _f2(double mach, constants* cts){
    double f1 = _f1(mach, cts);
    return pow(f1, -1) * (0.5 * (2 * pow(mach, 2) + 1) * exp(pow(mach, 2)) * erfc(mach) - mach / sqrt(M_PI)) + 0.5 * pow(f1, -2) * (1 - sqrt(M_PI) * mach * exp(pow(mach, 2)) * erfc(mach));
}
__device__ double _f2_d(double mach, constants* cts){
    double f1 = _f1(mach, cts);
    double f1_d = _f1_d(mach, cts);
    double a = (2 * pow(mach, 2) + 1);
    double b = exp(pow(mach, 2));
    double c = erfc(mach);
    return -1 * pow(f1, -2) * f1_d * (0.5 * a * b * c - mach / sqrt(M_PI)) +
    pow(f1, -1) * ((2 * mach * b + a * b * mach) * c - a * b * exp( - pow(mach, 2)) / sqrt(M_PI) - pow(M_PI, -0.5)) -
    pow(f1, -3) * f1_d * (1 - sqrt(M_PI) * mach * b * c) - 
    0.5 * sqrt(M_PI) * pow(f1, -2) * ((b + 2 * b * pow(mach, 2)) * c - 2 * mach * b * exp( - pow(mach, 2)) / sqrt(M_PI));
}
__device__ double _f4(double mach, double temp_surf, constants* cts){
    double f1 = _f1(sqrt( cts->HEATS_RAT / 2) * mach, cts);
    return  cts->HEATS_RAT * sqrt( cts->MASS_ATOM_ATM *  cts->HEATS_RAT * temp_surf /  cts->MASS_ATOM /  cts->HEATS_RAT_ATM /  cts->TEMP_ATM) * f1;
}
__device__ double _f4_d(double mach, double temp_surf, constants* cts){
    double f1 = _f1(sqrt( cts->HEATS_RAT / 2) * mach, cts);
    double f1_d = _f1_d(sqrt( cts->HEATS_RAT / 2) * mach, cts);
    return  cts->HEATS_RAT * sqrt( cts->MASS_ATOM_ATM *  cts->HEATS_RAT * temp_surf /  cts->MASS_ATOM /  cts->HEATS_RAT_ATM /  cts->TEMP_ATM) * f1_d;
}
__device__ double _f4_dt(double mach, double temp_surf, constants* cts)
{
    double f1 = _f1(sqrt( cts->HEATS_RAT / 2) * mach, cts);
    return  cts->HEATS_RAT * sqrt( cts->MASS_ATOM_ATM *  cts->HEATS_RAT * pow(f1, 2) /  cts->MASS_ATOM /  cts->HEATS_RAT_ATM /  cts->TEMP_ATM) * 0.5 / sqrt(temp_surf);
}
__device__ double _f3(double mach, double temp_surf, constants* cts)
{
  //double f4 = _f4(mach, temp_surf, cts);
  //return ( cts->HEATS_RAT + 1) / (4 *  cts->HEATS_RAT) * mach * f4;

    double f1 = _f1(sqrt( cts->HEATS_RAT / 2) * mach, cts);
    return ( cts->HEATS_RAT + 1) / 4 * sqrt( cts->MASS_ATOM_ATM *  cts->HEATS_RAT * temp_surf /  cts->MASS_ATOM /  cts->HEATS_RAT_ATM /  cts->TEMP_ATM) * mach * f1;
}
__device__ double _f3_d(double mach, double temp_surf, constants* cts)
{
    double f1   = _f1  (sqrt( cts->HEATS_RAT / 2) * mach, cts);
    double f1_d = _f1_d(sqrt( cts->HEATS_RAT / 2) * mach, cts);
  //return ( cts->HEATS_RAT + 1) / (4 *  cts->HEATS_RAT) * (f4 + f4_d * mach);
    return ( cts->HEATS_RAT + 1) / 4 * sqrt( cts->MASS_ATOM_ATM *  cts->HEATS_RAT * temp_surf /  cts->MASS_ATOM /  cts->HEATS_RAT_ATM /  cts->TEMP_ATM) * (f1 + mach * f1_d);
}
__device__ double _f3_dt(double mach, double temp_surf, constants* cts)
{
    double f1 = _f1(sqrt( cts->HEATS_RAT / 2) * mach, cts);
    // return ( cts->HEATS_RAT + 1) / (4 *  cts->HEATS_RAT) * mach * f4_dt;
    return ( cts->HEATS_RAT + 1) / 4 * sqrt( cts->MASS_ATOM_ATM *  cts->HEATS_RAT /  cts->MASS_ATOM /  cts->HEATS_RAT_ATM /  cts->TEMP_ATM) * mach * f1 * 0.5 / sqrt(temp_surf);

}
__device__ double f(double mach, double temp_surf, constants* cts)
{
    double f1 = _f1(sqrt( cts->HEATS_RAT / 2) * mach, cts);
    double f2 = _f2(sqrt( cts->HEATS_RAT / 2) * mach, cts);
    double f3 = _f3(mach, temp_surf, cts);
    double f4 = _f4(mach, temp_surf, cts);

  //return pow(f1, -2) * pow(f2, -1) * (1 + mach * f4 * (f3 + sqrt(1 + pow(f3, 2)))) - p_s /  cts->press_atm;
    return pow(f1, -2) * pow(f2, -1) * (1 + 4 *  cts->HEATS_RAT / ( cts->HEATS_RAT + 1) * f3 * (f3 + sqrt(1 + pow(f3, 2))));
}
__device__ double f_d(double mach, double temp_surf, constants* cts)
{
    double f1   = _f1  (sqrt( cts->HEATS_RAT / 2) * mach, cts);
    double f1_d = _f1_d(sqrt( cts->HEATS_RAT / 2) * mach, cts);
    double f2   = _f2  (sqrt( cts->HEATS_RAT / 2) * mach, cts);
    double f2_d = _f2_d(sqrt( cts->HEATS_RAT / 2) * mach, cts);
    double f3   = _f3  (mach, temp_surf, cts);
    // double f3_d = _f3_d(mach, temp_surf, cts);
    double f3_d = (_f3(mach, temp_surf, cts) - _f3(mach - 0.001, temp_surf, cts)) * 1000;
    double f4   = _f4  (mach, temp_surf, cts);
    double f4_d = _f4_d(mach, temp_surf, cts);

    // return (-2 * pow(f1, -3) * pow(f2, -1) * f1_d - pow(f1*f2, -2) * f2_d) * (1 + mach * f4 * (f3 + sqrt(1 + pow(f3, 2)))) + 
    // pow(f1, -2) * pow(f2, -1) * ((f4 + mach * f4_d) * (f3 + sqrt(1 + pow(f3, 2))) + mach * f4 * (f3_d + f3 * f3_d / (sqrt(1 + pow(f3, 2)))));

    return (-2 * pow(f1, -3) * pow(f2, -1) * f1_d - pow(f1*f2, -2) * f2_d) * (1 + 4 *  cts->HEATS_RAT / ( cts->HEATS_RAT + 1) * f3 * (f3 + sqrt(1 + pow(f3, 2)))) + 
    pow(f1, -2) * pow(f2, -1) * 4 *  cts->HEATS_RAT / ( cts->HEATS_RAT + 1) * (f3_d * (f3 + sqrt(1 + pow(f3, 2))) + f3 * (f3_d + f3 * f3_d / (sqrt(1 + pow(f3, 2)))));
}
__device__ double f_dt(double mach, double temp_surf, constants* cts)
{
    double f1 = _f1(sqrt( cts->HEATS_RAT / 2) * mach, cts);
    double f2 = _f2(sqrt( cts->HEATS_RAT / 2) * mach, cts);
    double f3    = _f3   (mach, temp_surf, cts);
    double f3_dt = _f3_dt(mach, temp_surf, cts);
    double f4    = _f4   (mach, temp_surf, cts);
    double f4_dt = _f4_dt(mach, temp_surf, cts);
    //return pow(f1, -2) * pow(f2, -1) *(mach * f4_dt * (f3 + sqrt(1 + pow(f3, 2))) + mach * f4 * (f3_dt + f3 * f3_dt / (sqrt(1 + pow(f3, 2)))));
    return pow(f1, -2) * pow(f2, -1) * 4 *  cts->HEATS_RAT / ( cts->HEATS_RAT + 1) * (f3_dt * (f3 + sqrt(1 + pow(f3, 2))) + f3 * (f3_dt + f3 * f3_dt / (sqrt(1 + pow(f3, 2)))));
}
__device__ double g(double temp_surf, constants* cts)
{
    return  cts->press_atm /  cts->PRESS_AMB * exp
    (
        -  cts->MASS_ATOM *  cts->EVAPOR_HEAT /  cts->BOLTZ_CONST * 
        (
            1 / temp_surf     * sqrt(1 - pow(temp_surf     /  cts->TEMP_CRIT, 2)) - 
            1 /  cts->TEMP_BOIL * sqrt(1 - pow( cts->TEMP_BOIL /  cts->TEMP_CRIT, 2)) +
            1 /  cts->TEMP_CRIT * (asin(temp_surf /  cts->TEMP_CRIT) - asin( cts->TEMP_BOIL /  cts->TEMP_CRIT))
        )
    );
}
__device__ double g_d(double temp_surf, constants* cts)
{
    return (g(temp_surf, cts) - g(temp_surf - 0.1, cts)) * 10;
    // return  cts->press_atm /  cts->PRESS_AMB * g(temp_surf, cts) *  cts->MASS_ATOM *  cts->EVAPOR_HEAT /  cts->BOLTZ_CONST * pow(temp_surf, -2) * sqrt(1 - pow(temp_surf /  cts->TEMP_CRIT, 2));
}
__device__ double g_new(double temp_surf, constants* cts)
{
    return pow(10, 10.945 - 16221 / temp_surf) /  cts->PRESS_AMB;
}
__device__ double g_new_dt(double temp_surf, constants* cts)
{
    return g_new(temp_surf, cts) * 2.3026 * 16221 / pow(temp_surf, 2);
}
__device__ double h(double mach, double t_s, constants* cts)
{
    return f(mach, t_s, cts) - g(t_s, cts);
}
__device__ double h_dt(double mach, double t_s, constants* cts)
{
    return f_dt(mach, t_s, cts) - g_d(t_s, cts);
}
__device__ double h_new(double mach, double t_s, constants* cts)
{
    return f(mach, t_s, cts) - g_new(t_s, cts);
}
__device__ double h_new_dt(double mach, double t_s, constants* cts)
{
    return f_dt(mach, t_s, cts) - g_new_dt(t_s, cts);
}
__device__ double activity_Al(double temp_surf, double mole_dole, constants* cts)
{
    int L_Ti_Al[3] {-108250 + 38 * (int)temp_surf, -6000 + 5 * (int)temp_surf, 15000};
    int sum = 0;
    for (int i = 0; i < 3; i++)
    {
        sum += L_Ti_Al[i] * pow((2 * mole_dole - 1), i);
    }
    // sum = -108250 + 38 * (int)temp_surf + ( - 6000 + 5 * (int)temp_surf ) * (2*mole_dole - 1) + 15000  * pow((2 * mole_dole - 1), 2);
    return exp(pow((1 - mole_dole), 2) * (sum + 2 * mole_dole * (L_Ti_Al[1] + 2 * L_Ti_Al[2] * (2 * mole_dole - 1))) / ( cts->R * temp_surf));
}
__device__ double activity_Ti(double temp_surf, double mole_dole, constants* cts)
{
    int L_Ti_Al[3] {-108250 + 38 * (int)temp_surf, -6000 + 5 * (int)temp_surf, 15000};
    int sum = 0;
    for (int i = 0; i < 3; i++)
    {
        sum += L_Ti_Al[i] * pow((2 * mole_dole - 1), i);
    }
    // sum = -108250 + 38 * (int)temp_surf + ( - 6000 + 5 * (int)temp_surf ) * (2*mole_dole - 1) + 15000  * pow((2 * mole_dole - 1), 2);
    return exp(pow(mole_dole, 2) * (sum - 2 * (1 - mole_dole) * (L_Ti_Al[1] + 2 * L_Ti_Al[2] * (2 * mole_dole - 1))) / ( cts->R * temp_surf));
}
__device__ double ps(double t_s, constants* cts)
{
    return  cts->press_atm * exp
    (
        -  cts->MASS_ATOM *  cts->EVAPOR_HEAT /  cts->BOLTZ_CONST * 
        (
            1 / t_s     * sqrt(1 - pow(t_s     /  cts->TEMP_CRIT, 2)) - 
            1 /  cts->TEMP_BOIL * sqrt(1 - pow( cts->TEMP_BOIL /  cts->TEMP_CRIT, 2)) +
            1 /  cts->TEMP_CRIT * (asin(t_s /  cts->TEMP_CRIT) - asin( cts->TEMP_BOIL /  cts->TEMP_CRIT))
        )
    );
}
__device__ double mach_exact(double t_s, constants* cts)
{
    double mach = 0.5;
    for (int j = 0; j < 10; j++)
    {
        mach -= h(mach, t_s, cts) / f_d(mach, t_s, cts);
    }
    return mach;
}
__device__ double heat_loss(double temp, constants* cts)
{
    if (g(temp, cts) < 1)
    {
        return 0;
    }

    double mach = Max(Min(1.0, mach_exact(temp, cts)), 0.0);
    float phi = sqrt(2 * M_PI *  cts->HEATS_RAT) * mach * _f2(sqrt( cts->HEATS_RAT / 2) * mach, cts) * _f1(sqrt( cts->HEATS_RAT / 2) * mach, cts);
    float j = phi * ps(temp, cts) * sqrt( cts->MASS_ATOM / (2 * M_PI *  cts->BOLTZ_CONST * temp));
    //std::cout << "ps = " << ps(temp, cts) << std::endl;
    float evap_loss = j * ( cts->EVAPOR_HEAT +  cts->FUSION_HEAT +  cts->HEAT_CAP_SOL * temp +  cts->HEAT_CAP_LIQ * (temp -  cts->TEMP_MELT));
    return evap_loss;
}
__device__ double mass_loss(double temp, constants* cts)
{
    if (g(temp, cts) < 1)
    {
        return 0;
    }

    double mach = Max(Min(1.0, mach_exact(temp, cts)), 0.0);
    float phi = sqrt(2 * M_PI *  cts->HEATS_RAT) * mach * _f2(sqrt( cts->HEATS_RAT / 2) * mach, cts) * _f1(sqrt( cts->HEATS_RAT / 2) * mach, cts);
    float j = phi * ps(temp, cts) * sqrt( cts->MASS_ATOM / (2 * M_PI *  cts->BOLTZ_CONST * temp));
    //std::cout << "ps = " << ps(temp, cts) << std::endl;
    return j;

}
__device__ double recoil(double temp, constants* cts)
{
    if (g(temp, cts) < 1)
    {
        return 0;
    }

    double mach = Max(Min(1.0, mach_exact(temp, cts)), 0.0);
    float phi = sqrt(2 * M_PI *  cts->HEATS_RAT) * mach * _f2(sqrt( cts->HEATS_RAT / 2) * mach, cts) * _f1(sqrt( cts->HEATS_RAT / 2) * mach, cts);
    float recoil = 0.5 * ps(temp, cts) * (1 + 0.5 * (1 - phi) * (1 + _f2(sqrt( cts->HEATS_RAT / 2) * mach, cts) * pow(_f1(sqrt( cts->HEATS_RAT / 2) * mach, cts), 2)));
    return recoil;
}
