#include <cstddef>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
//#include <cuda_runtime.h>
#include "vaporize.cuh"

__constant__ Data dat;

// __device__ inline double Max(double A, double B){if(A>B) return A; else return B;}
// __device__ inline double Min(double A, double B){if(A<B) return A; else return B;}

__global__ void CalcDiffusion(double* al, double* ti, double* al_next, double* ti_next, ST* state_field) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int z = blockIdx.y*blockDim.y + threadIdx.y;
    int center = x*dat.Nz+z;
    if (x < dat.Nx && z < dat.Nz) {
        if (state_field[center] == FLUID) {
            int left  = (x-1)*dat.Nz+z;
            int right = (x+1)*dat.Nz+z;
            int top    = x*dat.Nz+z-1;
            int bottom = x*dat.Nz+z+1;
            double al_mass_loss, ti_mass_loss;
            double leftal, leftti, rightal, rightti;
            double topal, topti, bottomal, bottomti;
            double al_dole = al[center] / (al[center] + ti[center]);
            al_mass_loss = 0;
            ti_mass_loss = 0;

            if (state_field[top] == GAS) {
                al_mass_loss = Max(0.0,      al_dole  * activity_Al(dat.temp, al_dole, dat.cts_Al_d) * mass_loss(dat.temp, dat.cts_Al_d) * dat.tau);
                ti_mass_loss = Max(0.0, (1 - al_dole) * activity_Ti(dat.temp, al_dole, dat.cts_Ti_d) * mass_loss(dat.temp, dat.cts_Ti_d) * dat.tau);
                //printf("loss: %e, param1: %e, param1: %e, temp: %e\n", mass_loss(dat.temp, dat.cts_Al_d), dat.cts_Al_d->MOL_MASS, dat.cts_Al_d->R, dat.temp);
            }
            if (state_field[left]   == WALL) {leftal   = al[center];leftti   = ti[center];} else {leftal   = al[left];  leftti   = ti[left];}
            if (state_field[right]  == WALL) {rightal  = al[center];rightti  = ti[center];} else {rightal  = al[right]; rightti  = ti[right];}
            if (state_field[top]    == WALL) {topal    = al[center];topti    = ti[center];} else {topal    = al[top];   topti    = ti[top];}
            if (state_field[bottom] == WALL) {bottomal = al[center];bottomti = ti[center];} else {bottomal = al[bottom];bottomti = ti[bottom];}

            al_next[center] = Max(0.0, al[center] + dat.tau * dat.D * ((rightal - 2*al[center] + leftal) / (dat.h*dat.h) + (bottomal - 2*al[center] + topal)/(dat.hz*dat.hz)) - al_mass_loss / 0.027 / dat.hz);
            ti_next[center] = Max(0.0, ti[center] + dat.tau * dat.D * ((rightti - 2*ti[center] + leftti) / (dat.h*dat.h) + (bottomti - 2*ti[center] + topti)/(dat.hz*dat.hz)) - ti_mass_loss / 0.048 / dat.hz);
        }
    }
}

__global__ void CalcHeatConduct(double *heat, double *heat_next, double *heat_source, double* al, double* ti, ST* state_field) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int z = blockIdx.y*blockDim.y + threadIdx.y;
    int index = x * dat.Nz + z;
    double heat_cond_al, heat_cond_al_dT;
    double heat_cond_ti, heat_cond_ti_dT;
    double al_mass_frac = al[index] / (al[index] + ti[index] * dat.cts_Ti_d->MOL_MASS / dat.cts_Al_d->MOL_MASS);
    int left  = (x-1)*dat.Nz+z;
    int right = (x+1)*dat.Nz+z;
    int top    = x*dat.Nz+z-1;
    int bottom = x*dat.Nz+z+1;

    if (x < dat.Nx && z < dat.Nz) {
        if(state_field[index] == FLUID || state_field[index] == SOLID){
            if (state_field[index] == FLUID) {
                heat_cond_al = dat.cts_Al_d->HEAT_COND_0_LIQ + dat.cts_Al_d->HEAT_COND_LIQ * 500;// * heat[index];
                heat_cond_al_dT = dat.cts_Al_d->HEAT_COND_LIQ;
                heat_cond_ti = dat.cts_Ti_d->HEAT_COND_0_LIQ + dat.cts_Ti_d->HEAT_COND_LIQ * 1000;//heat[index];
                heat_cond_ti_dT = dat.cts_Ti_d->HEAT_COND_LIQ;
            } else {
                heat_cond_al = dat.cts_Al_d->HEAT_COND_0_SOL + dat.cts_Al_d->HEAT_COND_SOL * 2000;//heat[index];
                heat_cond_al_dT = dat.cts_Al_d->HEAT_COND_SOL;
                heat_cond_ti = dat.cts_Ti_d->HEAT_COND_0_SOL + dat.cts_Ti_d->HEAT_COND_SOL * 3000;//heat[index];
                heat_cond_ti_dT = dat.cts_Ti_d->HEAT_COND_SOL;
            }
            
            // double heat_conduct = heat_cond_al    * al_mass_frac + heat_cond_ti    * (1 - al_mass_frac) - 0.72 * (heat_cond_al    - heat_cond_ti)    * al_mass_frac * (1-al_mass_frac);
            // double heat_conduct_dT = heat_cond_al_dT * al_mass_frac + heat_cond_ti_dT * (1 - al_mass_frac) - 0.72 * (heat_cond_al_dT - heat_cond_ti_dT) * al_mass_frac * (1-al_mass_frac);
            
            double leftD, rightD, topD, bottomD;
            if (state_field[top] == GAS) {topD = 0;}
            if (state_field[bottom] == GAS) {bottomD = 0;}
            if (state_field[left] == GAS) {leftD = 0;}
            if (state_field[right] == GAS) {rightD = 0;}

            double energy_source = 0;
            if (state_field[top] == GAS){
                energy_source = heat_source[x] + heat_loss(heat[index], dat.cts_Al_d);
            }

            heat_next[index] = heat[index] + dat.tau * (
                (rightD)*(heat[right] - heat[index]) - (leftD)  *(heat[index] - heat[left])  /(dat.h *dat.h ) +
                (topD)  *(heat[top]   - heat[index]) - (bottomD)*(heat[index] - heat[bottom])/(dat.hz*dat.hz) + energy_source
            );
        }
    }
}

__global__ void UpdateState(ST* state_field, double* heat) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int z = blockIdx.y*blockDim.y + threadIdx.y;
    int center = x*dat.Nz+z;
    if(x<dat.Nx && z<dat.Nz) {
        if(state_field[center] == FLUID && heat[center] <  dat.cts_Ti_d->TEMP_LIQ) {state_field[center] = SOLID;}
        if(state_field[center] == SOLID && heat[center] >= dat.cts_Ti_d->TEMP_LIQ) {state_field[center] = FLUID;}
    }
}

__global__ void CalcHeatSource(double *heat_source, int step) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if (x < dat.Nx) {
        double current_time = step * dat.tau;
        heat_source[x] = 2 * dat.beam_power / (M_PI * dat.beam_radius * dat.beam_radius) * exp(-2 * pow(current_time * dat.beam_vel + dat.beam_start * dat.h - x * dat.h, 2) / pow(dat.beam_radius, 2));
    }
}

__global__ void Move(double* al, double* ti, double* al_next, double* ti_next, ST* state_field){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int z = blockIdx.y*blockDim.y + threadIdx.y;
    int center = x*dat.Nz+z;
    int right  = (x+1)*dat.Nz+z;
    if (state_field[center] == FLUID && x==dat.Nx-2) {al_next[center] = dat.al_0; ti_next[center] = dat.ti_0;}
    if (state_field[center] == FLUID) {al_next[center] = al[right]; ti_next[center] = ti[right];}
}

__host__ void sim_temp(int temp1, int temp2, int temp_step, double time_stop, int drop_rate, std::string drop_dir)
{
    
    for (int temp=temp1; temp<temp2; temp+=temp_step)
    {   
        Data dat_host;
        std::ofstream output;
        dat_host.init((double)temp, drop_rate);
        dat_host.time_stop = time_stop;
        
        int threadsPerBlock = 8;
        int blocksPerGridX = (dat_host.Nx + threadsPerBlock - 1) / threadsPerBlock;
        int blocksPerGridY = (dat_host.Nz + threadsPerBlock - 1) / threadsPerBlock;
    
        dim3 blockShape(threadsPerBlock, threadsPerBlock);
        dim3 gridShape(blocksPerGridX, blocksPerGridY);

        cudaMemcpyToSymbol(dat, &dat_host, sizeof(Data));
        std::cout << "copy data: " << cudaGetErrorString(cudaGetLastError()) << "\n";
        //std::cout << dat_host.temp << std::endl;

        CalcHeatSource<<<blocksPerGridX, threadsPerBlock>>>(dat_host.heat_source_d, dat_host.step);
        // cudaMemcpy(dat_host.heat_source_h, dat_host.heat_source_d, dat_host.Nx*sizeof(double), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();    
        Gnuplot gp;
        std::vector<std::pair<double, double>> data;
        for (int x = 0; x < dat_host.Nx; x++) {
            double xPos = x * dat_host.h;
            double yTemp = dat_host.heat_source_h[x];
            data.push_back(std::make_pair(xPos, yTemp));
        }
        
        gp << "plot '-' u 1:2 w l\n";
        gp.send1d(data);

        while (dat_host.tim < time_stop)
        {
            dat_host.step++;
            std::cout << "temp " << temp << ", step " << dat_host.step << ": time = " << dat_host.tim << "; " << (dat_host.tim) / time_stop * 100 << "%" << std::endl;
            // CalcDiffusion<<<gridShape, blockShape>>>(dat_host.al_d, dat_host.ti_d, dat_host.al_next_d, dat_host.ti_next_d, dat_host.state_field_d);
            // std::swap(dat_host.al_next_d, dat_host.al_d);
            // std::swap(dat_host.ti_next_d, dat_host.ti_d);
            // cudaDeviceSynchronize();
            CalcHeatSource<<<blocksPerGridX, threadsPerBlock>>>(dat_host.heat_source_d, dat_host.step);
            // std::swap(dat_host.heat_next_d, dat_host.heat_d);
            // cudaDeviceSynchronize();
            // CalcHeatConduct<<<gridShape, blockShape>>>(dat_host.heat_d, dat_host.heat_next_d, dat_host.heat_source_d, dat_host.al_d, dat_host.ti_d, dat_host.state_field_d);
            // cudaDeviceSynchronize();
            
            if (dat_host.step % dat_host.drop_rate == 0) {
                dat_host.extract();
                Gnuplot gp;
                gp << "set terminal png size 1600,900; \nset view map; \nset pm3d at b corners2color c4; \nset palette model RGB rgbformulae 15,5,7; set palette negative; \nset samples 100; \nset isosamples 100; \nset xyplane relative 0; "; // set palette defined (0.0 \"white\", 1.0 \"black\")\n";
                gp << "set cbrange [0:1]; set xrange [0:" << dat_host.Nx * dat_host.h << "]; set yrange [" << 3*dat_host.Nz*dat_host.hz/2 << ":" << -dat_host.Nz*dat_host.hz/2  <<"]\n";
                output.open("data/dataT.txt");
                std::cout << "set output \"" << drop_dir << "/cmap" << (int)dat_host.temp << "_" << std::setfill('0') << std::setw(7) << (int)dat_host.step << ".png\"\n";
                gp        << "set output \"" << drop_dir << "/cmap" << temp << "_" << std::setfill('0') << std::setw(7) << (int)dat_host.step << ".png\"; ";
                gp << "set title \"Aluminium melt pool at t = " << std::setprecision(3) << dat_host.tim << ", T = " << (int)dat_host.temp << "\"; ";
                cudaDeviceSynchronize();
                for (int x = 0; x < dat_host.Nx; x++)
                {
                    for (int z = 0; z < dat_host.Nz; z++)
                    {
                        output << x*dat_host.h << " " << z*dat_host.hz << " " << dat_host.al_h[x*dat_host.Nz+z]/dat_host.al_0 << " " << dat_host.ti_h[x*dat_host.Nz+z]/dat_host.ti_0 << " " << dat_host.state_field_h[x*dat_host.Nz+z]/3.0 << "\n";
                    }
                    output << "\n";
                }
                std::cout << "plotting\n";
                output.close();
                gp << "splot \"data/dataT.txt\" u 1:2:3 with pm3d\n";
                std::cout << "Done\n";
            }
            dat_host.tim += dat_host.tau;
            dat_host.deltaX += dat_host.beam_vel * dat_host.tau;
        }
        dat_host.clean();
    }
    cudaDeviceSynchronize();
}

int main()
{
    // sim_temp(2700.0, 3300.0, 100.0, 1e-5, 1000, "plots/cmapT_anim11_45");
    // sim_temp(3300.0, 4400.0, 200.0, 1e-5, 1000, "plots/cmapT_anim11_45");
    sim_temp(3000.0, 3001.0, 200.0, 1e-4, 1000, "plots/cmap_anim11_45");

    return 0;
}
