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

__global__ void CalcDiffusion(double* al, double* ti, double* al_next, double* ti_next, ST* state_field, double* heat) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int center = y*dat.Nx*dat.Nz + x*dat.Nz + z;
    printf("hey");
    if (x < dat.Nx && y<dat.Ny && z < dat.Nz && state_field[center] == FLUID) {
        int xp = y*dat.Nx*dat.Nz + (x+1)*dat.Nz + z;
        int xm = y*dat.Nx*dat.Nz + (x-1)*dat.Nz + z;
        int zp = y*dat.Nx*dat.Nz + x*dat.Nz + z + 1;
        int zm = y*dat.Nx*dat.Nz + x*dat.Nz + z - 1;
        int yp = (y+1)*dat.Nx*dat.Nz + x*dat.Nz + z;
        int ym = (y-1)*dat.Nx*dat.Nz + x*dat.Nz + z;
        double al_dole = (al[center]+ti[center] == 0) ? 0 : al[center] / (al[center] + ti[center]);
        double al_mass_loss = 0;
        double ti_mass_loss = 0;

        if (state_field[zm] == GAS) {
            al_mass_loss = Max(0.0,      al_dole  * activity_Al(heat[center], al_dole, dat.cts_Al_d) * mass_loss(heat[center], dat.cts_Al_d) * dat.tau);
            ti_mass_loss = Max(0.0, (1 - al_dole) * activity_Ti(heat[center], al_dole, dat.cts_Ti_d) * mass_loss(heat[center], dat.cts_Ti_d) * dat.tau);
        }
        double xmD = (state_field[xm] == FLUID) ? dat.D : 0;
        double xpD = (state_field[xp] == FLUID) ? dat.D : 0;
        double ymD = (state_field[ym] == FLUID) ? dat.D : 0;
        double ypD = (state_field[yp] == FLUID) ? dat.D : 0;
        double zmD = (state_field[zm] == FLUID) ? dat.D : 0;
        double zpD = (state_field[zp] == FLUID) ? dat.D : 0;

        al_next[center] = Max(0.0, al[center] + dat.tau * (
            (xmD*(al[xm] - al[center]) + xpD*(al[xp] - al[center]))/(dat.h*dat.h) + 
            (ymD*(al[ym] - al[center]) + ypD*(al[yp] - al[center]))/(dat.h*dat.h) + 
            (zmD*(al[zm] - al[center]) + zpD*(al[zp] - al[center]))/(dat.h*dat.h)
        ) - al_mass_loss / 0.027 / dat.h);

        ti_next[center] = Max(0.0, ti[center] + dat.tau * (
            (xmD*(ti[xm] - ti[center]) + xpD*(ti[xp] - ti[center]))/(dat.h*dat.h) + 
            (ymD*(ti[ym] - ti[center]) + ypD*(ti[yp] - ti[center]))/(dat.h*dat.h) + 
            (zmD*(ti[zm] - ti[center]) + zpD*(ti[zp] - ti[center]))/(dat.h*dat.h)
        ) - ti_mass_loss / 0.048 / dat.h);
        // if (x == 200 && z == 1 && y == 1) {
        //     printf("D: %e, al_next: %e, mass_loss: %e\n", dat.D, al_next[center], dat.tau * al_mass_loss / 0.048 / dat.h);
        // }

        // if (isnan(al_next[center])) {
        //     printf("X:%i, z: %i, al_next: %e, loss: %e, mass_loss/mu/h: %e, mass_loss: %e, activity_Al: %e, al_dole: %e \n",x, z, al_next[center], al_mass_loss / 0.027 / dat.hz, al_mass_loss, mass_loss(heat[center], dat.cts_Al_d) * dat.tau, activity_Al(heat[center], al_dole, dat.cts_Al_d), al_dole);
        //     al_next[center] = dat.al_0;
        // }
    }
}

__global__ void CalcHeatConduct(double *heat, double *heat_next, double *heat_source, double* al, double* ti, ST* state_field) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int center = y*dat.Nx*dat.Nz + x*dat.Nz + z;
    int xp = y*dat.Nx*dat.Nz + (x+1)*dat.Nz + z;
    int xm = y*dat.Nx*dat.Nz + (x-1)*dat.Nz + z;
    int zp = y*dat.Nx*dat.Nz + x*dat.Nz + z + 1;
    int zm = y*dat.Nx*dat.Nz + x*dat.Nz + z - 1;
    int yp = (y+1)*dat.Nx*dat.Nz + x*dat.Nz + z;
    int ym = (y-1)*dat.Nx*dat.Nz + x*dat.Nz + z;

    if (x < dat.Nx && y<dat.Ny && z < dat.Nz && (state_field[center] == FLUID || state_field[center] == SOLID)) {
        // double al_mass_frac = al[center] / (al[center] + ti[center] * dat.cts_Ti_d->MOL_MASS / dat.cts_Al_d->MOL_MASS);
        double heat_cond = (dat.cts_Al_d->HEAT_COND_0_SOL + dat.cts_Ti_d->HEAT_COND_0_SOL)/2;
        double heat_cap  = (dat.cts_Al_d->HEAT_CAP_SOL    + dat.cts_Ti_d->HEAT_CAP_SOL)/2;
        double density   = (dat.cts_Al_d->SOL_DENSITY     + dat.cts_Ti_d->SOL_DENSITY)/2;
        double D = heat_cond / (heat_cap * density);
        double topD = D;
        double energy_source = 0;

        if (state_field[zm] == GAS){
            energy_source = heat_source[y*dat.Nx + x] - heat_loss(heat[center], dat.cts_Al_d);
            energy_source /= (heat_cap * density);
            topD = 0;
        }

        heat_next[center] = heat[center] + dat.tau * (
            ((D)   *(heat[xm] - heat[center]) + (D)*(heat[xp] - heat[center]))/(dat.h*dat.h) +
            ((0)   *(heat[ym] - heat[center]) + (0)*(heat[yp] - heat[center]))/(dat.h*dat.h) +
            ((topD)*(heat[zm] - heat[center]) + (D)*(heat[zp] - heat[center]))/(dat.h*dat.h) + energy_source/dat.h
        );
        // if (x == 200 && z == 1 && y == 1) {
        //     printf("D: %e, heat_next: %e, source: %e\n", D, heat_next[center  ], dat.tau * energy_source/dat.hz);
        // }
    }
}

__global__ void UpdateState(ST* state_field, double* heat) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int center = y*dat.Nx*dat.Nz + x*dat.Nz + z;
    if(x<dat.Nx && y<dat.Ny && z<dat.Nz && (state_field[center] == SOLID || state_field[center] == FLUID)) {
        if     (heat[center] <  dat.cts_Ti_d->TEMP_LIQ) {state_field[center] = SOLID;}
        else if(heat[center] >= dat.cts_Ti_d->TEMP_LIQ) {state_field[center] = FLUID;}
    }
}

__global__ void CalcHeatSource(double *heat_source, int step) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int center = y*dat.Nx + x;
    if (x < dat.Nx && y < dat.Ny) {
        double current_time = step * dat.tau;
        double beam_xpos = fmax(0.0, current_time * dat.beam_vel - 2e-6) + dat.beam_xstart * dat.h;
        double beam_ypos = dat.beam_ystart*dat.h;
        double sigma = dat.beam_radius / 2;
        // heat_source[x] = 2 * dat.beam_power / (M_PI * dat.beam_radius * dat.beam_radius) * exp(-2 * pow(current_time * dat.beam_vel + dat.beam_start * dat.h - x * dat.h, 2) / pow(dat.beam_radius, 2));
        heat_source[center] = dat.beam_power / pow(sigma * sqrt(2*M_PI), 2) * exp(-0.5*pow((x*dat.h - beam_xpos) / sigma, 2) - 0.5*pow((y*dat.h - beam_ypos) / sigma, 2));
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
        dat_host.init((double)temp, drop_rate);
        dat_host.time_stop = time_stop;
        
        int threadsPerBlock = 8;
        int blocksPerGridX = (dat_host.Nx + threadsPerBlock - 1) / threadsPerBlock;
        int blocksPerGridY = (dat_host.Ny + threadsPerBlock - 1) / threadsPerBlock;
        int blocksPerGridZ = (dat_host.Nz + threadsPerBlock - 1) / threadsPerBlock;
    
        dim3 blockShape(threadsPerBlock, threadsPerBlock, threadsPerBlock);
        dim3 gridShape(blocksPerGridX, blocksPerGridY, blocksPerGridZ);

        dim3 blockShapeSrc(threadsPerBlock, threadsPerBlock);
        dim3 gridShapeSrc(blocksPerGridX, blocksPerGridY);

        cudaMemcpyToSymbol(dat, &dat_host, sizeof(Data));
        std::cout << "copy data: " << cudaGetErrorString(cudaGetLastError()) << "\n";

        while (dat_host.tim < time_stop)
        {
            dat_host.step++;
            //std::cout << "temp " << temp << ", step " << dat_host.step << ": time = " << dat_host.tim << "; " << (dat_host.tim) / time_stop * 100 << "%" << std::endl;
            CalcHeatSource<<<gridShapeSrc, blockShapeSrc>>>(dat_host.heat_source_d, dat_host.step);
            cudaDeviceSynchronize();

            CalcHeatConduct<<<gridShape, blockShape>>>(dat_host.heat_d, dat_host.heat_next_d, dat_host.heat_source_d, dat_host.al_d, dat_host.ti_d, dat_host.state_field_d);
            std::swap(dat_host.heat_next_d, dat_host.heat_d);
            cudaDeviceSynchronize();

            UpdateState<<<gridShape, blockShape>>>(dat_host.state_field_d, dat_host.heat_d);
            cudaDeviceSynchronize();
            
            CalcDiffusion<<<gridShape, blockShape>>>(dat_host.al_d, dat_host.ti_d, dat_host.al_next_d, dat_host.ti_next_d, dat_host.state_field_d, dat_host.heat_d);
            std::swap(dat_host.al_next_d, dat_host.al_d);
            std::swap(dat_host.ti_next_d, dat_host.ti_d);            
            if (dat_host.step % dat_host.drop_rate == 0) {
                cudaDeviceSynchronize();
                dat_host.extract();
                std::ofstream output;
                output.open("data/dataT.txt");
                double maxT = 0;
                cudaDeviceSynchronize();
                double scale = 1e6;
                for (int x = 0; x < dat_host.Nx; x++)
                {
                    for (int z = 0; z < dat_host.Nz; z++)
                    {
                        int center = 1*dat_host.Nx*dat_host.Nz + x*dat_host.Nz + z;
                        if (dat_host.heat_h[center] > maxT) {maxT = dat_host.heat_h[center];}
                        output << x*dat_host.h*scale << " " << z*dat_host.hz*scale << " " << dat_host.heat_h[center] << " " << dat_host.al_h[center]/dat_host.al_0 << " " << dat_host.ti_h[center]/dat_host.ti_0 << " " << dat_host.state_field_h[center] << "\n";
                    }
                    output << "\n";
                }
                output.close();
                std::cout << "Tmax is: " << maxT << "\n";
                float highest_temp = 6000;
                float melt_temp = dat_host.cts_Ti_h->TEMP_MELT;
                float melt_frac = melt_temp / highest_temp;

                Gnuplot gp;
                gp << "load \"scriptT.gp\" \n";
                std::cout << "set output \"" << drop_dir << "/Tmap_" << std::setfill('0') << std::setw(7) << (int)dat_host.step << ".png\"\n";
                gp        << "set output \"" << drop_dir << "/Tmap_" << std::setfill('0') << std::setw(7) << (int)dat_host.step << ".png\"\n";
                gp << "set xrange [" << (fmax(0.0, dat_host.step*dat_host.tau * dat_host.beam_vel - 2e-6) + (dat_host.beam_xstart - 100) * dat_host.h)*scale << ":" << (fmax(0.0, dat_host.step*dat_host.tau * dat_host.beam_vel - 2e-6) + (dat_host.beam_xstart + 100) * dat_host.h)*scale << "]; set yrange [" << 3*dat_host.Nz*dat_host.hz/2*scale*1/3 << ":" << -dat_host.Nz*dat_host.hz/2*scale*0.25  <<"]\n";
                // gp << "set xrange [0:" << dat_host.Nx * dat_host.h << "]; set yrange [" << 0 << ":* ]\n";
                gp << "set palette defined (0 \"black\"," << melt_frac/2 << "\"blue\", " << melt_frac << "\"red\", " << melt_frac << "\"#00efc1\", " << (1+melt_frac)/2 << "\"#ffff46\", 1 \"grey90\") \n";
                gp << "set cbrange [0:" << highest_temp << "] \n";
                gp << "set title \"Al-Ti melt pool heat map at t = " << std::setprecision(3) << dat_host.tim << ", T_{max} = " << (int)maxT << "\"; ";
                gp << "splot \"data/dataT.txt\" u 1:2:3 with pm3d\n";

                std::cout << "set output \"" << drop_dir << "/Tmapzoomed_" << std::setfill('0') << std::setw(7) << (int)dat_host.step << ".png\"\n";
                gp        << "set output \"" << drop_dir << "/Tmapzoomed_" << std::setfill('0') << std::setw(7) << (int)dat_host.step << ".png\"\n";
                gp << "set xrange [0:" << dat_host.Nx * dat_host.h*scale << "]; set yrange [" << 3*dat_host.Nz*dat_host.hz/2*scale << ":" << -dat_host.Nz*dat_host.hz/2*scale  <<"]\n";
                // gp << "set xrange [0:" << dat_host.Nx * dat_host.h << "]; set yrange [" << 0 << ":* ]\n";
                gp << "set palette defined (0 \"black\"," << melt_frac/2 << "\"blue\", " << melt_frac << "\"red\", " << melt_frac << "\"#00efc1\", " << (1+melt_frac)/2 << "\"#ffff46\", 1 \"grey90\") \n";
                gp << "set cbrange [0:" << highest_temp << "] \n";
                gp << "set title \"Al-Ti melt pool heat map at t = " << std::setprecision(3) << dat_host.tim << ", T_{max} = " << (int)maxT << "\"; ";
                gp << "splot \"data/dataT.txt\" u 1:2:3 with pm3d\n";


                Gnuplot gpc;
                gpc << "load \"scriptC.gp\" \n";
                std::cout << "set output \"" << drop_dir << "/cmap_" << std::setfill('0') << std::setw(7) << (int)dat_host.step << ".png\"\n";
                gpc       << "set output \"" << drop_dir << "/cmap_" << std::setfill('0') << std::setw(7) << (int)dat_host.step << ".png\"\n ";
                gpc << "set xrange [0:" << dat_host.Nx * dat_host.h*scale << "]; set yrange [" << 3*dat_host.Nz*dat_host.hz/2*scale << ":" << -dat_host.Nz*dat_host.hz/2*scale  <<"]\n";
                gpc << "set title \"Al concentration in Al-Ti melt pool at t = " << std::setprecision(3) << dat_host.tim << "\"; ";
                gpc << "splot \"data/dataT.txt\" u 1:2:4 with pm3d\n";
                
                std::cout << "set output \"" << drop_dir << "/cmapzoomed_" << std::setfill('0') << std::setw(7) << (int)dat_host.step << ".png\"\n";
                gpc       << "set output \"" << drop_dir << "/cmapzoomed_" << std::setfill('0') << std::setw(7) << (int)dat_host.step << ".png\"\n ";
                gpc << "set xrange [" << (fmax(0.0, dat_host.step*dat_host.tau * dat_host.beam_vel - 2e-6) + (dat_host.beam_xstart - 40) * dat_host.h)*scale << ":" << (fmax(0.0, dat_host.step*dat_host.tau * dat_host.beam_vel - 2e-6) + (dat_host.beam_xstart + 40) * dat_host.h)*scale << "]; set yrange [" << 3*dat_host.Nz*dat_host.hz/2*scale*1/12 << ":" << -dat_host.Nz*dat_host.hz/2*scale*0.12  <<"]\n";
                gpc << "set title \"Al concentration in Al-Ti melt pool at t = " << std::setprecision(3) << dat_host.tim << "\"; ";
                gpc << "splot \"data/dataT.txt\" u 1:2:4 with pm3d\n";


                Gnuplot gps;
                gpc << "load \"scriptState.gp\" \n";
                std::cout << "set output \"" << drop_dir << "/statemap_" << std::setfill('0') << std::setw(7) << (int)dat_host.step << ".png\"\n";
                gpc       << "set output \"" << drop_dir << "/statemap_" << std::setfill('0') << std::setw(7) << (int)dat_host.step << ".png\"\n ";
                gpc << "set xrange [0:" << dat_host.Nx * dat_host.h*scale << "]; set yrange [" << 3*dat_host.Nz*dat_host.hz/2*scale << ":" << -dat_host.Nz*dat_host.hz/2*scale  <<"]\n";
                gpc << "set title \"State in Al-Ti melt pool at t = " << std::setprecision(3) << dat_host.tim << "\"; ";
                gpc << "splot \"data/dataT.txt\" u 1:2:6 with pm3d\n";
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
    sim_temp(3000.0, 3001.0, 200.0, 2e-3, 1000, "plots/maps_anim11_05");

    return 0;
}
