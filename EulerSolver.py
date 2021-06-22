# Solution to the compressible Euler Equations

import aerosandbox as asb
from construct2d import Construct2D
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

def main():


    naca = asb.Airfoil("naca6409")

    euler2d = CompressibleEuler2D(airfoil=naca,
                                  angle_of_attack=45)
    
    euler2d.generate_grid()
    euler2d.check_grid()
    euler2d.determine_inlet_outlet()
    euler2d.initialize_flow_variables()
    euler2d.set_constants()
    euler2d.flow_guess()
    euler2d.set_timestep()
    euler2d.runge_kutta()
    
    
    plt.figure(dpi=300)
    plt.contourf(euler2d.x,
                 euler2d.y,
                 euler2d.rovy,
                 50)
# =============================================================================
#     for i in range(euler2d.ni):
#         plt.plot(euler2d.x[i,:],euler2d.y[i,:],'b',lw=0.1) 
#     for j in range(euler2d.nj):
#         plt.plot(euler2d.x[:,j],euler2d.y[:,j],'b',lw=0.1) 
#     plt.axis("equal")
# =============================================================================
    
    return euler2d
    


class CompressibleEuler2D():
    """
    An analysis of the 2D compressible euler equations over any airfoil using
    a finite volume scheme. 
    
    The mesh can be generated from a p3d file or automatically by providing an airfoil. 
    
    The airfoil can either be a name of any airfoil in the UIUC database or 
    an Aerosandbox airfoil object
    """
    def __init__(self,
                 airfoil=None,
                 p3d_filepath=None,
                 angle_of_attack=0,
                 supersonic_inlet=False,
                 facsec=0, #adam bashforth coefficient
                 fcorr = 0.9 # Amount of artificial vscosity to be canceled out by deferred corrections (1 = 100% and therefore not stable) 
                 ):
        if isinstance(airfoil,str):
            self.airfoil = asb.Airfoil(airfoil)
        else:
            self.airfoil = airfoil
            
        self.angle_of_attack = np.deg2rad(angle_of_attack)
        self.p3d_filepath = p3d_filepath
        self.supersonic_inlet = supersonic_inlet
        self.facsec = facsec
        self.fcorr = fcorr
        
        if airfoil == None and p3d_filepath != None:
            self.import_p3d()
        elif airfoil != None and p3d_filepath != None:
            self.import_p3d()
        elif airfoil != None and p3d_filepath == None:
            self.run_construct2d()
        else: 
            raise Exception("You must provide either an airfoil or a p3d file")
            
            
        
    def run_construct2d(self):
        self.c2d = Construct2D(airfoil=self.airfoil,
                      working_directory=r"C:\Users\ag458\Documents\Projects\AeroProjects\Compressible-Euler-Solver")
        self.c2d.change_default_parameters(num_points_on_surface=200,
                                  farfield_radius=5,
                                  num_points_normal_direction=20,
                                  viscous_y_plus=30)
        self.c2d.run_construct2D()
        
        self.ni = self.c2d.imax
        self.nj = self.c2d.jmax
        self.nk = self.c2d.kmax
        self.x = self.c2d.x 
        self.y = self.c2d.y
        self.threed = self.c2d.threed
    
        
    def inspect_grid(self): 
            self.c2d.postpycess()
        
    def import_p3d(self):
        # Read imax, jmax
        # 3D grid specifies number of blocks on top line
        fname = self.p3d_filepath
        f = open(fname)
        line1 = f.readline()
        flag = len(line1.split())
        if flag == 1:
          threed = True
        else:
          threed = False
          
        if threed:
          line1 = f.readline()
          imax, kmax, jmax = [int(x) for x in line1.split()]
        else:
          imax, jmax = [int(x) for x in line1.split()]
          kmax = 1
          
        # Read geometry data
        x = np.zeros((imax,jmax))
        y = np.zeros((imax,jmax))
        if threed:
          for j in range(0, jmax):
              for k in range(0, kmax):
                  for i in range(0, imax):
                      x[i,j] = float(f.readline())
          for j in range(0, jmax):
              for k in range(0, kmax):
                  for i in range(0, imax):
                      dummy = float(f.readline())
          for j in range(0, jmax):
              for k in range(0, kmax):
                  for i in range(0, imax):
                      y[i,j] = float(f.readline())
        else:
          for j in range(0, jmax):
              for i in range(0, imax):
                  x[i,j] = float(f.readline())
          
          for j in range(0, jmax):
              for i in range(0, imax):
                  y[i,j] = float(f.readline())
          
        # Print message
        print('Successfully read grid file ' + fname)
          
        # Close the file
        f.close
         
    
        self.ni = imax
        self.nj = jmax
        self.nk = kmax
        self.x = x 
        self.y = y
        self.threed = threed
        
        
        
    def generate_grid(self):
        ni,nj = self.ni,self.nj
        x = self.x
        y = self.y
        area = np.zeros([ni-1, nj-1])
        dliy = np.zeros([ni, nj-1])
        dlix = np.zeros([ni, nj-1])
        dljy = np.zeros([ni-1, nj])
        dljx = np.zeros([ni-1, nj])
                   
    
        #Calculate cell areas
        for i in range(ni-1):
            for j in range(nj-1):
                ax = x[i,j+1] - x[i+1,j]
                ay = y[i,j+1] - y[i+1,j]
                a = ax,ay
                bx = x[i+1,j+1]-x[i,j]
                by = y[i+1,j+1]-y[i,j]
                b = bx,by
                area[i,j] = 0.5*np.cross(b,a)
    
                if area[i,j] < 0:
                    print('ERROR: Negative area exists in the mesh')
    
                
        #Calculate components of the normal vectors of each cell face. 
        dmin = 100000
        for i in range(ni):
            for j in range(nj-1):
                dliy[i,j] = x[i,j]- x[i,j+1]
                dlix[i,j] = y[i,j+1]-y[i,j]
                test = np.sqrt(dliy[i,j]**2 + dlix[i,j]**2)
                if test < dmin and test != 0:
                    dmin = test
                
    
    
        for j in range(nj):
            for i in range(ni-1):
                dljy[i,j] =  x[i+1,j]-x[i,j]
                dljx[i,j] =  y[i,j]- y[i+1,j]
                test = np.sqrt(dljy[i,j]**2 + dljx[i,j]**2)
                if test < dmin and test != 0:
                    dmin = test
        print('Overall minimum element size: ',round(dmin,5))
        
        self.area = area 
        self.dliy = np.copy(dliy)
        self.dlix = np.copy(dlix)
        self.dljy = np.copy(dljy)
        self.dljx = np.copy(dljx)
        self.dmin = np.copy(dmin)
    
    
    def determine_inlet_outlet(self):
        self.is_inlet = np.full(self.ni-1,True)
        angle_of_attack_vector = np.array([np.cos(self.angle_of_attack),
                                           np.sin(self.angle_of_attack)])
        for i in range(self.ni-1):
            normal_vector = np.array([self.dljx[i,-1],self.dljy[i,-1]])
            if np.dot(normal_vector,angle_of_attack_vector) > 0:
                self.is_inlet[i] = False
        
        self.inlet_panels = np.where(self.is_inlet)[0]
        self.inlet_nodes = np.where(self.is_inlet)[0]
        self.outlet_panels = np.where(self.is_inlet==False) [0]
        self.outlet_nodes = np.append(np.where(self.is_inlet==False)[0],np.max(self.outlet_panels)+1)
        
# =============================================================================
#         plt.figure(dpi=300)
#         plt.scatter(self.x[:,-1][self.inlet_nodes],self.y[:,-1][self.inlet_nodes],10,color="blue")
#         plt.scatter(self.x[:,-1][self.outlet_nodes],self.y[:,-1][self.outlet_nodes],2,color="red")
#         plt.axis("equal")
# =============================================================================
                        
            
    def check_grid(self):
    
        # This function checks that each of the cells has 0 sum between all normal vectors
        small = 0.000001
    
        for i in range(self.ni-1):
            for j in range(self.nj-1):
                xSum = self.dlix[i,j]-self.dlix[i+1,j]+self.dljx[i,j]-self.dljx[i,j+1]
                ySum = self.dliy[i,j]-self.dliy[i+1,j]+self.dljy[i,j]-self.dljy[i,j+1]
                
                if xSum > small or ySum > small:
                    print('ERROR: Cell face vector sum is not 0')
            
        
    def initialize_flow_variables(self):
        ni = self.ni
        nj  = self.nj
        self.rovx = np.zeros([ni,nj])
        self.rovy = np.zeros([ni,nj])
        self.ro = np.zeros([ni,nj])
        self.roe = np.zeros([ni,nj])
        self.vx = np.zeros([ni,nj])
        self.vy = np.zeros([ni,nj])
        self.pstatic = np.zeros([ni,nj])
        self.hstag = np.zeros([ni,nj])
        self.tstatic = np.zeros([ni,nj])
    
        
        self.delro = np.zeros([ni-1,nj-1])
        self.delroe = np.zeros([ni-1,nj-1])
        self.delrovx = np.zeros([ni-1,nj-1])
        self.delrovy = np.zeros([ni-1,nj-1])
        self.ro_inc = np.zeros([ni,nj])
        self.roe_inc = np.zeros([ni,nj])
        self.rovx_inc = np.zeros([ni,nj])
        self.rovy_inc = np.zeros([ni,nj])
        self.roinlet = np.zeros([ni])
        self.residuals = np.zeros([1,5])
        
        self.corr_ro = np.zeros([ni,nj])
        self.corr_roe = np.zeros([ni,nj])
        self.corr_rovx = np.zeros([ni,nj])
        self.corr_rovy = np.zeros([ni,nj])   
    
        
        self.fluxi_mass = np.zeros([ni, nj-1])
        self.fluxj_mass = np.zeros([ni-1,nj])
        self.fluxi_xmom = np.zeros([ni, nj-1])
        self.fluxj_xmom = np.zeros([ni-1,nj])
        self.fluxi_ymom = np.zeros([ni, nj-1])
        self.fluxj_ymom = np.zeros([ni-1,nj])
        self.fluxi_enth = np.zeros([ni,nj-1])
        self.fluxj_enth = np.zeros([ni-1,nj])
        
    def set_constants(self):
        self.gamma = 1.4
        self.rgas = 287.1  
        self.pstagin = 100000
        self.tstagin = 300
        self.pdown = 8500
        self.cfl = 1.
        self.smooth_fac = 0.5 * self.cfl
        self.conlim = 0.0001 * self.cfl
        self.nsteps = 10000
        self.nrkut_max = 4
        
        
        self.emax       = 1000000.0
        self.eavg       = self.emax
        self.cp         = self.rgas*self.gamma/(self.gamma-1.0)
        self.cv         = self.cp/self.gamma
        self.fga        = (self.gamma - 1.0)/self.gamma
               
        self.roin      =  self.pstagin/self.rgas/self.tstagin
        self.ref_ro    =  (self.pstagin-self.pdown)/self.rgas/self.tstagin
        self.ref_t     =  self.tstagin*(self.pdown/self.pstagin)**self.fga
        self.ref_v     =  np.sqrt(2*self.cp*(self.tstagin-self.ref_t))
        self.ref_rovx  =  self.roin*self.ref_v
        self.ref_rovy  =  self.ref_rovx
        self.ref_roe   =  self.roin*self.cv*(self.tstagin-self.ref_t)
        self.ncells = (self.ni-1)*(self.nj-1)
        
        
    def flow_guess(self):
        ro_guess = np.zeros([self.ni])
        v_guess = np.zeros([self.ni])
        t_guess = np.zeros([self.ni])
        p_guess = np.zeros([self.ni])
        aflow = np.zeros([self.ni])
    
        #In this subroutine we make an initial guess of the primary variables
        #i.e.  ro, rovx, rovy and roe. Isentropic flow assumptions are used
    
        for i in range(self.ni):
            for j in range(self.nj-1):
                aflow[i] = aflow[i] + np.sqrt(self.dlix[i,j]**2 + self.dliy[i,j]**2)
    
    
        tdown = self.tstagin*(self.pdown/self.pstagin)**((self.gamma-1)/self.gamma)
        ro_guess[self.ni-1] = self.pdown/(self.rgas*tdown)
        v_guess[self.ni-1] = np.sqrt(2.0*self.cp*(self.tstagin-tdown))
        mflow = ro_guess[self.ni-1]*v_guess[self.ni-1]*aflow[self.ni-1]
        
    
    
        
        if not self.supersonic_inlet:
            
            machlim = 1.0
            tlim = self.tstagin/(1.0 + 0.5*(self.gamma-1.0)*machlim**2)
    
    
            for i in range(self.ni):
                ro_guess[i] = ro_guess[self.ni-1]
                v_guess[i] = mflow/(ro_guess[i]*aflow[i])
                t_guess[i] = self.tstagin - (v_guess[i]**2)/(2.0*self.cp)
    
                if t_guess[i] < tlim:
                    t_guess[i] = tlim
    
                p_guess[i] = self.pstagin*((t_guess[i]/self.tstagin)**(self.gamma/(self.gamma-1)))
                ro_guess[i] = p_guess[i]/(self.rgas*t_guess[i])
                v_guess[i] = mflow/(ro_guess[i]*aflow[i])
                
            
            for j in range(self.nj):
                for i in range(self.ni-1):
                    
                    dx = self.x[i+1,j] - self.x[i,j]
                    dy = self.y[i+1,j] - self.y[i,j]
                    dxy = np.sqrt(dx**2 +dy**2)
                    vx_guess =v_guess[i]*dx/dxy
                    vy_guess =v_guess[i]*dy/dxy
                    self.ro[i,j] = ro_guess[i]
                    self.rovx[i,j] =  self.ro[i,j]*vx_guess
                    self.rovy[i,j] =  self.ro[i,j]*vy_guess
                    e = self.cv*t_guess[i]+0.5*v_guess[i]**2
                    self.roe[i,j] = self.ro[i,j]*e
                    
                self.rovx[self.ni-1,j] = self.rovx[self.ni-2,j]
                self.rovy[self.ni-1,j] = self.rovy[self.ni-2,j]
                self.ro[self.ni-1,j] = self.ro[self.ni-2,j]
                self.roe[self.ni-1,j] = self.roe[self.ni-2,j]
                
        else:
                
            ro_guess[:] = ro_guess[self.ni-1]
            v_guess[:] = v_guess[self.ni-1]
            t_guess = self.tstagin - (np.square(v_guess))/(2.0*self.cp)
            
    
            for j in range(self.nj):
                for i in range(self.ni-1):
                    dx = self.x[i+1,j] - self.x[i,j]
                    dy = self.y[i+1,j] - self.y[i,j]
                    dxy = np.sqrt(dx**2 +dy**2)
                    vx_guess =v_guess[i]*dx/dxy
                    vy_guess =v_guess[i]*dy/dxy
                    self.ro[i,j] = ro_guess[i]
                    self.rovx[i,j] =  self.ro[i,j]*vx_guess
                    self.rovy[i,j] =  self.ro[i,j]*vy_guess
                    e = self.cv*t_guess[i]+0.5*v_guess[i]**2
                    self.roe[i,j] = self.ro[i,j]*e
                    
                self.rovx[self.ni-1,j] = self.rovx[self.ni-2,j]
                self.rovy[self.ni-1,j] = self.rovy[self.ni-2,j]
                self.ro[self.ni-1,j] = self.ro[self.ni-2,j]
                self.roe[self.ni-1,j] = self.roe[self.ni-2,j]
        
                
            self.rovx[self.ni-1,j] = self.rovx[self.ni-2,j]
            self.rovy[self.ni-1,j] = self.rovy[self.ni-2,j]
            self.ro[self.ni-1,j] = self.ro[self.ni-2,j]
            self.roe[self.ni-1,j] = self.roe[self.ni-2,j]
                
    
            for j in range(self.nj):
                for i in range(self.ni-1):
                
                    dx = self.x[i+1,j] - self.x[i,j]
                    dy = self.y[i+1,j] - self.y[i,j]
                    dxy = np.sqrt(dx**2 +dy**2)
                    vx_guess =v_guess[i]*dx/dxy
                    vy_guess =v_guess[i]*dy/dxy
                    self.ro[i,j] = ro_guess[i]
                    self.rovx[i,j] =  self.ro[i,j]*vx_guess
                    self.rovy[i,j] =  self.ro[i,j]*vy_guess
                    e = self.cv*t_guess[i]+0.5*v_guess[i]**2
                    self.roe[i,j] = self.ro[i,j]*e
                    
                self.rovx[self.ni-1,j] = self.rovx[self.ni-2,j]
                self.rovy[self.ni-1,j] = self.rovy[self.ni-2,j]
                self.ro[self.ni-1,j] = self.ro[self.ni-2,j]
                self.roe[self.ni-1,j] = self.roe[self.ni-2,j]
        
        self.ro_old = np.copy(self.ro)
        self.rovx_old = np.copy(self.rovx)
        self.rovy_old = np.copy(self.rovy)
        self.roe_old = np.copy(self.roe)
        
        
        self.rovx = np.full((self.ni,self.nj),5)
        self.rovy = np.full((self.ni,self.nj),5)
        self.ro = np.full((self.ni,self.nj),5)
        self.roe = np.full((self.ni,self.nj),5)
        
        
        
        
    
    def set_timestep(self):
        #This subroutine sets the length of the time step based on the
        #stagnation speed of sound "astag" and the minimum length scale
        #of any element, "dmin". The timestep must be called "deltat"
    
        #An assumption that the maximum flow speed will be equal to "astag"
        #is also made. This will be pessimistic for subsonic flows
        #but may be optimistic for supersonic flows. In the latter case the
        #length of the time step as determined by "cfl" may need to be reduced.
    
        astag  = np.sqrt(self.gamma*self.rgas*self.tstagin)
        umax   = astag
        self.deltat = self.cfl*self.dmin/(umax+astag)  
        
    def set_others(self):
        self.vx,self.vy,self.tstatic,self.pstatic,self.hstag = \
        set_non_primitives(self.ro,self.roe,self.rovx,self.rovy,self.tstagin,
                           self.cp,self.rgas,self.gamma,self.vx,self.vy,
                           self.tstatic,self.pstatic,self.hstag,self.ni,self.nj)
        
    
    def apply_boundary_conditions(self):
        self.vx,self.vy,self.rovx,self.rovy,self.roe,self.hstag,self.pstatic,self.roinlet,self.ro = \
            apply_bconds(self.roinlet,self.vx,self.vy,self.cp,self.cv,self.angle_of_attack,self.hstag,self.pstagin,\
            self.rgas,self.tstagin,self.gamma,self.ni,self.nj,self.nstep,self.ro,self.rovx,self.rovy,self.roe,self.pstatic,self.pdown,self.supersonic_inlet,self.inlet_nodes,self.outlet_nodes)
                            
    def set_fluxes(self):
        self.fluxi_mass,self.fluxj_mass,self.fluxi_xmom,self.fluxj_xmom,self.fluxi_ymom,self.fluxj_ymom,\
                        self.fluxi_enth,self.fluxj_enth,self.flow = set_all_fluxes(self.fluxi_mass,self.fluxj_mass,self.fluxi_xmom,\
                        self.fluxj_xmom,self.fluxi_ymom,self.fluxj_ymom,self.fluxi_enth,self.fluxj_enth,self.ni,self.nj,self.rovx,self.rovy, \
                        self.dlix,self.dliy,self.dljx,self.dljy,self.vx,self.vy,self.pstatic,self.hstag)            
    
    def sum_fluxes(self):
        self.delro,self.ro_inc = sum_all_fluxes(self.fluxi_mass,self.fluxj_mass,self.delro,self.ro_inc,self.ni,self.nj,self.deltat,self.area,self.frkut,self.facsec)
        self.delroe,self.roe_inc = sum_all_fluxes(self.fluxi_enth,self.fluxj_enth,self.delroe,self.roe_inc,self.ni,self.nj,self.deltat,self.area,self.frkut,self.facsec)
        self.delrovx,self.rovx_inc = sum_all_fluxes(self.fluxi_xmom,self.fluxj_xmom,self.delrovx,self.rovx_inc,self.ni,self.nj,self.deltat,self.area,self.frkut,self.facsec)
        self.delrovy,self.rovy_inc = sum_all_fluxes(self.fluxi_ymom,self.fluxj_ymom,self.delrovy,self.rovy_inc,self.ni,self.nj,self.deltat,self.area,self.frkut,self.facsec)

    def smooth(self):
        self.ro,self.corr_ro = smooth_all(self.ro,self.ni,self.nj,self.smooth_fac,self.corr_ro,self.fcorr)
        self.roe,self.corr_roe = smooth_all(self.roe,self.ni,self.nj,self.smooth_fac,self.corr_roe,self.fcorr)
        self.rovx,self.corr_rovx = smooth_all(self.rovx,self.ni,self.nj,self.smooth_fac,self.corr_rovx,self.fcorr)
        self.rovy,self.corr_rovy = smooth_all(self.rovy,self.ni,self.nj,self.smooth_fac,self.corr_rovy,self.fcorr)
    
    def check_convergence(self):
        self.residuals,self.emax,self.eavg,self.ro_old,self.rovx_old,self.rovy_old,self.roe_old = check_conv(self.residuals,self.flow,self.ncells,self.ref_ro,self.ref_rovx,\
            self.ref_rovy,self.ref_roe,self.emax,self.eavg,self.nstep,self.ni,self.nj,self.ro,self.roe,self.rovx,self.rovy,self.ro_old,self.roe_old,self.rovx_old,self.rovy_old)
                
                
    
    def runge_kutta(self):
        
        
        for self.nstep in range(self.nsteps):
  
            ro_start = np.copy(self.ro)
            roe_start = np.copy(self.roe)
            rovx_start = np.copy(self.rovx)
            rovy_start = np.copy(self.rovy)
        
            for nrkut in range(self.nrkut_max):
                self.frkut = 1./(self.nrkut_max-nrkut)
            
                self.set_others()
                self.apply_boundary_conditions()
                self.set_fluxes()
                self.sum_fluxes()
                
                self.ro = np.add(ro_start,self.ro_inc)
                self.roe = np.add(roe_start,self.roe_inc)
                self.rovx = np.add(rovx_start,self.rovx_inc)
                self.rovy = np.add(rovy_start,self.rovy_inc)
            
            self.smooth()
            
            if (self.nstep) % 1 == 0:
                self.check_convergence()
            
            if self.emax < self.conlim and self.eavg < (0.5*self.conlim):
                print('Calculation converged in ',self.nstep,' iterations')
                print('To a convergence limit of ',self.conlim)
                break
        
        residuals = np.array(self.residuals)
    
    
    
        #output(x,y,ni,nj,ro,rovx,rovy,roe,pstatic,residuals)
    
    
        
        plt.figure()
        labels =['ro','rovx','rovy','roe']
        plt.semilogy(residuals[5:,0],residuals[5:,1:])
        plt.legend(labels)
        plt.show()
            
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
             
        
@njit
def set_non_primitives(ro,roe,rovx,rovy,tstagin,cp,rgas,gamma,vx,vy,tstatic,pstatic,hstag,ni,nj):
    for i in range(ni):
        for j in range(nj):
            vx[i,j] = rovx[i,j]/ro[i,j]
            vy[i,j] = rovy[i,j]/ro[i,j]
            v_square = vx[i,j]**2 +vy[i,j]**2
            tstatic[i,j] = tstagin - 0.5*v_square/cp
            pstatic[i,j] = (gamma-1)*(roe[i,j]-0.5*ro[i,j]*v_square)
            hstag[i,j] = (roe[i,j]+pstatic[i,j])/ro[i,j]
    
    return vx,vy,tstatic,pstatic,hstag   

@njit
def apply_bconds(roinlet,vx,vy,cp,cv,angle_of_attack,hstag,pstagin,rgas,tstagin,gamma,ni,nj,nstep,ro,rovx,rovy,roe,pstatic,pdown,supersonic_inlet,inlet_nodes,outlet_nodes):
    # Here the boundary conditions are set. At the inlet, stagnation pressure
    # and stagnation temperature along with an inlet flow angle. At the outlet
    # static pressure is specified. Subsonic flow assumption has been made.

    
    rfin     = 0.25
    rfin1    = 1.0-rfin
    rostagin = pstagin/(rgas*tstagin)
    gm1      = gamma - 1.0
    
   
    
    for i in inlet_nodes:
        if nstep == 0:
            roinlet[i] = ro[i,-1]
        else:
            roinlet[i] = rfin*ro[i,-1]+rfin1*roinlet[i]

        if roinlet[i]>0.9999*rostagin:
            roinlet[i] = 0.9999*rostagin
            
        if supersonic_inlet:
            roinlet[i] = rostagin*(pdown/pstagin)**(1./gamma)
            
        tstat = tstagin*(roinlet[i]/rostagin)**gm1
        pstatic[i,-1] = rgas*tstat*roinlet[i]
        vel = np.sqrt(2.0*cp*(tstagin-tstat))
        vx[i,-1] = vel*np.cos(angle_of_attack)
        vy[i,-1] = vel*np.sin(angle_of_attack)
        
        ro[i,-1] = roinlet[i]
        rovx[i,-1] = vx[i,-1]*roinlet[i]
        rovy[i,-1] = vy[i,-1]*roinlet[i]
        
        
        eke = cv*tstat+0.5*vel**2
        roe[i,-1] = roinlet[i]*eke
        hstag[i,-1] = (roe[i,-1]+pstatic[i,-1])/roinlet[i]
        
    if not supersonic_inlet:
        for j in outlet_nodes:
            pstatic[j,-1] = pdown
        

    return vx,vy,rovx,rovy,roe,hstag,pstatic,roinlet,ro
        


@njit
def set_all_fluxes(fluxi_mass,fluxj_mass,fluxi_xmom,fluxj_xmom,fluxi_ymom,fluxj_ymom,\
        fluxi_enth,fluxj_enth,ni,nj,rovx,rovy,dlix,dliy,dljx,dljy,vx,vy,pstatic,hstag):
    
    #This routine sets all the fluxes using control volume equations.

    flow = np.zeros_like(rovx[:,0])

    for j in range(nj-1):
        for i in range(ni-1):
            
        
            fluxi_mass[i,j] = 0.5*((rovx[i,j]+rovx[i,j+1])*dlix[i,j]+ \
                                   (rovy[i,j]+rovy[i,j+1])*dliy[i,j])
            flow[i] = flow[i] + fluxi_mass[i,j]
            
            fluxi_xmom[i,j] = 0.5*(fluxi_mass[i,j]*(vx[i,j]+vx[i,j+1]) + \
                                   (pstatic[i,j]+pstatic[i,j+1])*dlix[i,j])

            fluxi_ymom[i,j] = 0.5*(fluxi_mass[i,j]*(vy[i,j]+vy[i,j+1]) + \
                                   (pstatic[i,j]+pstatic[i,j+1])*dliy[i,j])
                
            fluxi_enth[i,j] = 0.5*(fluxi_mass[i,j]*(hstag[i,j]+hstag[i,j+1]))
        
        
        fluxi_mass[ni-1,j] = fluxi_mass[0,j]
        
        flow[ni-1] = flow[0] 
        
        fluxi_xmom[ni-1,j] = fluxi_xmom[0,j]

        fluxi_ymom[ni-1,j] = fluxi_ymom[0,j]
            
        fluxi_enth[ni-1,j] = fluxi_enth[0,j]
        
        
    
    
            
    for i in range(ni-1):
        
        ip1 = i+1
        if i == ni-1:
            ip1 = 1


        
        for j in range(1,nj):
    
            fluxj_mass[i,j] = 0.5*((rovx[i,j]+rovx[ip1,j])*dljx[i,j] + \
                                   (rovy[i,j]+rovy[ip1,j])*dljy[i,j])
            

    for i in range(ni-1):
        fluxj_mass[i,0] = 0.0
        
        


    for i in range(ni-1):
        ip1 = i+1
        if i == ni-1:
            ip1 = 1
       
        for j in range(nj):
            fluxj_xmom[i,j] = 0.5*(fluxj_mass[i,j]*(vx[i,j]+vx[ip1,j]) + \
                                   (pstatic[i,j]+pstatic[ip1,j])*dljx[i,j])
                
            fluxj_ymom[i,j] = 0.5*(fluxj_mass[i,j]*(vy[i,j]+vy[ip1,j]) + \
                                   (pstatic[i,j]+pstatic[ip1,j])*dljy[i,j])
                
            fluxj_enth[i,j] = 0.5*(fluxj_mass[i,j]*(hstag[i,j]+hstag[ip1,j]))



    return fluxi_mass,fluxj_mass,fluxi_xmom,fluxj_xmom,fluxi_ymom,fluxj_ymom,fluxi_enth,fluxj_enth,flow



@njit
def sum_all_fluxes(iflux,jflux,delprop,prop_inc,ni,nj,deltat,area,frkut,facsec):

#This subroutine sums the fluxes for each element, calculates the changes 
#in the variable "prop" appropriate to a cell (delprop) and distributes them to the
#four corners of the element (stored in prop_inc).
    store = np.zeros_like(delprop)
    previous = np.copy(delprop)
    fcs1 = 1+facsec
    

    #Calculate fluxes of cells
    for i in range(ni-1):
        for j in range(nj-1):
            delprop[i,j] = frkut*(deltat/area[i,j])*(iflux[i,j]-iflux[i+1,j]+ \
                                                     jflux[i,j]-jflux[i,j+1])
            store[i,j] =  delprop[i,j]   
            
            delprop[i,j] =  fcs1*delprop[i,j] -  frkut*facsec*previous[i,j]
            
        
    
    #Distribute to all interior nodes 
    for i in range(1,ni-1):
        for j in range(1,nj-1):
            prop_inc[i,j] = 0.25*(delprop[i,j]+delprop[i-1,j]+delprop[i,j-1]+ \
                                  delprop[i-1,j-1])

    #Distribute to upper and lower boundary nodes (except corners)
    for i in range(1,ni-1):
        prop_inc[i,0] = 0.5*(delprop[i-1,0]+delprop[i,0])

        prop_inc[i,nj-1] = 0.5*(delprop[i-1,nj-2]+delprop[i,nj-2])

    #Distribute to Inlet and Outlet
    for j in range(1,nj-1):
        prop_inc[ni-1,j] = 0.5*(delprop[ni-2,j]+delprop[ni-2,j-1])

        prop_inc[0,j] = 0.5*(delprop[0,j]+delprop[0,j-1])

    #Distribute to corners
    prop_inc[0,0] = delprop[0,0]

    prop_inc[0,nj-1] = delprop[0,nj-2]

    prop_inc[ni-1,0] = delprop[ni-2,0]

    prop_inc[ni-1,nj-1] = delprop[ni-2,nj-2]
    
    delprop = np.copy(store)
    
    return delprop,prop_inc




@njit
def smooth_all(prop,ni,nj,smooth_fac,corr_prop,fcorr):
    
    # THIS IS WHERE YOU LEFT OFF, SMOOTH THEN AVERAGE

    store = np.zeros_like(prop)
    
    sf = smooth_fac
    sfm1 = 1-sf
    
    
    
    for j in range(1,nj):
            jp1 = j+1
    
            if j == nj-1:
                jp1 = nj-1
                
            jm1 = j-1
    
            for i in range(1,ni-1):
                avg = 0.25*(prop[i,jp1] + prop[i,jm1] + prop[i+1,j] + prop[i-1,j])
                corrnew = fcorr*(prop[i,j]-avg)
                corr_prop[i,j] = 0.99*corr_prop[i,j]+0.01*corrnew
                store[i,j] = sfm1*prop[i,j] + sf*(avg+corr_prop[i,j])
                
    
            avg_1 = (prop[0,jm1]+prop[0,jp1]+ 2.0*prop[1,j]- prop[2,j])/3.0
            corrnew1 = fcorr*(prop[0,j]-avg_1)
            corr_prop[0,j] = 0.99*corr_prop[0,j]+0.01*corrnew1
        
            
            avg_ni = (prop[ni-1,jm1]+prop[ni-1,jp1]+ 2.0*prop[ni-2,j]- prop[ni-3,j])/3.0
            corrnewni = fcorr*(prop[ni-1,j]-avg_ni)
            corr_prop[ni-1,j] = 0.99*corr_prop[ni-1,j]+0.01*corrnewni
            
            store[0,j] = sfm1*prop[0,j]+sf*(avg_1+corr_prop[0,j])
            store[ni-1,j] = sfm1*prop[ni-1,j]+sf*(avg_ni+corr_prop[ni-1,j])
            
    for i in range(ni):
        ip1 = i+1

        if i == ni-1:
            ip1 = ni-1
            
        im1 = i-1

        if i == 0:
            im1 = 0
            
        avg_foil = (prop[im1,0]+prop[ip1,0]+ 2.0*prop[i,1]- prop[i,2])/3.0
        corrnewfoil = fcorr*(prop[i,0]-avg_foil)
        corr_prop[i,0] = 0.99*corr_prop[i,0]+0.01*corrnewfoil
        store[i,0] = sfm1*prop[i,0] + sf*(avg_foil+corr_prop[i,0])
        
    
    average = (store[0,:] + store[ni-1,:])/2 
    
    store[0,:]  = average
    store[ni-1,:]  = average
    
    
    prop = np.copy(store)
    
    return prop,corr_prop
        
        
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    for i in range(ni):
        ip1 = i+1

        if i == ni-1:
            ip1 = ni-1
            
        im1 = i-1

        if i == 0:
            im1 = 0

        for j in range(1,nj-1):
            avg = 0.25*(prop[ip1,j] + prop[im1,j] + prop[i,j-1] + prop[i,j+1])
            corrnew = fcorr*(prop[i,j]-avg)
            corr_prop[i,j] = 0.99*corr_prop[i,j]+0.01*corrnew
            store[i,j] = sfm1*prop[i,j] + sf*(avg+corr_prop[i,j])

        avg_1 = (prop[im1,0]+prop[ip1,0]+ 2.0*prop[i,1]- prop[i,2])/3.0
        corrnew1 = fcorr*(prop[i,0]-avg_1)
        corr_prop[i,0] = 0.99*corr_prop[i,0]+0.01*corrnew1
        
        avg_nj = (prop[im1,nj-1]+prop[ip1,nj-1]+ 2.0*prop[i,nj-2]- prop[i,nj-3])/3.0
        corrnewnj = fcorr*(prop[i,nj-1]-avg_nj)
        corr_prop[i,nj-1] = 0.99*corr_prop[i,nj-1]+0.01*corrnewnj
        
        store[i,0] = sfm1*prop[i,0]+sf*(avg_1+corr_prop[i,0])
        store[i,nj-1] = sfm1*prop[i,nj-1]+sf*(avg_nj+corr_prop[i,nj-1])

    prop = np.copy(store)

    return prop,corr_prop

@njit
def check_conv(residuals,flow,ncells,ref_ro,ref_rovx,ref_rovy,ref_roe,emax,eavg,nstep,ni,nj,ro,roe,rovx,rovy,ro_old,roe_old,rovx_old,rovy_old):
    
    delromax   = 0.0
    delrovxmax = 0.0
    delrovymax = 0.0
    delroemax  = 0.0
    delroavg   = 0.0
    delrovxavg = 0.0
    delrovyavg = 0.0
    delroeavg  = 0.0
    imax = 0
    jmax = 0

    #"imax,jmax" is the grid point where the change in rovx is a max.

    for i in range(ni):
        for j in range(nj):
            
          delta = abs(ro[i,j] - ro_old[i,j])
          if delta > delromax:
              delromax = delta

          delroavg = delroavg + delta
     
          delta = abs(rovx[i,j]-rovx_old[i,j])
          if delta > delrovxmax:
             delrovxmax = delta
             imax = i
             jmax = j
          
          delrovxavg = delrovxavg + delta
     
          delta = abs(rovy[i,j] - rovy_old[i,j])
          if delta > delrovymax:
              delrovymax = delta
              
          delrovyavg = delrovyavg + delta
     
          delta = abs(roe[i,j] - roe_old[i,j])
          if delta > delroemax:
              delroemax = delta
              
          delroeavg = delroeavg + delta

    delroavg   = delroavg/float(ncells)/ref_ro
    delrovxavg = delrovxavg/float(ncells)/ref_rovx
    delrovyavg = delrovyavg/float(ncells)/ref_rovy
    delroeavg  = delroeavg/float(ncells)/ref_roe
    delrovxmax = delrovxmax/ref_rovx
    delrovymax = delrovymax/ref_rovy


    emax = max(delrovxmax,delrovymax)
    eavg = max(delrovxavg,delrovyavg)

    ro_old = np.copy(ro)
    rovx_old = np.copy(rovx)
    rovy_old = np.copy(rovy)
    roe_old = np.copy(roe)

    
    row = np.array([[nstep,delroavg,delrovxavg,delrovyavg,delroeavg]])
    residuals = np.concatenate((residuals,row),axis=0)
    flow_ratio = flow[ni-1]/flow[0]
        
    if np.mod(nstep,5) == 0 :
        print('time step number: ', nstep)
        print('emax =',round(emax,9),'imax =',imax,'jmax =',jmax,'eavg =',round(eavg,9))
        print('Inlet Flow:',round(flow[0],4),'Outlet to Inlet Flow Ratio:',round(flow_ratio,4))
        print('\n')

    return residuals,emax,eavg,ro_old,rovx_old,rovy_old,roe_old

    


if __name__ == "__main__":
    euler2d = main()