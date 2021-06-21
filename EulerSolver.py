# Solution to the compressible Euler Equations

import aerosandbox as asb
from construct2d import Construct2D
import numpy as np
import matplotlib.pyplot as plt

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
    euler2d.runge_kutta()
    
    
    plt.figure(dpi=300)
    plt.contourf(euler2d.x,
                 euler2d.y,
                 euler2d.rovx,
                 50)
    plt.axis("equal")
    
    
    


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
                 supersonic_inlet=False):
        if isinstance(airfoil,str):
            self.airfoil = asb.Airfoil(airfoil)
        else:
            self.airfoil = airfoil
            
        self.angle_of_attack = np.deg2rad(angle_of_attack)
        self.p3d_filepath = p3d_filepath
        self.supersonic_inlet = supersonic_inlet
        
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
                      working_directory=r"C:\Users\ag458\Documents\Projects\AeroProjects\AsbEuler")
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
        self.roinlet = np.zeros([nj])
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
        self.cfl = 1.5
        self.smoothing_factor = 0.5 * self.cfl
        self.convergence_limit = 10000 * self.cfl
        self.nsteps = 5000
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
    
    def Set_Timestep(self):
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
        

                                          
    def runge_kutta(self):
        
        for self.nstep in range(self.nsteps):
  
            ro_start = np.copy(self.ro)
            roe_start = np.copy(self.roe)
            rovx_start = np.copy(self.rovx)
            rovy_start = np.copy(self.rovy)
        
            for nrkut in range(self.nrkut_max):
                frkut = 1./(self.nrkut_max-nrkut)
            
                self.set_others()
                self.apply_boundary_conditions()
             
        
#@njit
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


if __name__ == "__main__":
    euler2d = main()