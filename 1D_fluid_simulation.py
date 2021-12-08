import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

class oneD_simulation():
    
    def __init__(self,width,dt,dx,tau):
        self.width = width
        self.dt = dt
        self.dx = dx
        self.tau = tau

        self.k = 9.8*(self.dt**2)/(2*(self.dx**2))
        
        self.h = np.zeros(width,dtype=(np.float32))
        self.b = np.zeros(width,dtype=(np.float32))
        self.d = np.zeros(width,dtype=(np.float32))
    
    
    def createEnv(self,x_axis_range):

        self.x_axis = np.arange(-150,150)

        for i in range(len(self.x_axis)):
            self.b[i] = self.x_axis[i]**2/10000.0
        
        for j in range(30,50):
            self.d[j] = 3.0
        
        
    def calcE(self):
        
        self.e = np.zeros(self.width,dtype=(np.float32))
        
        for i in range(0,self.width):
            if(i == 0):
                self.e[i] = 1 + (self.k*(self.d[0]+self.d[1]))
            elif(0<i and i<self.width-1):
                self.e[i] = 1 + (self.k*(self.d[i-1] + 2*self.d[i] + self.d[i+1]))
            elif ( i == self.width-1):
                self.e[i] = 1 + (self.k*(self.d[self.width-2] + self.d[self.width-1]))
        
        return self.e
    
    def calF(self):
        self.f = np.zeros(self.width-1,dtype=(np.float32))
        
        for i in range(0,self.width-1):
           self.f[i] = self.k*(self.d[i] + self.d[i+1]) * -1.0
        
        return self.f
    
    
    def simulateEq21 (self):
        
        self.h_0 = self.b + self.d
        
        for n in range ( 1, self.width):
            
            e = self.calcE()
            f = self.calF()
  
            A = diags(e) + diags(f,-1) + diags(f,+1)
         
            if ( n == 1):
                self.y = self.h_0.copy()
                prevY = self.y.copy()
            else:
                self.y = (self.b + self.d) + (1 - self.tau) * (self.b + self.d - prevY)
                prevY = self.y.copy()
                
            self.h = spsolve(A,self.y)
            
            self.d = self.h-self.b
            
            for k in range(len(self.d)):
                if(self.d[k] < 0 ):
                    self.d[k] = 0
            
            self.h = self.b + self.d
            
            
            plt.plot(self.x_axis*self.dx, self.b+self.d, color ="blue")
            plt.plot(self.x_axis*self.dx, self.b, color ="red")
            plt.ylim(0, 8)
            plt.show()
            
    def simulateEq45(self):
        self.h_current = self.b + self.d
        
        for n in range ( 1, self.width ):
            e = self.calcE()
            f = self.calF()
            
            if ( n == 1):
                self.y = self.h_current.copy()
                prevY = self.y.copy()
                
            else:
                self.y = (self.b + self.d) + (1 - self.tau) * (self.b + self.d - prevY)
                prevY = self.y.copy()
                                
            for i in range(0,self.width):
                if i == 0:
                    self.h[i] = (self.y[i] - f[i]*self.h_current[i+1]) / e[i]
                elif (i == self.width -1 ):
                    self.h[i] = (self.y[i] - f[i-1]*self.h_current[i-1]) / e[i]
                else:
                    self.h[i] = (self.y[i] - f[i-1]*self.h_current[i-1] - f[i]*self.h_current[i+1]) / e[i]
                
        
            self.d = self.h-self.b
          
            for k in range(len(self.d)):
                if(self.d[k] < 0 ):
                    self.d[k] = 0
            
            
            self.h = self.b + self.d
            self.h_current = self.h.copy()
                       
            plt.plot(self.x_axis*self.dx, self.b+self.d, color ="blue")
            plt.plot(self.x_axis*self.dx, self.b, color ="red")
            plt.ylim(0, 8)
            plt.show()
            
    

if __name__ == '__main__':
    
    oneDsim = oneD_simulation(300,0.1,0.1,0.01)
    
    oneDsim.createEnv(250)
    
    # oneDsim.simulateEq21()
    oneDsim.simulateEq45()
    
