import numpy as np
import random
from RLcore import DPRL, MDP


    

class GridWorld(MDP):
    def __init__(self, height, width, goals, blocks):
        super(GridWorld,self).__init__(height*width,['up','down','left', 'right'],[-1,-10],np.ravel_multi_index([goals[:,0],goals[:,1]],[height,width]))
        self.width = width
        self.height = height        
        #self.policy = np.random.choice(self.actions,height*width)
        self.grid = np.empty([height,width])
        self.grid.fill(-1)
        if len(blocks)!=0:
            self.grid[blocks[:,0],blocks[:,1]] = -10
    def printGrid(self):
        print(self.grid)       
    def dynamics(self, sp,r,s,a): 
        if s in self.terminal:
            return 0;
        if a == 'up':
            next = s - self.width
        elif a == 'down':
            next = s + self.width
        elif a == 'left':
            next = s - int((s%self.width)!=0)
        elif a == 'right':
            next = s + int(((s+1)%self.width)!=0)
        else:
            return 0
        if next < 0 or next>=self.width*self.height:
            next = s;
        if sp == next and r == self.grid[next//self.width,next%self.width]:
            return 1 
        else:
            return 0
    def printPolicy(self,policy):
        token = ''
        arrows = ['\u2191','\u2193','\u2190','\u2192']
        goal = '\u272A'
        for s in self.states:
            print(s,end=": ")
            if s in self.terminal:
                print(goal)
                continue
            for a in range(len(self.actions)):
                if policy[s,a]!=0:
                    print(arrows[a],end=" ")
            print("")   
            

WIDTH = 4
HEIGHT = 4
gw = GridWorld(HEIGHT,WIDTH,np.array([[0,0]]),np.array([[0,1]]))
gw.printGrid()  
discount = 1.0

solver = DPRL(gw,discount)
print(solver.policy)
solver.policyIteration()
print(solver.policy)
gw.printPolicy(solver.policy)

