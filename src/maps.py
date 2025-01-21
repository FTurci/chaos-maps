import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
import matplotlib
matplotlib.use('cairo')

class Map:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
        self.bounds= kwargs['bounds']
        self.pos= None

    def plot(self,dpi=None):
        # print(self.pos)
        # plt.plot(self.pos[:,0], self.pos[:,1],',', color="royalblue")
        if dpi is not None:
            scatter = plt.scatter(self.pos[:,0], self.pos[:,1],marker=',',s=(72./fig.dpi)**2*6,linewidths=0,c=self.pos[:,2], cmap=cmr.horizon)
        plt.xlim((self.bounds[0][0], self.bounds[1][0]))
        plt.ylim((self.bounds[0][1], self.bounds[1][1]))
        return scatter
 

class StandardMap(Map):
    def __init__(self, k=None,bounds= [[0,0],[2*np.pi,2*np.pi]]):
        # Call the base class constructor with any parameters
        super().__init__(k=k,bounds=bounds)

        self.k = self.kwargs['k']

    def iterate(self, niter=30,npoints=100,seed=None):
        if seed!=None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed)
        # x = np.array([[1,1.5]])#
        
        # x = rng.uniform(self.bounds[0], self.bounds[1], (npoints,2))
        dtheta = 2*np.pi/npoints
        thetas = np.arange(0,2*np.pi,dtheta)
        x = np.column_stack((thetas, np.pi*np.ones_like(thetas)))
        # print(x)
        pos = np.zeros((niter*npoints,3))
        
        for i in range(niter):
            # update theta
            x[:,0] = (x[:,0]+self.k*np.sin(x[:,1]))%(2*np.pi)
            x[:,1] = (x[:,1]+x[:,0])%(2*np.pi)
            pos[i*npoints:(i+1)*npoints,:2] = x.copy()
            pos[i*npoints:(i+1)*npoints,2] = np.arange(npoints)
        # print("pos")
        # print(pos.shape)
        pos[:,0] = (pos[:,0]-np.pi)%(2*np.pi)
        self.pos = pos


# M = StandardMap(2.03)
# M.iterate(niter=5000)
# M.plot()
# plt.axis('equal')
# plt.show()

fig = plt.figure(figsize=(5,5),dpi=300)
plt.gca().set_facecolor('black')
import tqdm
M = StandardMap(0)
M.iterate(niter=300,seed=1,npoints=100)
scatter = M.plot(dpi=fig.dpi)
plt.axis('equal')
# plt.axis('off')
plt.xticks([],[])
plt.yticks([],[])
plt.tight_layout()

for k in tqdm.tqdm(np.linspace(0,4,100)):
    M = StandardMap(k)
    M.iterate(niter=300,seed=1,npoints=100)

    scatter.set_offsets(M.pos[:,:2])
    plt.savefig(f"variation_frame_k{k}.png",bbox_inches='tight',dpi=fig.dpi)
    # plt.clf()
    # plt.show()