# Implementation de hal-00744813
# Par Rabin Julien, Gabriel Peyré
# Régularisation de Wasserstein. Application au Transfert de Couleur

#%% Importations
import cv2
import numpy as np
import matplotlib.pyplot as plt

#%% Opérateurs différentiels: Ile opèrent sur des images couleur 
# de taille H W d (d vaut 3 pour une image couleur) H=Hauteur W=largeur 


def grad_champ(im):
    """ Renvoie le gradient d'une image. La taille de retour est 2,H,W,d si la
    taille de l'image est H,W,d"""
    
    out=np.zeros((2,*im.shape),dtype=im.dtype)
    out[0,:,:-1]=im[:,1:]-im[:,:-1]
    out[1,:-1,:]=im[1:,:]-im[:-1,:]
    return out

def div_champ(ch):
    """Renvoies la divergence d'un champ de vecteur. C'est - transposé de
    l'opérateur grad_champ. La sortie est de taille H,W,d si l'entrée est de 
    taille 2,H,W,d"""
    out=np.zeros(ch.shape[1:],dtype=ch.dtype)
    out[:,1:-1]=ch[0,:,1:-1]-ch[0,:,:-2]
    out[:,-1]-=ch[0,:,-2]
    out[:,0]=ch[0,:,0]
    out[1:-1,:]+=ch[1,1:-1,:]-ch[1,:-2,:]
    out[0,:]+=ch[1,0,:]
    out[-1,:]-=ch[1,-2,:]
    
    return out

def prod_scal(X,Y):
    return (X*Y).sum()

def norm_grad_champ(ch):
    """ si entree = 2,H,W,3 la sortie est de taille H,W,3
    on rajoute une fausse dimension 1,H,W,3 pour pouvoir diviser point par point"""
    out=(ch[0]**2+ch[1]**2)**0.5
    return out.reshape((-1,*out.shape))

def projinf(x,lmb):
    Nx=norm_grad_champ(x)
    Nx1=Nx.reshape(-1)
    mask=(Nx>lmb)
    mult=np.ones(Nx.shape,Nx.dtype)
    mult1=mult.reshape(-1)
    mask1=mask.reshape(-1)
    mult1[mask1]=lmb/Nx1[mask1]
    return x*mult

def chambolle(w,lmb,nbpas=100):
    rho=1/12
    x=np.zeros((2,*w.shape),dtype=w.dtype)
    for k in range(nbpas):
        x=projinf(x+rho*grad_champ(div_champ(x)-w),lmb)
    return x

def prox_TV(w,lmb):
    xetoile=chambolle(w,lmb)
    #print ('norme de xetoile ',norm2(xetoile))
    return w-div_champ(xetoile)

def grad_F(w,u,divgradu,lmbL,lmbLS):
    lmbL*(w-u)
    
def tirage_thetas(N=10):
    thetas=np.random.randn(N,3)
    tNs=((thetas**2).sum(axis=1,keepdims=True))**0.5
    return thetas/tNs

      

def projette_ordonne(v,theta):
    vals=(v*(theta.reshape((1,1,3)))).sum(axis=-1).reshape(-1)
    vals.sort()
    return vals

def prepare_SW2(v,N=10):
    thetas=tirage_thetas(N=N)
    vords=np.zeros((N,v.shape[0]*v.shape[1]),v.dtype)
    for k in range(N):
        vords[k]=projette_ordonne(v,thetas[k])
    return thetas,vords

def grad_SW2_un_angle(w,vord,theta):
    vals=(w*(theta.reshape((1,1,3)))).sum(axis=-1).reshape(-1)
    idx=vals.argsort()
    idxm1=idx.argsort()
    return (((vals[idx]-vord)[idxm1]).reshape((-1,1))*(theta.reshape((1,3)))).reshape((w.shape[0],w.shape[1],3))

def grad_SW2(w,vords,thetas):
    g=np.zeros(w.shape,w.dtype)
    for k in range(thetas.shape[0]):
        g+=grad_SW2_un_angle(w,vords[k],thetas[k])
    return g/(vords.shape[0]/2)


def norm2(X):
    return (X**2).sum()**0.5
def transfert_couleurs(u,v,nbetapes=100,lmbL=0.1,lmbR=0.5,lmbLS=0.5,lmbS=1,N=10):
    """ transfert les couleurs de v vers l'image u. Renvoie une image couleur de la taille de u"""
    
    thetas,vords=prepare_SW2(v,N=N)
    w=u.copy()
    H=np.zeros((3,3))
    for k in range(N):
        H+=thetas[k].reshape((-1,1))@thetas[k].reshape((1,-1))
    H*=(2.0/N)
    mv=np.linalg.eigh(H)[0][-1]
    tmax=1/(lmbL+lmbS/2*mv)
    tau=tmax/2
    gu=grad_champ(u)
    eps=1e-6
    ngu=norm_grad_champ(gu)+eps
    cst=lmbLS*div_champ(gu/ngu)-lmbL*u
    for k in range(nbetapes):
        print ('Etape numero',k,' sur ', nbetapes)
        #ETAPE E
        gW2=grad_SW2(w,vords,thetas)*(lmbS/2)
        #print ('norme de gW2 ',norm2(gW2), 'norme de cst ', norm2(cst))
        grad_tot=(lmbL*w+cst+gW2)
        #print('norme grad_tot ',norm2(grad_tot))
        wk05=w-tau*grad_tot
        #ETAPE I
        w=prox_TV(wk05,tau*lmbR)
        #print('norme de la tranlation par prox', norm2(w-wk05))
    return w

#%%
test=np.random.randn(256,256,3)
gtest=grad_champ(test)

# on verifie que les deux operateurs sont conjugues 
#<grad im | toto>=<im|-div toto> ?
toto=np.random.randn(2,67,89,3)
titi=np.random.randn(67,89,3)
a=prod_scal(grad_champ(titi),toto)
b=prod_scal(titi,div_champ(toto))
print (a+b,a,b)

#%% test du transfert de couleurs

ufile='/home/onyxia/work/img_u.png'
vfile='/home/onyxia/work/img_v.png'

#ufile='arecoloriser.png'
#vfile='source_couleur.png'


v = cv2.imread(vfile, cv2.IMREAD_COLOR).astype(np.float32)
u = cv2.imread(ufile, cv2.IMREAD_COLOR).astype(np.float32)
#u=cv2.cvtColor(u,cv2.COLOR_BGR2RGB)
#v=cv2.cvtColor(v,cv2.COLOR_BGR2RGB)
#v=v[:215,:176,:]
#%%subsample
'''uu=np.zeros((u.shape[0]//2,u.shape[1]//2,3))
vv=np.zeros((v.shape[0]//2,v.shape[1]//2,3))
for k in range(2):
    for l in range(2):
        uu+=u[k::2,l::2]
        vv+=v[k::2,l::2]
u=uu/4
v=vv/4'''
#%%

cv2.imwrite('source_u_2.png', u)
cv2.imwrite('target_v_2.png', v)
#%%
#w=transfert_couleurs(u,v,nbetapes=100,lmbS=1)

#%%
#cv2.imwrite('result_w.png', w)

