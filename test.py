import numpy as np

n = 3
Dk = np.zeros((n,n),dtype=np.float)
Ek = np.zeros((n**2,n**2),dtype=np.float)
Fk = np.zeros((n**3,n**3),dtype=np.float)

k = 2/(1 + 1 + 1)

# Setup 1D FD matrix
Dk[0,0] = -k
Dk[0,1] = 1
Dk[-1,-1] = -k
Dk[-1,-2] = 1/1

for i in range(1,n-1):
    Dk[i,i] = -k
    Dk[i,i-1] = 1/1
    Dk[i,i+1] = 1/1

# Setup 2D FD matrix
I = np.identity(n)/1

Ek[0:n,0:n] = Dk
Ek[0:n,n:(2*n)] = I
Ek[(n-1)*n:,(n-1)*n:] = Dk
Ek[(n-1)*n:,(n-2)*n:(n-1)*n] = I

for i in range(n,((n-1)*n-1),n):
    Ek[i:(i+n),i:(i+n)] = Dk
    Ek[i:(i+n),(i-n):i] = I
    Ek[i:(i+n),(i+n):(i+2*n)] = I
    
# Setup 3D FD matrix
J = np.identity(n**2)/1

Fk[0:n**2,0:n**2] = Ek
Fk[0:n**2,n**2:(2*n**2)] = J
Fk[(n-1)*n**2:,(n-1)*n**2:] = Ek
Fk[(n-1)*n**2:,(n-2)*n**2:(n-1)*n**2] = J


for i in range(n**2,((n-1)*n**2-1),n**2):
    Fk[i:(i+n**2),i:(i+n**2)] = Ek
    Fk[i:(i+n**2),(i-n**2):i] = J
    Fk[i:(i+n**2),(i+n**2):(i+2*n**2)] = J


print(Fk[12:18,12:18])