import numpy as np
# from einops import rearrange

def example(mF,R,Z):
    omega_vel = 1
    # output = R*0
    # output = np.hstack(output,R*omega_vel)
    # output = np.hstack(output,R*0)
    if mF == 0:
        return np.concatenate((R*0,R*omega_vel,R*0)) #first are compo R, then theta and then Z
    else:
        return np.zeros(3*len(R))


def solid_body(mF,R,Z):
    omega_vel = 1
    # output = R*0
    # output = np.hstack(output,R*omega_vel)
    # output = np.hstack(output,R*0)
    if mF == 0:
        return np.concatenate((R*0,R*omega_vel,R*0))
    else:
        return np.zeros(3*len(R))