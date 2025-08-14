# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 11:28:56 2025

@author: kubap
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random

"""
I = np.array([[32.99,0,0],[0,778.8,0],[0,0,778.8]])
hI = 0.713
aT = 1.5
m = 3.5
ro = 1.1225
v0 = 60
g = 10
fi = g/v0
A = 0.05**2*np.pi
f0 = ro*v0**2*A/2
Icg = np.array([[32.99,0,0],[0,778.8-m*hI**2,0],[0,0,778.8-m*hI**2]])
IcgInv = sp.linalg.inv(Icg)
weights = np.array([[0,1],[0.2,1],[0.2,1]])
weights2 = np.array([[0,1],[1,1],[1,1]])
weightX = 50
Cmax = np.array([[0.0003],[0.1],[0.1]])
delT = 0.1
dt = 0.01
duration = 1/fi
"""

"""
PD controller for the rocket attitude control.
Adam Gocel and Manfred Gawlas
Script assumes following:
OX - axis longlitudal to rocket, axis pointing to rocket tip
OY, OZ - other axis
"""

# Physical parameters of the rocket
I = np.array([
    [0.00639, 0,     0],
    [0,     0.93, 0],
    [0,     0,     0.93]
    ]) # Inertia tensor at CG
Icg = I
hI = 0.713  # Distance to rocket mass center from tip - position (-0.713, 0, 0)
aT = 1.5  # Torque arm length (m)
m = 3.5  # Rocket mass (kg)
ro = 1.1225  # Air density (kg/m^3)
v0 = 60  # Initial velocity (m/s)
g = 9.81  # Gravitational acceleration (m/s^2)
fi = g/v0  # Deceleration parameter
A = 0.05**2*np.pi  # Reference area (m^2)
f0 = ro*v0**2*A/2  # Reference force (N/m^2)

IcgInv = sp.linalg.inv(I)  # Inverse of inertia tensor

# Control weights for different axes
weights = 1 * np.array([[0,1],[0.2,1],[0.2,1]])  # Weights for [roll, pitch, yaw]
weights2 = 100 * np.array([[0,1],[1,1],[1,1]])  # Secondary weights
weightX = 50  # Extra weight for roll control


# Control and simulation parameters
Cmax = np.array([[0.0003],[0.1],[0.1]])  # Maximum control coefficients
delT = 0.05  # Control update interval (s)
dt = 0.0001  # Simulation time step (s)
duration = v0 / g  # Total simulation duration (s)

def TorqueS(angles):
    """
    Calculate aerodynamic torques based on control surface angles.
    
    Args:
        angles: Array [az, bz, ay, by] representing control surface angles
    
    Returns:
        Torque vector [Tx, Ty, Tz]
    """
    az = angles[0]
    bz = angles[1]
    ay = angles[2]
    by = angles[3]

    # Z-axis torque components
    Tz1 = 0.000508 * np.sin(3 * bz) + 0.000653 * np.cos(2 * bz) * np.sin(3 * az) + 0.000125 * np.cos(3 * az) * np.sin(3 * bz)
    Tz2 = 0.00834 * np.cos(4 * az) - 0.00452 * np.cos(4 * bz) - 0.00258 * np.cos(4 * bz) * np.sin(4 * az)
    Tz3 = 0.189 * np.sin(2 * bz) - 0.0277 * np.sin(4 * az) - 0.00607 * np.cos(4 * az) - 0.269 * np.sin(bz)

    # Y-axis torque components
    Ty1 = 0.000508 * np.sin(3 * by) + 0.000653 * np.cos(2 * by) * np.sin(3 * ay) + 0.000125 * np.cos(3 * ay) * np.sin(3 * by)
    Ty2 = 0.00834 * np.cos(4 * ay) - 0.00452 * np.cos(4 * by) - 0.00258 * np.cos(4 * by) * np.sin(4 * ay)
    Ty3 = 0.189 * np.sin(2 * by) - 0.0277 * np.sin(4 * ay) - 0.00607 * np.cos(4 * ay) - 0.269 * np.sin(by)
    
    """
    return(np.array([[Tz1+Ty1],[Tz2+Ty3],[Tz3-Ty2]]))
    """
    
    return np.array([[Tz1+Ty1],[Tz2+Ty3],[Tz3-Ty2]])

def ForceS(angles):
    """
    Calculate aerodynamic forces based on control surface angles.
    
    Args:
        angles: Array [az, bz, ay, by] representing control surface angles
    
    Returns:
        Force vector [Fx, Fy, Fz]
    """
    az = angles[0]
    bz = angles[1]
    ay = angles[2]
    by = angles[3]

    # Force components
    Fz2 = 0.00588 * np.cos(2 * az) + 0.1 * np.sin(bz) - 0.103 * np.cos(bz) * np.sin(az)
    Fz3 = 0.0358 * np.cos(2 * bz) - 0.0314 * np.cos(2 * az) - 0.00335 * np.cos(2 * bz) * np.cos(2 * az)
    Fy2 = 0.00588 * np.cos(2 * ay) + 0.1 * np.sin(by) - 0.103 * np.cos(by) * np.sin(ay)
    Fy3 = 0.0358 * np.cos(2 * by) - 0.0314 * np.cos(2 * ay) - 0.00335 * np.cos(2 * by) * np.cos(2 * ay)
    
    """
    return(np.array([[0],[Fz2+Fy3],[Fz3-Fy2]]))
    """
    
    return np.array([[0],[Fz2+Fy3],[Fz3-Fy2]])

def TorqueCoeffs(angles):
    
    """
    T = TorqueS(angles)
    F = ForceS(angles)
    return(np.array([[T[0,0]*aT],[T[1,0]*aT-F[2,0]*hI],[T[2,0]*aT+F[1,0]*hI]]))
    """
    
    """
    Calculate total torque coefficients
    
    Returns:
        Torque coefficient vector
    """

    T = TorqueS(angles)  # Pure torque coeffs
    F = ForceS(angles)   # Aerodynamic force coeffs

    # Combined torque coeffs
    return np.array([[T[0,0]*aT],
                     [T[1,0]*aT-F[2,0]*hI],  # Pitch torque - force contribution
                     [T[2,0]*aT+F[1,0]*hI]]) # Yaw torque + force contribution

def findAngles(TCopt):
    """
    Find optimal control surface angles to achieve desired torque coefficients.
    
    Args:
        TCopt: Desired torque coefficients
    
    Returns:
        Optimal control surface angles
    """
    def Fcost(angles):
        """Cost function for optimization (weighted error between actual and desired torque)"""
        T = TorqueCoeffs(angles)
        
        """
        return ((weightX**2*(T[0, 0] - TCopt[0, 0]) ** 2 + (T[1, 0] - TCopt[1, 0]) ** 2 + (T[2, 0] - TCopt[2, 0]) ** 2) ** 0.5)
        """
        return ((weightX**2*(T[0, 0] - TCopt[0, 0])**2 + 
                 (T[1, 0] - TCopt[1, 0])**2 + 
                 (T[2, 0] - TCopt[2, 0])**2)**0.5)

    # Minimize cost function starting from neutral position [0,0,0,0]
    Q = sp.optimize.minimize(Fcost, np.array([0, 0, 0, 0])).x
    """
    return(Q)
def AddFromTC(TC,t):
    """
    return Q

def AddFromTC(TC, t):
    """
    Convert torque coefficients to angular accelerations.
    
    Args:
        TC: Torque coefficients
        t: Current time
    
    Returns:
        Angular acceleration vector
    """
    # Scale by dynamic pressure factor (1-fi*t)^2 and apply inverse inertia
    # ERROR?
    return (np.matmul(IcgInv, TC)*f0*(1-fi*t)**2)

"""
def StateTimeDerivative(S,add):
    return(np.array([[S[3,0]],[S[4,0]],[S[5,0]],[add[0,0]],[add[1,0]],[add[2,0]]]))
def stateIncrement(S,add,dt):
"""

def StateTimeDerivative(S, add):
    """
    Calculate time derivative of state vector.
    
    Args:
        S: Current state
        add: Angular acceleration vector
    
    Returns:
        State derivative vector
    """
    return np.array([[S[3,0]],
                     [S[4,0]],
                     [S[5,0]],
                     [add[0,0]],
                     [add[1,0]],
                     [add[2,0]]])

def stateIncrement(S, add, dt):
    """
    Calculate state increment RK4
    
    Args:
        S: Current state
        add: Angular acceleration vector
        dt: Time step
    
    Returns:
        State increment
    """
    k1 = StateTimeDerivative(S, add)
    k2 = StateTimeDerivative(S + k1 * dt / 2, add)
    k3 = StateTimeDerivative(S + k2 * dt / 2, add)
    k4 = StateTimeDerivative(S + k3 * dt, add)
    return (dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

"""
def desiredCoeffs(S,t):
    desC = np.array([[0], [0], [0]])
    desAdd = np.array([[0], [0], [0]])
    AddMax = np.matmul(IcgInv,Cmax)*f0*(1-fi*t)**2
    #Zero conditions
    A0 = np.array([[0], [delT ** 2 * weights2[1, 0] * AddMax[1, 0]], [delT ** 2 * weights2[2, 0] * AddMax[2, 0]]])
    Ad0 = np.array([[delT * weights2[0, 1] * AddMax[0, 0]], [delT * weights2[1, 1] * AddMax[1, 0]], [delT * weights2[2, 1] * AddMax[2, 0]]])
    #yaw
    if S[1,0]>A0[1,0]:
        if S[4,0]>=-Ad0[1,0]:
            desAdd = np.array([[desAdd[0,0]],[-AddMax[1,0]],[desAdd[2,0]]])
"""

def desiredCoeffs(S, t):
    """
    Implement PID control algorithm to calculate desired torque coeffs.
    
    Args:
        S: Current state vector
        t: Current time
    
    Returns:
        Desired torque coeffs
    """
    desC = np.array([[0], [0], [0]])  # Initialize desired coeffs
    desAdd = np.array([[0], [0], [0]])  # Initialize desired angular accs

    # Maximum possible angular accs at current time
    AddMax = np.matmul(IcgInv, Cmax)*f0*(1-fi*t)**2

    # Threshold conditions for position and velocity control
    A0 = np.array([[0], 
                   [delT**2 * weights2[1, 0] * AddMax[1, 0]], 
                   [delT**2 * weights2[2, 0] * AddMax[2, 0]]])

    Ad0 = np.array([[delT * weights2[0, 1] * AddMax[0, 0]], 
                    [delT * weights2[1, 1] * AddMax[1, 0]], 
                    [delT * weights2[2, 1] * AddMax[2, 0]]])

    # Yaw
    if S[1,0] > A0[1,0]:  # If yaw angle is too high
        if S[4,0] >= -Ad0[1,0]:  # And not decreasing fast enough
            desAdd = np.array([[desAdd[0,0]], [-AddMax[1,0]], [desAdd[2,0]]])  # Apply max negative acc
        else:
            
            """
            #APcheck
            AP = 0
    elif S[1, 0] < -A0[1, 0]:
        if S[4, 0] <= Ad0[1, 0]:
            desAdd = np.array([[desAdd[0, 0]], [AddMax[1, 0]], [desAdd[2, 0]]])
            """
            pass  # Already decreasing at appropriate rate
    elif S[1,0] < -A0[1,0]:  # If yaw angle is too low
        if S[4,0] <= Ad0[1,0]:  # And not increasing fast enough
            desAdd = np.array([[desAdd[0,0]], [AddMax[1,0]], [desAdd[2,0]]])  # Apply max positive acc
        else:
            
            """
            #APcheck
            AP = 0
    else:
        if S[4, 0] < -Ad0[1, 0]:
            desAdd = np.array([[desAdd[0, 0]], [AddMax[1, 0]], [desAdd[2, 0]]])
        elif S[4,0] > Ad0[1,0]:
            desAdd = np.array([[desAdd[0,0]],[-AddMax[1,0]],[desAdd[2,0]]])
            """
            pass  # Already increasing at appropriate rate
    else:  # Yaw angle is within acceptable range
        if S[4,0] < -Ad0[1,0]:  # Angular velocity too negative
            desAdd = np.array([[desAdd[0,0]], [AddMax[1,0]], [desAdd[2,0]]])
        elif S[4,0] > Ad0[1,0]:  # Angular velocity too positive
            desAdd = np.array([[desAdd[0,0]], [-AddMax[1,0]], [desAdd[2,0]]])
        else:  # Both angle and velocity in acceptable range - apply proportional control
            desAdd = np.array([[desAdd[0,0]], 
                              [-S[4,0]/delT*weights[1,1] - S[1,0]/delT**2*weights[1,0]], 
                              [desAdd[2,0]]])

    # Pitch
    if S[2,0] > A0[2,0]:  # If pitch angle is too high
        if S[5,0] >= -Ad0[2,0]:  # And not decreasing fast enough
            desAdd = np.array([[desAdd[0,0]], [desAdd[1,0]], [-AddMax[2,0]]])  # Apply max negative acc
        else:
            """
            desAdd = np.array([[desAdd[0, 0]], [-S[4, 0] / delT * weights[1, 1] -S[1, 0] / delT**2 * weights[1, 0]], [desAdd[2, 0]]])
    #pitch
    if S[2,0]>A0[2,0]:
        if S[5,0]>=-Ad0[2,0]:
            desAdd = np.array([[desAdd[0,0]],[desAdd[1,0]],[-AddMax[2,0]]])
            """
            
            pass  # Already decreasing at appropriate rate
    elif S[2,0] < -A0[2,0]:  # If pitch angle is too low
        if S[5,0] <= Ad0[2,0]:  # And not increasing fast enough
            desAdd = np.array([[desAdd[0,0]], [desAdd[1,0]], [AddMax[2,0]]])  # Apply max positive acc
        else:
            
            """
            #APcheck
            AP = 0
    elif S[2, 0] < -A0[2, 0]:
        if S[5, 0] <= Ad0[2, 0]:
            desAdd = np.array([[desAdd[0, 0]], [desAdd[1, 0]], [AddMax[2, 0]]])
        else:
            #APcheck
            AP = 0
    else:
        if S[5, 0] < -Ad0[2, 0]:
            desAdd = np.array([[desAdd[0, 0]], [desAdd[1, 0]], [AddMax[2, 0]]])
        elif S[5,0] > Ad0[2,0]:
            desAdd = np.array([[desAdd[0,0]],[desAdd[1,0]],[-AddMax[2,0]]])
        else:
            desAdd = np.array([[desAdd[0, 0]], [desAdd[1, 0]], [-S[5, 0] / delT * weights[2, 1] -S[2, 0] / delT**2 * weights[2, 0]]])
    #roll
    if S[3,0]>Ad0[0,0]:
        desAdd = np.array([[-AddMax[0,0]],[desAdd[1,0]],[desAdd[2,0]]])
    elif S[3,0]<-Ad0[0,0]:
        desAdd = np.array([[AddMax[0,0]],[desAdd[1,0]],[desAdd[2,0]]])
    else:
        desAdd = np.array([[-S[3,0]/delT*weights[0,1]],[desAdd[1,0]],[desAdd[2,0]]])
    #ADD--->C
    desC = np.matmul(Icg,desAdd)/f0/(1-fi*t)**2
            """
            
            pass  # Already increasing at appropriate rate
    else:  # Pitch angle is within acceptable range
        if S[5,0] < -Ad0[2,0]:  # Angular velocity too negative
            desAdd = np.array([[desAdd[0,0]], [desAdd[1,0]], [AddMax[2,0]]])
        elif S[5,0] > Ad0[2,0]:  # Angular velocity too positive
            desAdd = np.array([[desAdd[0,0]], [desAdd[1,0]], [-AddMax[2,0]]])
        else:  # Both angle and velocity in acceptable range - apply proportional control
            desAdd = np.array([[desAdd[0,0]], 
                              [desAdd[1,0]], 
                              [-S[5,0]/delT*weights[2,1] - S[2,0]/delT**2*weights[2,0]]])

    # Roll
    if S[3,0] > Ad0[0,0]:  # If roll rate is too high
        desAdd = np.array([[-AddMax[0,0]], [desAdd[1,0]], [desAdd[2,0]]])  # Apply negative acc
    elif S[3,0] < -Ad0[0,0]:  # If roll rate is too low
        desAdd = np.array([[AddMax[0,0]], [desAdd[1,0]], [desAdd[2,0]]])  # Apply positive acc
    else:  # Roll rate is within acceptable range
        desAdd = np.array([[-S[3,0]/delT*weights[0,1]], [desAdd[1,0]], [desAdd[2,0]]])  # Proportional control

    # Convert desired angular accs to torque coeffs
    desC = np.matmul(Icg, desAdd)/f0/(1-fi*t)**2

    # Limit to maximum control authority
    for i in range(3):
        
        """
        if desC[i,0]>Cmax[i,0]:
            desC[i, 0] = Cmax[i, 0]
        if desC[i,0]<-Cmax[i,0]:
            desC[i, 0] = -Cmax[i, 0]
    return(desC)
state = np.array([[0],[0.00003],[-0.0004],[0.00001],[0],[0]])
X = []
Y = []
Z = []
VX = []
VY = []
VZ = []
        """
        
        if desC[i,0] > Cmax[i,0]:
            desC[i,0] = Cmax[i,0]
        if desC[i,0] < -Cmax[i,0]:
            desC[i,0] = -Cmax[i,0]

    return desC


def to_rad(x):
    return x/180*np.pi

def to_deg(x):
    return x*180/np.pi

state = np.array([[0],[to_rad(5)],[to_rad(5)],[to_rad(10)],[0],[0]])

# Store
X = []  # Roll angle
Y = []  # Yaw angle
Z = []  # Pitch angle
VX = []  # Roll rate
VY = []  # Yaw rate
VZ = []  # Pitch rate
AX = []
AY = []
AZ = []
T = []

"""
timesince = delT
C = np.array([[0],[0],[0]])
"""

AZ_canard = []
BZ_canard = []
AY_canard = []
BY_canard = []

timesince = delT  # Time since last control update
C = np.array([[0],[0],[0]])  # Initial control torque coefficients
current_angles = np.array([0, 0, 0, 0])  # Initial canard angles

# Main simulation
for i in range(int(duration/dt)):
    
    """
    t = i*dt
    if timesince>=delT:
        C = TorqueCoeffs(findAngles(desiredCoeffs(state,t)))
        timesince += -delT
    ADD = np.matmul(IcgInv,C)*f0*(1-fi*t)**2
    state = state + stateIncrement(state,ADD,dt)
    """
    
    t = i*dt  # Current time

    # Update control inputs at specified intervals
    if timesince >= delT:
        # Find optimal canard angles based on desired coefficients
        current_angles = findAngles(desiredCoeffs(state, t))
        C = TorqueCoeffs(current_angles)
        timesince = 0  # Reset timer

    # Calculate angular accs from torque coefs
    ADD = np.matmul(IcgInv, C)*f0*(1-fi*t)**2

    # Update state using RK4
    state = state + stateIncrement(state, ADD, dt)

    # Increment control timer
    timesince += dt

    # Store results
    T.append(t)
    
    """
    X.append(state[0,0])
    Y.append(state[1, 0])
    Z.append(state[2, 0])
    VX.append(state[3, 0])
    VY.append(state[4, 0])
    VZ.append(state[5, 0])
    AX.append(ADD[0, 0])
    AY.append(ADD[1, 0])
    AZ.append(ADD[2, 0])
plt.plot(T,X)
plt.plot(T,VX)
plt.plot(T,AX)
plt.show()
plt.plot(T,Y)
plt.plot(T,VY)
plt.plot(T,AY)
    """
    
    X.append(state[0,0])  # Roll angle
    Y.append(state[1,0])  # Yaw angle
    Z.append(state[2,0])  # Pitch angle
    VX.append(state[3,0])  # Roll vel
    VY.append(state[4,0])  # Yaw vel
    VZ.append(state[5,0])  # Pitch vel
    AX.append(ADD[0,0])   # Roll acc
    AY.append(ADD[1,0])   # Yaw acc
    AZ.append(ADD[2,0])   # Pitch acc

    AZ_canard.append(current_angles[0])
    BZ_canard.append(current_angles[1])
    AY_canard.append(current_angles[2])
    BY_canard.append(current_angles[3])

# Rad to deg for plotting
X_deg = [to_deg(x) for x in X]
Y_deg = [to_deg(y) for y in Y]
Z_deg = [to_deg(z) for z in Z]

VX_deg = [to_deg(vx) for vx in VX]
VY_deg = [to_deg(vy) for vy in VY]
VZ_deg = [to_deg(vz) for vz in VZ]

AX_deg = [to_deg(ax) for ax in AX]
AY_deg = [to_deg(ay) for ay in AY]
AZ_deg = [to_deg(az) for az in AZ]

AZ_canard_deg = [to_deg(az) for az in AZ_canard]
BZ_canard_deg = [to_deg(bz) for bz in BZ_canard]
AY_canard_deg = [to_deg(ay) for ay in AY_canard]
BY_canard_deg = [to_deg(by) for by in BY_canard]

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Rocket Dynamics and Control', fontsize=16)

# Roll dynamics 
axs[0, 0].plot(T, X_deg, 'b-', label='Roll Angle (deg)')
axs[0, 0].plot(T, VX_deg, 'g-', label='Roll Rate (deg/s)')
axs[0, 0].plot(T, AX_deg, 'r-', label='Roll Accel (deg/s²)')
axs[0, 0].set_title('Roll Dynamics')
axs[0, 0].set_xlabel('Time (s)')
axs[0, 0].set_ylabel('Magnitude (degrees)')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Yaw dynamics
axs[0, 1].plot(T, Y_deg, 'b-', label='Yaw Angle (deg)')
axs[0, 1].plot(T, VY_deg, 'g-', label='Yaw Rate (deg/s)')
axs[0, 1].plot(T, AY_deg, 'r-', label='Yaw Accel (deg/s²)')
axs[0, 1].set_title('Yaw Dynamics')
axs[0, 1].set_xlabel('Time (s)')
axs[0, 1].set_ylabel('Magnitude (degrees)')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Ptich
axs[1, 0].plot(T, Z_deg, 'b-', label='Pitch Angle (deg)')
axs[1, 0].plot(T, VZ_deg, 'g-', label='Pitch Rate (deg/s)')
axs[1, 0].plot(T, AZ_deg, 'r-', label='Pitch Accel (deg/s²)')
axs[1, 0].set_title('Pitch Dynamics')
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('Magnitude (degrees)')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Cann
axs[1, 1].plot(T, AZ_canard_deg, 'b-', label='Z-axis Primary (deg)')
axs[1, 1].plot(T, BZ_canard_deg, 'g-', label='Z-axis Secondary (deg)')
axs[1, 1].plot(T, AY_canard_deg, 'r-', label='Y-axis Primary (deg)')
axs[1, 1].plot(T, BY_canard_deg, 'y-', label='Y-axis Secondary (deg)')
axs[1, 1].set_title('Canard Control Angles')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('Angle (deg)')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

"""
plt.plot(T,Z)
plt.plot(T,VZ)
plt.plot(T,AZ)
plt.show()
"""