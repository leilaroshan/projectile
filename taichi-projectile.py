import taichi as ti
import math
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu)  

g  = np.array([0, -9.8])  # Earth's gravity
v0 = 10      # initial velocity (m/s)

angle = 30   # Angle of falling (degrees)
theta = math.radians(angle)  # Convert degrees to radians
dt=0.01      # time step 

# Create Taichi vector fields with a specified shape
N = 1
position = ti.Vector.field(2, dtype=ti.f32, shape=N, needs_grad=True)   #[0,0]=[x,y]
velocity = ti.Vector.field(2, dtype=ti.f32, shape=N, needs_grad=True)   # [0,0]=[vx,vy]
acc = ti.Vector.field(2, dtype=ti.f32, shape=N, needs_grad=True)

v_init = ti.Vector.field(2, dtype=ti.f32)
loss = ti.field(dtype=ti.f64)

target = [40.0, 0.0]

ti.root.place(v_init)
ti.root.place(loss)
ti.root.lazy_grad()

# Initialize variables  (initial positions are already zero x0=y0=0 , we just update velocity)

@ti.kernel
def initialize():
    position[0] = [0, 0]
    acc[0] = [0, -9.81]
    velocity[0] = v_init[None]
    

@ti.kernel
def advance():
    # Update position (x)
    position[0] += velocity[0] * dt
    # Update velocity (vy)
    velocity[0] += acc[0] * dt  

# Loss Calculation 
@ti.kernel
def compute_loss():
  loss[None] = (position[0][0] - target[0])**2 + (position[0][1] - target[1])**2


def forward(steps):
    # Main loop
    initialize()
    for i in range(steps):
        advance()
        """
        # Don't update if y below ground (y<0)
        if position[0][1] < 0:   #(if y<0)
            break
        """
    compute_loss()
    
learning_rate = 0.1
def optimization():
    angle = math.radians(30)
    v_init[None] = [v0 * np.cos(angle), v0 * np.sin(angle)]
    for i in range(200):
        steps = int((2 * v_init[None][1])/(-g[1])/dt)  
        with ti.ad.Tape(loss):
            forward(steps)
        
        for k in range(2):
            v_init[None][k] -= learning_rate * v_init.grad[None][k]

        print("Iteration: {} final position: {}".format(i,  position[0]))
optimization()

# Convert trajectory data to NumPy arrays for plotting
initialize()
steps = int((2 * v_init[None][1])/-g[1]/dt) 
for i in range(steps):
    position[0] += dt * velocity[0]
    velocity[0] += dt * acc[0]
    plt.scatter(position[0][0], position[0][1], 3)

plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Projectile Motion with the Angle of ' + str(theta) + ' Degrees')
plt.show()
