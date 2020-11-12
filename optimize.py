
from dolfin import *
import numpy
import scipy . optimize as optimize
from scipy . optimize import minimize

def func(data):
# Create mesh and define function space
#mesh = Mesh("circle.xml.gz")
mesh = Mesh("sphere.xml.gz")
Q = FunctionSpace(mesh, 'CG', 1)
# Define parameters:
#T = 50
T=1
h = mesh.hmin()
dt = h # time step
alpha = 0.01

# Create subdomain for Dirichlet boundary
class DirichletBoundary(SubDomain):
    def inside(self, x, on boundary):
        return on boundary

# Set up boundary condition
g = Constant (0.0)
bc = DirichletBC (Q, g, DirichletBoundary ())
# Define Initial condition
#indata = Expression('(sqrt((0.5-sqrt((x[0]*x[0] + x[1]*x[1])))*(0.5-sqrt((x[0]*x[0] + x[1]*x[1]))))) <= 0.2 ? 40.0: 0.0')
indata = Expression('(B-sqrt(x[0]*x[0] + x[1]*x[1]))*(B-sqrt(x[0]*x[0] + x[1]*x[1])) +x[2]*x[2] <= C*C ? A: 0.0', A=data[0], B=data[1], C=data[2], degree = 2)

u0 = Function (Q)
u0 = interpolate(indata, Q)

# Define variational problem
u = TrialFunction(Q)
v = TestFunction(Q)
f = Constant(0.0)

# Create bilinear and linear forms
a = u*v*dx + dt*alpha*inner(nabla grad(u), nabla grad(v))*dx
L = (u0 + dt*f)*v*dx

# Output file
#out file = File("results/hej.pvd")
out file = File("results/hej2.pvd")
A = assemble(a) # assemble only once, before the time stepping
b = None # necessary for memory saving assemeble call

# Compute solution
u = Function(Q) # the unknown at a new time level
t = dt
while t <= 5:
    b = assemble(L, tensor=b)
    g.t = t
    bc.apply(A, b)
    solve(A, u.vector(), b)
    #Copy initial data
    u initial = Function(Q)
    u initial = interpolate(indata, Q)
    #Define an integral functional
    M = (u initial - u)*dx
    #Compute the functional
    mass1 = assemble(M)
    t += dt
    u0.assign(u)

t = dt

while t <=7:
    b = assemble(L, tensor=b)
    g.t = t
    bc.apply(A, b)
    solve(A, u.vector(), b)
    #Copy initial data
    u initial = Function(Q)
    u initial = interpolate(indata, Q)
    #Define an integral functional
    M = (u initial - u)*dx
    #Compute the functional
    mass2 = assemble(M)
    t += dt
    u0.assign(u)

t = dt

while t <=30:
    b = assemble(L, tensor=b)
    g.t = t
    bc.apply(A, b)
    solve(A, u.vector(), b)
    #Copy initial data
    u initial = Function(Q)
    u initial = interpolate(indata, Q)
    #Define an integral functional
    M = (u initial - u)*dx
    #Compute the functional
    mass3 = assemble(M)
    t += dt
    u0.assign(u)

print((mass1-10.0)**2+(mass2-15.0)**2+(mass3-30.0)**2)

return (mass1-10.0)**2+(mass2-15.0)**2+(mass3-30.0)**2

data = [20 , 0.5, 0.1]

res=minimize(func, data, method = 'nelder-mead', options={'xtol':1e-3, 'disp':True})

print(res)
