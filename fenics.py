from dolfin import *
import numpy

# Create mesh and define function space
#mesh = Mesh("circle.xml.gz")
mesh = Mesh("sphere.xml.gz")
Q = FunctionSpace(mesh, 'CG', 1)

# Define parameters:
T = 20
#T=50
h = mesh.hmin()
dt = h # time step
alpha = 0.01
#print dt

# Create subdomain for Dirichlet boundary
class DirichletBoundary(SubDomain):
    def inside(self, x, on boundary):
        return on boundary

# Set up boundary condition
g = Constant (0.0)
bc = DirichletBC (Q, g, DirichletBoundary ())
#indata = Expression('abs(0.5-sqrt(x[0]*x[0]+x[1]*x[1])) <= 0.3? 10.0: 0.0 ', degree=1)
#indata = Expression('(sqrt((0.5-sqrt((x[0]*x[0] + x[1]*x[1])))*(0.5-sqrt((x[0]*x[0] + x[1]*x[1]))))) <= 0.2 ') #, degree == 1
indata = Expression('(0.5-sqrt(x[0]*x[0] + x[1]*x[1]))*(0.5-sqrt(x[0]*x[0] + x[1]*x[1])) +x[2]*x[2] <= 0.2*0.2 ? 10.0: 0.0', degree=2)
#indata = Expression('(0.5-sqrt(x[0]*x[0] + x[1]*x[1]))*(0.5-sqrt(x[0]*x[0] + x[1]*x[1])) +x[2]*x[2] <= 0.2*0.2 ? 20.0: 0.0', degree=2)
#indata = Expression('(0.5-sqrt(x[0]*x[0] + x[1]*x[1]))*(0.5-sqrt(x[0]*x[0] + x[1]*x[1])) +x[2]*x[2] <= 0.2*0.2 ? 40.0: 0.0', degree=2)

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
myfile = open('mxyz.txt', 'w') # write the mass to file
myfile2 = open('txy.txt', 'w') # write the time to file
while t <= T:
    myfile2.write("%s\n " % t)
    #myfile2.read().replace('\n', '')
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
    mass = assemble(M)
    myfile.write("%s\n " % mass)
    #print(mass)
    # Plot solution
    #plot(u)
    # Save the solution to file
    out file << (u, t)
    t += dt
    u0.assign(u)
myfile.close()
myfile2.close()
