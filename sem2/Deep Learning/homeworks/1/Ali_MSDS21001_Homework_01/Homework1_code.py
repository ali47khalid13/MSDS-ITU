from sympy import *
from sympy.plotting.plot import MatplotlibBackend, Plot

def get_sympy_subplots(plot: Plot):
    backend = MatplotlibBackend(plot)

    backend.process_series()
    backend.fig.tight_layout()
    return backend.fig, backend.ax[0]

######## QUESTION 1, PART 1 ################
x,y= symbols('x y', real= True)
f = (x**4 + y**4 + 16*x*y)
df_x = diff(f,x) 
df_y = diff(f,y)
critical_points = solve([df_x,df_y],(x,y))
f_at_cp = [f.subs([(x,a[0]),(y,a[1])]) for a in critical_points]
print(critical_points)
print(f_at_cp)
p1 = plotting.plot3d(f, (x, -5,5), (y, -5,5), show = False)
fig, axe = get_sympy_subplots(p1)
x_cp = [critical_points[i][0] for i in range(0,len(critical_points))]
y_cp = [critical_points[i][1] for i in range(0,len(critical_points))]
axe.plot(x_cp,y_cp,f_at_cp,'o')
p1.show()

######## QUESTION 1, PART 2 ################
# x,y= symbols('x y', real= True)
# f = (sqrt(x**2 + y**2) + 1)
# df_x = diff(f,x) 
# df_y = diff(f,y)
# critical_points = solve([df_x,df_y],(x,y))
# f_at_cp = [f.subs([(x,a[0]),(y,a[1])]) for a in critical_points]
# print(critical_points)
# print(f_at_cp)
# p1 = plotting.plot3d(f, (x, -5,5), (y, -5,5), show = False)
# fig, axe = get_sympy_subplots(p1)
# x_cp = [critical_points[i][0] for i in range(0,len(critical_points))]
# y_cp = [critical_points[i][1] for i in range(0,len(critical_points))]
# axe.plot(x_cp,y_cp,f_at_cp,'o')
# p1.show()

######## QUESTION 1, PART 3 ################
# x,y= symbols('x y', real= True)
# f = exp(-(x**2 + y**2+ 2*x))
# df_x = diff(f,x) 
# print(df_x)
# df_y = diff(f,y)
# print(df_y)
# critical_points = solve([df_x,df_y],(x,y))
# f_at_cp = [f.subs([(x,a[0]),(y,a[1])]) for a in critical_points]
# print(critical_points)
# print(f_at_cp)
# p1 = plotting.plot3d(f, (x, -5,5), (y, -5,5), show = False)
# fig, axe = get_sympy_subplots(p1)
# x_cp = [critical_points[i][0] for i in range(0,len(critical_points))]
# y_cp = [critical_points[i][1] for i in range(0,len(critical_points))]
# axe.plot(x_cp,y_cp,f_at_cp,'o')
# p1.show()

######## QUESTION 2, PART 1 ################
# x, y = symbols("x y",real=True)
# f =  x*exp(y) - exp(x)
# gradient = derive_by_array(f, (x, y))
# print(gradient)
# hessian = Matrix(derive_by_array(gradient, (x, y)))
# print(hessian)
# stationary_points = solve(gradient, (x, y))
# f_at_cp = [f.subs([(x,a[0]),(y,a[1])]) for a in stationary_points]
# print(stationary_points)
# for p in stationary_points:
#     value = f.subs({x: p[0], y: p[1]})
#     hess = hessian.subs({x: p[0], y: p[1]})
#     print(hess)
#     eigenvals = hess.eigenvals()
#     if all(ev > 0 for ev in eigenvals):
#         print("Local minimum at {} with value {}".format(p, value))
#     elif all(ev < 0 for ev in eigenvals):
#         print("Local maximum at {} with value {}".format(p, value))
#     elif any(ev > 0 for ev in eigenvals) and any(ev < 0 for ev in eigenvals):
#         print("Saddle point at {} with value {}".format(p, value))
#     else:
#         print("Could not classify the stationary point at {} with value {}".format(p, value))

# p1 = plotting.plot3d(f, (x, -5,10), (y, -5,10), show = False)
# fig, axe = get_sympy_subplots(p1)
# x_cp = [stationary_points[i][0] for i in range(0,len(stationary_points))]
# y_cp = [stationary_points[i][1] for i in range(0,len(stationary_points))]
# axe.plot(x_cp,y_cp,f_at_cp,'o')
# p1.show()

# ####### QUESTION 2, PART 2 ################
# x, y = symbols("x y",real=True)
# f =  x*sin(y)
# gradient = derive_by_array(f, (x, y))
# print(gradient)
# hessian = Matrix(derive_by_array(gradient, (x, y)))
# print(hessian)
# stationary_points = solve(gradient, (x, y))
# f_at_cp = [f.subs([(x,a[0]),(y,a[1])]) for a in stationary_points]
# print(stationary_points)
# for p in stationary_points:
#     value = f.subs({x: p[0], y: p[1]})
#     hess = hessian.subs({x: p[0], y: p[1]})
#     print(hess)
#     eigenvals = hess.eigenvals()
#     if all(ev > 0 for ev in eigenvals):
#         print("Local minimum at {} with value {}".format(p, value))
#     elif all(ev < 0 for ev in eigenvals):
#         print("Local maximum at {} with value {}".format(p, value))
#     elif any(ev > 0 for ev in eigenvals) and any(ev < 0 for ev in eigenvals):
#         print("Saddle point at {} with value {}".format(p, value))
#     else:
#         print("Could not classify the stationary point at {} with value {}".format(p, value))

# p1 = plotting.plot3d(f, (x, -5,10), (y, -5,10), show = False)
# fig, axe = get_sympy_subplots(p1)
# x_cp = [stationary_points[i][0] for i in range(0,len(stationary_points))]
# y_cp = [stationary_points[i][1] for i in range(0,len(stationary_points))]
# axe.plot(x_cp,y_cp,f_at_cp,'o')
# p1.show()

####### QUESTION 2, PART 3 ################
# x, y = symbols("x y",real=True)
# f =  4*x*y - x**4 - y**4
# gradient = derive_by_array(f, (x, y))
# hessian = Matrix(derive_by_array(gradient, (x, y)))
# print(hessian)
# stationary_points = solve(gradient, (x, y))
# f_at_cp = [f.subs([(x,a[0]),(y,a[1])]) for a in stationary_points]
# print(stationary_points)
# for p in stationary_points:
#     value = f.subs({x: p[0], y: p[1]})
#     hess = hessian.subs({x: p[0], y: p[1]})
#     print(hess)
#     eigenvals = hess.eigenvals()
#     if all(ev > 0 for ev in eigenvals):
#         print("Local minimum at {} with value {}".format(p, value))
#     elif all(ev < 0 for ev in eigenvals):
#         print("Local maximum at {} with value {}".format(p, value))
#     elif any(ev > 0 for ev in eigenvals) and any(ev < 0 for ev in eigenvals):
#         print("Saddle point at {} with value {}".format(p, value))
#     else:
#         print("Could not classify the stationary point at {} with value {}".format(p, value))

# p1 = plotting.plot3d(f, (x, -5,10), (y, -5,10), show = False)
# fig, axe = get_sympy_subplots(p1)
# x_cp = [stationary_points[i][0] for i in range(0,len(stationary_points))]
# y_cp = [stationary_points[i][1] for i in range(0,len(stationary_points))]
# axe.plot(x_cp,y_cp,f_at_cp,'o')
# p1.show()