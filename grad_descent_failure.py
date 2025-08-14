import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

A, B = 3.0, 4.0
eta, steps = 0.05, 80

def f(x, y):
    return y**2 + x**2 + A*np.cos(B*x)

def grad_f(x, y):
    return np.array([2*x - A*B*np.sin(B*x), 2*y], dtype=float)

def run_gd(x0, y0):
    xs, ys, zs = [x0], [y0], [f(x0, y0)]
    x, y = float(x0), float(y0)
    for _ in range(steps):
        gx, gy = grad_f(x, y)
        x, y = x - eta*gx, y - eta*gy
        xs.append(x); ys.append(y); zs.append(f(x, y))
    return np.array(xs), np.array(ys), np.array(zs)

# ② 離れた浅い極小へ
xl, yl, zl = run_gd(2.8, 1.0)

# サーフェス
xs = ys = np.linspace(-4, 4, 300)
X, Y = np.meshgrid(xs, ys)
Z = f(X, Y)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.65, linewidth=0)

ax.plot(xl, yl, zl, 'o-', label='to local min (far well)')

# 参照点を注記
ax.scatter(3.0, 0, f(3.0,0), s=60)

ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('f(x,y)')
ax.set_title('Simple function: local minima')
ax.legend(); plt.tight_layout(); plt.show()
