import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# f(x,y) = |x| + |y|
def f(x, y):
    return np.abs(x) + np.abs(y)

# prox_{eta*|.|}(v) = soft-thresholding
def prox_l1(v, eta):
    return np.sign(v) * np.maximum(np.abs(v) - eta, 0.0)

# 初期点と学習率
x, y = 2.5, 1.5
eta = 0.3
steps = 15

# 軌跡を記録
pts = [(x, y, f(x, y))]
for _ in range(steps):
    v = np.array([x, y])
    x, y = prox_l1(v, eta)  # L1項のproxのみ
    pts.append((x, y, f(x, y)))
pts = np.array(pts)

# 描画用メッシュ
xr = yr = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(xr, yr)
Z = f(X, Y)

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')

# サーフェス
ax.plot_surface(X, Y, Z, cmap="plasma", alpha=0.7, linewidth=0, antialiased=False)

# 軌跡
ax.plot(pts[:,0], pts[:,1], pts[:,2], marker='o', color="red", label="prox grad path")
ax.scatter(pts[0,0], pts[0,1], pts[0,2], s=60, color="blue", label="start")
ax.scatter(pts[-1,0], pts[-1,1], pts[-1,2], s=60, color="green", label="end")

ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("f(x,y)")
ax.set_title(r"Proximal Gradient on $f(x,y)=|x|+|y|$")
ax.legend()
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# f(x,y) = |x| + |y|
def f(x, y):
    return np.abs(x) + np.abs(y)

# prox_{eta*|.|}(v) = soft-thresholding
def prox_l1(v, eta):
    return np.sign(v) * np.maximum(np.abs(v) - eta, 0.0)

# 初期点と学習率
x, y = 2.5, 1.5
eta = 0.3
steps = 15

# 近接勾配の軌跡
pts = [(x, y, f(x, y))]
for _ in range(steps):
    v = np.array([x, y])
    x, y = prox_l1(v, eta)
    pts.append((x, y, f(x, y)))
pts = np.array(pts)

# 描画用メッシュ
xr = yr = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(xr, yr)
Z = f(X, Y)

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')

# サーフェス（色味は例に合わせて alpha=0.6, グレー青系）
ax.plot_surface(X, Y, Z, alpha=0.6, linewidth=0, antialiased=True)

# 軌跡（表面上に線と点を置く）
ax.plot(pts[:,0], pts[:,1], pts[:,2], marker='o')

# 始点・終点を強調
ax.scatter(pts[0,0], pts[0,1], pts[0,2], s=60, color="blue", label="start")
ax.scatter(pts[-1,0], pts[-1,1], pts[-1,2], s=60, color="green", label="end")

ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('f(x,y)')
ax.set_title(r'Proximal Gradient on $f(x,y)=|x|+|y|$')
plt.tight_layout()
plt.show()
