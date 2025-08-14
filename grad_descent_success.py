import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 例：ボウル型 f(x,y)=x^2+y^2
def f(x, y):
    return x**2 + y**2

def grad_f(x, y):
    return np.array([2*x, 2*y])

# 初期点と学習率
x, y = 2.5, 1.5
eta = 0.15
steps = 12

# 勾配降下の軌跡
pts = [(x, y, f(x, y))]
for _ in range(steps):
    g = grad_f(x, y)
    x, y = x - eta*g[0], y - eta*g[1]
    pts.append((x, y, f(x, y)))
pts = np.array(pts)

# 描画用メッシュ
xr = yr = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(xr, yr)
Z = f(X, Y)

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')

# サーフェス
ax.plot_surface(X, Y, Z, alpha=0.6, linewidth=0, antialiased=True)

# 軌跡（表面上に線と点を置く）
ax.plot(pts[:,0], pts[:,1], pts[:,2], marker='o')
# 始点・終点を強調したければ以下を解放
# ax.scatter(pts[0,0], pts[0,1], pts[0,2], s=60)
# ax.scatter(pts[-1,0], pts[-1,1], pts[-1,2], s=60)

ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('f(x,y)')
ax.set_title('Gradient Descent on 3D Surface')
plt.tight_layout()
plt.savefig("gd_3d.png", dpi=200)
plt.show()
