import numpy as np
import matplotlib.pyplot as plt

# 例の関数と勾配
def f(x, y):
    return x**2 + y**2

def grad_f(x, y):
    return np.array([2*x, 2*y])

# 初期点
x, y = 2.5, 1.5
eta = 0.1
points = [(x, y)]

# 勾配降下法
for _ in range(10):
    g = grad_f(x, y)
    x, y = x - eta*g[0], y - eta*g[1]
    points.append((x, y))

points = np.array(points)

# 等高線プロット
X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z = f(X, Y)

plt.figure(figsize=(6,6))
plt.contour(X, Y, Z, levels=20)
plt.plot(points[:,0], points[:,1], 'o-', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent Path')
plt.grid(True)
plt.show()
