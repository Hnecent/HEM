import matplotlib.pyplot as plt
import numpy as np


def io_test():
    fig = plt.figure()  # 生成画布
    plt.ion()  # 打开交互模式
    for index in range(50):
        fig.clf()  # 清空当前Figure对象
        fig.suptitle("3d io pic")

        # 生成数据
        point_count = 100  # 随机生成100个点
        x = np.random.random(point_count)
        y = np.random.random(point_count)
        z = np.random.random(point_count)
        color = np.random.random(point_count)
        scale = np.random.random(point_count) * 100
        ax = fig.add_subplot(111, projection="3d")
        # 绘图
        ax.scatter3D(x, y, z, s=scale, c=color, marker="o")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        # 暂停
        plt.pause(0.2)

    # 关闭交互模式
    plt.ioff()

    plt.show()


if __name__ == '__main__':
    io_test()
