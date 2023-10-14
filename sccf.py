from math import sin, exp, pi
import matplotlib.pyplot as plt
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.autoinit


class Sccf:
    def __init__(self, t_imp, tau, f0, coef_dts0, dt0, fs):
        self.coef_dts0 = coef_dts0
        self.period_sample = 10 ** (-11)
        self.dts0 = coef_dts0 * self.period_sample
        self.tau = tau * 10 ** (-6)
        self.f0 = f0 * 10 ** 6
        self.dt0 = dt0 * 10 ** (-6)
        self.fs = fs * 10 ** 6

        self.t = np.arange(0.0, t_imp * 10 ** (-6), self.period_sample)
        self.tu_samples = np.arange(0.0 + self.dts0, t_imp * 10 ** (-6) + self.dts0, self.period_sample)
        self.tw_samples = [x + self.dt0 for x in self.tu_samples]

        self.y0 = self.create_y0()
        self.y1 = self.create_y1()
        self.y2 = self.create_y2()
        self.u = []
        self.w = []
        self.c = np.array([], dtype=np.float32)
        self.c_sum = np.array([], dtype=np.float32)

    def run(self):
        self.sampling()
        self.create_c()
        self.create_csum()

    def create_c(self):
        pass

    def create_csum(self):
        pass

    def create_y0(self):
        return [(1 - exp(1) ** (-ti / self.tau) - (ti / self.tau) * exp(1) ** (
                -ti / self.tau)) * sin(2 * pi * self.f0 * ti) for ti in self.t]

    def create_y1(self):
        return [
            (1 - exp(1) ** ((-ti + self.dts0) / self.tau) - (ti / self.tau) * exp(1) ** (
                    (-ti + self.dts0) / self.tau)) * sin(2 * pi * self.f0 * (ti + self.dts0)) for ti in self.tu_samples]

    def create_y2(self):
        return [
            (1 - exp(1) ** (-(ti - self.dt0 - self.dts0) / self.tau) - (ti / self.tau) * exp(1) ** (
                    -(ti - self.dt0 - self.dts0) / self.tau)) * sin(2 * pi * self.f0 * (ti + self.dt0 + self.dts0))
            for ti in self.tw_samples]

    def sampling(self):
        self.u = self.y1[::int(1 / self.fs / 10 ** (-11))]
        self.w = self.y2[::int(1 / self.fs / 10 ** (-11))]
        self.tu_samples = self.tu_samples[::int(1 / self.fs / 10 ** (-11))]
        self.tw_samples = self.tw_samples[::int(1 / self.fs / 10 ** (-11))]


class SccfCPU(Sccf):
    def create_c(self):
        n = len(self.u)
        m = n - int(n * 0.4)
        c = np.zeros(shape=(n, n))
        i = 0
        while i < n:
            j = 0
            k = i
            while j < m:
                if j <= i:
                    c[i][j] = self.u[k] * self.w[i]
                    k -= 1
                j += 1
            i += 1
        self.c = c

    def create_csum(self):
        self.c_sum = self.c.sum(axis=0)


class SccfGPU(Sccf):
    def create_c(self):
        n = len(self.u)
        m = n - int(n * 0.4)

        u_host = np.array(self.u).astype(np.float32)
        w_host = np.array(self.w).astype(np.float32)
        c_host = np.zeros(shape=(n, n), dtype=np.float32)

        u_dev = cuda.mem_alloc(n * u_host.dtype.itemsize)
        w_dev = cuda.mem_alloc(n * w_host.dtype.itemsize)
        c_dev = cuda.mem_alloc(c_host.size * c_host.dtype.itemsize)

        cuda.memcpy_htod(u_dev, u_host)
        cuda.memcpy_htod(w_dev, w_host)
        cuda.memcpy_htod(c_dev, c_host)

        mod = SourceModule("""
            __global__ void convolution(float *u, float *w, float *c, int n, int m)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    for (int i = 0; i < m; i++)
                        if (idx >= i)
                            c[idx*n + i] = u[idx-i] * w[idx];
                }
            }
        """)

        convolution = mod.get_function("convolution")

        block_size = 256
        grid_size = (n + block_size - 1) // block_size
        convolution(u_dev, w_dev, c_dev, np.int32(n), np.int32(m), block=(block_size, 1, 1), grid=(grid_size, 1, 1))

        cuda.memcpy_dtoh(c_host, c_dev)
        self.c = c_host

    def create_csum(self):
        n = len(self.u)
        m = n - int(n * 0.4)

        c_host = self.c.astype(np.float32)
        csum_host = np.empty(m, 'float32')

        c_dev = cuda.mem_alloc(c_host.size * c_host.dtype.itemsize)
        csum_dev = cuda.mem_alloc(m * csum_host.dtype.itemsize)

        cuda.memcpy_htod(c_dev, c_host)

        mod = SourceModule("""
            __global__ void column_sum(float *c, float *csum, int n, int m)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    float sum = 0.0;
                    for (int i = 0; i < n; i++) {
                        sum += c[i*n+idx];
                    }
                    csum[idx] = sum;
                }
            }
        """)

        column_sum = mod.get_function("column_sum")

        block_size = 1024
        grid_size = (n + block_size - 1) // block_size
        column_sum(c_dev, csum_dev, np.int32(n), np.int32(m), block=(block_size, 1, 1),
                   grid=(grid_size, 1, 1))

        cuda.memcpy_dtoh(csum_host, csum_dev)
        self.c_sum = csum_host


if __name__ == "__main__":
    n = 500
    m = n - int(n * 0.4)

    u = np.array([np.random.uniform(0, 20) for x in range(n)]).astype(np.float32)
    w = np.array([np.random.uniform(0, 20) for x in range(m, n+m)]).astype(np.float32)
    c = np.zeros(shape=(n, n), dtype=np.float32)

    u_gpu = cuda.mem_alloc(n * u.dtype.itemsize)
    w_gpu = cuda.mem_alloc(n * u.dtype.itemsize)
    c_gpu = cuda.mem_alloc(n * n * c.dtype.itemsize)

    cuda.memcpy_htod(u_gpu, u)
    cuda.memcpy_htod(w_gpu, w)
    cuda.memcpy_htod(c_gpu, c)

    mod = SourceModule("""
                        __global__ void convolution(float *u, float *w, float *c, int n, int m)
                        {
                            int idx = blockIdx.x * blockDim.x + threadIdx.x;
                            if (idx < n) {
                                for (int i = 0; i < m; i++){
                                    if (idx >= i)
                                        c[idx*n + i] = u[idx-i] * w[idx];
                                }
                            }
                        }
                    """)

    column_sum = mod.get_function("convolution")

    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    column_sum(u_gpu, w_gpu, c_gpu, np.int32(n), np.int32(m), block=(block_size, 1, 1), grid=(grid_size, 1, 1))

    cuda.memcpy_dtoh(c, c_gpu)



    c2 = np.zeros(shape=(n, n), dtype=np.float32)
    i = 0
    while i < n:
        j = 0
        k = i
        while j < m:
            if j <= i:
                c2[i][j] = u[k] * w[i]
                print(f'{i} {k}', end="  ")
                k -= 1
            j += 1
        i += 1
        print()



    _, axes = plt.subplots(1, 3)
    axes[0].imshow(c2 - c)
    axes[1].imshow(c)
    axes[2].imshow(c2)
    plt.show()
