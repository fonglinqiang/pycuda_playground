import numpy as np
from pycuda import driver as cuda, compiler, autoinit
from timeit import default_timer as timer


module = compiler.SourceModule("""
    __global__ void matrix_multiplication(float*a, float *b, float * c, int n)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
                      
        float temp_sum = 0;
        
        if (row < n && col < n) {
            for (int k = 0; k < n; k++) {
                temp_sum += a[row * n + k] * b[k * n + col];       
            }
        }
        c[row*n + col] = temp_sum;
    }
    """)

if __name__ == "__main__":
    matrix_multiplication = module.get_function('matrix_multiplication')

    # general setting
    n = 1024
    block_size = (16,16,1)
    grid_size = (64,64,1)

    # initialise variables
    n_np = np.int32(n)
    a = np.random.randn(n,n).astype(np.float32)
    b = np.random.randn(n,n).astype(np.float32)
    c = np.empty((n,n),dtype=np.float32)

    gpu_time = timer()
    matrix_multiplication(cuda.In(a),cuda.In(b),cuda.Out(c),n_np,block=block_size,grid=grid_size)
    gpu_time = timer() - gpu_time
    print(f'GPU time: {gpu_time}')
    print(c[:3])

    cpu_time = timer()
    c_cpu = np.dot(a,b)
    cpu_time = timer() - cpu_time
    print(f'CPU time: {cpu_time}')
    print(c_cpu[:3])

    # print((c - c_cpu)[:10,:10])
