from pycuda import driver as cuda, compiler#, autoinit
import numpy as np
from timeit import default_timer as timer

cuda.init()
device = cuda.Device(0)
context = device.make_context()

module = compiler.SourceModule("""
    __global__ void vector_addition(float *a, float *b, float *c, int n)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (tid < n) {
            c[tid] = a[tid] + b[tid];
        }
    }
""")

if __name__ == "__main__":
    vector_addition = module.get_function('vector_addition')

    # general settings
    mode = 'manual' # 'manual' or 'auto'
    print(f'mode: {mode}')
    n = np.int32(1000000)
    block_num = 1024
    grid_num = int(n/block_num) + 1
    print(f'number of threads: {n}, block_size: ({block_num},1,1), grid_size: ({grid_num},1,1)')

    # initialise variables
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    c = np.empty(n).astype(np.float32)

    start = timer()
    if mode == 'manual':
        # gpu manual memory allocation
        a_gpu = cuda.mem_alloc(a.nbytes)
        b_gpu = cuda.mem_alloc(b.nbytes)
        c_gpu = cuda.mem_alloc(c.nbytes)

        # host to device
        cuda.memcpy_htod(a_gpu,a)
        cuda.memcpy_htod(b_gpu,b)
        cuda.memcpy_htod(c_gpu,c)

        # run function
        vector_addition(a_gpu,b_gpu,c_gpu,n,block=(block_num,1,1),grid=(grid_num,1,1))

        # device to host
        cuda.memcpy_dtoh(c,c_gpu)

        # Free device memory
        a_gpu.free()
        b_gpu.free()
        c_gpu.free()

    elif mode == 'auto':
        # gpu auto memory allocation
        vector_addition(cuda.In(a),cuda.In(b),cuda.Out(c),n,block=(block_num,1,1),grid=(grid_num,1,1))
    end = timer() - start

    # Destroy the CUDA context
    context.pop()

    print('gpu\n',c[:10])
    print(end)

    # cpu check
    c_cpu = np.empty(n).astype(np.float32)
    start = timer()
    for i in range(len(a)):
        c_cpu[i] = a[i]+b[i]
    end = timer() - start

    print('cpu\n',c_cpu[:10])
    print(end)