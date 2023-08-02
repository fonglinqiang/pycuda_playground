import cv2
import pycuda.autoinit
from pycuda import compiler
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np

# Define the CUDA kernel function
module = compiler.SourceModule("""
__global__ void convolution_kernel(float *image, float *kernel, float *output, int width, int height, int kernel_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        int half_kernel_size = kernel_size / 2;
        
        for (int i = -half_kernel_size; i <= half_kernel_size; i++) {
            for (int j = -half_kernel_size; j <= half_kernel_size; j++) {
                int image_x = x + j;
                int image_y = y + i;
                
                if (image_x >= 0 && image_x < width && image_y >= 0 && image_y < height) {
                    float image_value = image[image_y * width + image_x];
                    float kernel_value = kernel[(i + half_kernel_size) * kernel_size + (j + half_kernel_size)];
                    sum += image_value * kernel_value;
                }
            }
        }
        
        output[y * width + x] = sum;
    }
}
""")
                               
# Get the kernel function from the module
convolution_kernel = module.get_function("convolution_kernel")

if __name__=="__main__":
    # Load the image using OpenCV
    image = cv2.imread("monarch_butterfly.jpg").astype(np.float32)

    # Split the image into 3 channels
    image_b, image_g, image_r = cv2.split(image)

    # Define the convolution kernel
    kernel = np.array([[1/16, 2/16, 1/16], [2/16,4/16,2/16], [1/16, 2/16, 1/16]], dtype=np.float32)

    # Allocate device memory and copy data to it
    image_b_gpu = gpuarray.to_gpu(image_b)
    image_g_gpu = gpuarray.to_gpu(image_g)
    image_r_gpu = gpuarray.to_gpu(image_r)
    kernel_gpu = gpuarray.to_gpu(kernel)

    # Allocate output memory
    output_b_gpu = gpuarray.empty_like(image_b_gpu)
    output_g_gpu = gpuarray.empty_like(image_g_gpu)
    output_r_gpu = gpuarray.empty_like(image_r_gpu)

    # Set the block and grid sizes
    block_size = (16, 16, 1)
    grid_size = ((image_b.shape[1] + block_size[0] - 1) // block_size[0], (image_b.shape[0] + block_size[1] - 1) // block_size[1], 1)

    # Call the kernel function for each channel
    convolution_kernel(image_b_gpu, kernel_gpu, output_b_gpu, np.int32(image_b.shape[1]), np.int32(image_b.shape[0]), np.int32(kernel.shape[0]), block=block_size, grid=grid_size)
    convolution_kernel(image_g_gpu, kernel_gpu, output_g_gpu, np.int32(image_g.shape[1]), np.int32(image_g.shape[0]), np.int32(kernel.shape[0]), block=block_size, grid=grid_size)
    convolution_kernel(image_r_gpu, kernel_gpu, output_r_gpu, np.int32(image_r.shape[1]), np.int32(image_r.shape[0]), np.int32(kernel.shape[0]), block=block_size, grid=grid_size)

    # Copy the results back to the host
    output_b = output_b_gpu.get()
    output_g = output_g_gpu.get()
    output_r = output_r_gpu.get()

    # Merge the output channels back into an image
    output = cv2.merge((output_b, output_g, output_r))

    # Save the result to a file
    cv2.imwrite("monarch_butterfly_blurred.jpg", output.astype(np.uint8))