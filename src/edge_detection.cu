#include <stdio.h>
#include <cuda_runtime.h>

__global__ void edgeDetect(int *input, int *output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n)
    {
        output[idx] = abs(input[idx] - input[idx+1]);
    }
}

int main()
{
    int n = 1024;

    int *h_input, *h_output;
    int *d_input, *d_output;

    h_input = (int*)malloc(n*sizeof(int));
    h_output = (int*)malloc(n*sizeof(int));

    for(int i=0;i<n;i++)
        h_input[i] = i;

    cudaMalloc(&d_input,n*sizeof(int));
    cudaMalloc(&d_output,n*sizeof(int));

    cudaMemcpy(d_input,h_input,n*sizeof(int),cudaMemcpyHostToDevice);

    edgeDetect<<<4,256>>>(d_input,d_output,n);

    cudaMemcpy(h_output,d_output,n*sizeof(int),cudaMemcpyDeviceToHost);

    printf("GPU Edge Detection Completed\n");

    cudaFree(d_input);
    cudaFree(d_output);

    free(h_input);
    free(h_output);

    return 0;
}
