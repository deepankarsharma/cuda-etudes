#include <filesystem>
#include <csignal>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "curdir.h"

#define DebugBreak() std::raise(SIGINT)

__global__ void invert_image_kernel(unsigned char* image, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		int index = (y * width + x) * 4;
		image[index + 0] = 255 - image[index + 0];
		image[index + 1] = 255 - image[index + 1];
		image[index + 2] = 255 - image[index + 2];
	}
}

__global__ void block_colors(unsigned char* image, int width, int height)
{
	// create an array of 16 distinguishable colors
	unsigned char colors[16][4] = {
		{255, 0, 0, 255},
		{0, 255, 0, 255},
		{0, 0, 255, 255},
		{255, 255, 0, 255},
		{0, 255, 255, 255},
		{255, 0, 255, 255},
		{128, 0, 0, 255},
		{0, 128, 0, 255},
		{0, 0, 128, 255},
		{128, 128, 0, 255},
		{0, 128, 128, 255},
		{128, 0, 128, 255},
		{192, 0, 0, 255},
		{0, 192, 0, 255},
		{0, 0, 192, 255},
		{192, 192, 0, 255}
	};

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		int index = (y * width + x) * 4;
		// calculate the color index based on the x and y coordinates
		int color_index = (blockIdx.x + blockIdx.y) % 16;

		image[index + 0] = colors[color_index][0];
		image[index + 1] = colors[color_index][1];
		image[index + 2] = colors[color_index][2];
		image[index + 3] = colors[color_index][3];
	}
}

// __global__ void blur(unsigned char* input, unsigned char* output, int width, int height)
// {
// 	int x = blockIdx.x * blockDim.x + threadIdx.x;
// 	int y = blockIdx.y * blockDim.y + threadIdx.y;

// 	if (x < width && y < height)
// 	{
// 		int index = (y * width + x) * 4;
// 		float r = 0.0f;
// 		float g = 0.0f;
// 		float b = 0.0f;
// 		for (int i = -1; i <= 1; i++)
// 		{
// 			for (int j = -1; j <= 1; j++)
// 			{
// 				int neighbor_x = x + i;
// 				int neighbor_y = y + j;
// 				if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height)
// 				{
// 					int neighbor_index = (neighbor_y * width + neighbor_x) * 4;
// 					r += input[neighbor_index + 0];
// 					g += input[neighbor_index + 1];
// 					b += input[neighbor_index + 2];
// 				}
// 			}
// 		}
// 		r /= 9.0f;
// 		g /= 9.0f;
// 		b /= 9.0f;
// 		output[index + 0] = (unsigned char)r;
// 		output[index + 1] = (unsigned char)g;
// 		output[index + 2] = (unsigned char)b;
// 	}
// }

struct MemImage {

    MemImage(const std::filesystem::path& path_) {
        std::filesystem::path path = current_dir() / "assets/images" / path_;
        unsigned char* image_data = stbi_load(path.string().c_str(), &width, &height, NULL, 4);
        if (image_data == NULL) {
            DebugBreak();
        }
        data = image_data;
    }

    ~MemImage() {
        stbi_image_free(data);
    }

    unsigned char* data;
    int width;
    int height;
};

void applyKernel(unsigned char* data, int width, int height) {
	// use cudaMemCpy to copy the image data to the GPU
	unsigned char* data_d;
	auto cudaStatus = cudaMalloc((void**)&data_d, 4 * width * height * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMemcpy(data_d, data, 4 * width * height * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	dim3 block(33, 33);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	block_colors << <grid, block >> > (data_d, width, height);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaMemcpy(data, data_d, 4 * width * height * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaFree(data_d);
}


int main(int argc, char** argv) {
    MemImage m_img("space1.jpg"); 
    applyKernel(m_img.data, m_img.width, m_img.height);
    stbi_write_png("/tmp/out.png", m_img.width, m_img.height, 4, m_img.data, 4 * m_img.width);
    return 0;
}