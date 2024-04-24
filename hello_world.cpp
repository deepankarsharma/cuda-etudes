// Part of ImGui Bundle - MIT License - Copyright (c) 2022-2024 Pascal Thomet - https://github.com/pthom/imgui_bundle
#include "immapp/immapp.h"
#ifdef IMGUI_BUNDLE_WITH_IMPLOT
#include "implot/implot.h"
#endif
#include "imgui_md_wrapper.h"
#include "stb_image.h"
#include <GLFW/glfw3.h>
#include <cmath>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <filesystem>


__global__ void cuda_hello() {
    printf("Hello World from GPU!\n");
}


// cuda kernel to invert an image
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

// struct that contains GLuint texture id and image size
struct ImageData
{
	GLuint texture;
	ImVec2 size;
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

	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	invert_image_kernel << <grid, block >> > (data_d, width, height);

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

// Simple helper function to load an image into a OpenGL texture with common settings
bool LoadTextureFromFile(const char* filename, GLuint* out_texture, int* out_width, int* out_height)
{
	// Load from file
	int image_width = 0;
	int image_height = 0;
	unsigned char* image_data = stbi_load(filename, &image_width, &image_height, NULL, 4);

	
	if (image_data == NULL)
		return false;
	applyKernel(image_data, image_width, image_height);

	// Create a OpenGL texture identifier
	GLuint image_texture;
	glGenTextures(1, &image_texture);
	glBindTexture(GL_TEXTURE_2D, image_texture);

	// Setup filtering parameters for display
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Upload pixels into texture
#if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
	glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
#endif
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data);
	stbi_image_free(image_data);

	*out_texture = image_texture;
	*out_width = image_width;
	*out_height = image_height;

	return true;
}

ImageData ImageRen(const char* filename)
{

	int my_image_width = 0;
	int my_image_height = 0;
	GLuint my_image_texture = 0;
	bool ret = LoadTextureFromFile(filename, &my_image_texture, &my_image_width, &my_image_height);
	IM_ASSERT(ret);

	return { my_image_texture, ImVec2(my_image_width, my_image_height) };
}

void Gui()
{
	auto path = std::filesystem::current_path() / "assets/images/world.png";
	auto path_str = path.string();
	static ImageData image = ImageRen(path_str.c_str());
	ImGui::Image((void*)(intptr_t)image.texture, image.size);
	HelloImGui::ImageFromAsset("images/world.png");
}


int main(int , char *[])
{
    HelloImGui::SimpleRunnerParams runnnerParams;
    runnnerParams.guiFunction = Gui;
    runnnerParams.windowSize = {800, 800};

    ImmApp::AddOnsParams addOnsParams;
    addOnsParams.withMarkdown = true;
    addOnsParams.withImplot = true;

    ImmApp::Run(runnnerParams, addOnsParams);
    return 0;
}
