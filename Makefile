VULKAN_SDK_PATH = /home/sasha/vulkan/1.2.154.0/x86_64

CFLAGS = -ggdb -std=c++17 -I$(VULKAN_SDK_PATH)/include
LDFLAGS = -L$(VULKAN_SD_PATH)/lib `pkg-config --static --libs glfw3` -lvulkan

Triangle: main.cpp shaders/vert.spv shaders/frag.spv
	clang++ main.cpp $(CFLAGS) $(LDFLAGS) -o Triangle

shaders/vert.spv: shaders/shader.vert
	glslc shaders/shader.vert -o shaders/vert.spv

shaders/frag.spv: shaders/shader.frag
	glslc shaders/shader.frag -o shaders/frag.spv

.PHONY: test clean

test: Triangle
	LD_LIBRARY_PATH=$(VULKAN_SDK_PATH)/lib \
	VK_LAYER_PATH=$(VULKAN_SDK_PATH)/etc/vulkan/explicit_layer.d \
	./Triangle

debug: Triangle
	LD_LIBRARY_PATH=$(VULKAN_SDK_PATH)/lib \
	VK_LAYER_PATH=$(VULKAN_SDK_PATH)/etc/vulkan/explicit_layer.d \
	gdb ./Triangle

clean:
	rm -f Triangle shaders/vert.spv shaders/frag.spv

