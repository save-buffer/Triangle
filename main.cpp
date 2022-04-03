#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

struct Vertex
{
    glm::vec2 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription GetBindingDescription()
    {
        VkVertexInputBindingDescription desc = {};
        desc.binding = 0;
        desc.stride = sizeof(Vertex);
        desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return desc;
    }

    static std::array<VkVertexInputAttributeDescription, 2> GetAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 2> descs = {};
        descs[0].binding = 0;
        descs[0].location = 0;
        descs[0].format = VK_FORMAT_R32G32_SFLOAT;
        descs[0].offset = offsetof(Vertex, pos);

        descs[1].binding = 0;
        descs[1].location = 1;
        descs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        descs[1].offset = offsetof(Vertex, color);
        return descs;
    }
};

struct UniformBufferObject
{
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

const std::vector<Vertex> vertices =
{
    { { -0.5f, -0.5f }, { 1.0f, 0.0f, 0.0f } },
    { {  0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f } },
    { {  0.5f,  0.5f }, { 0.0f, 0.0f, 1.0f } },
    { { -0.5f,  0.5f }, { 1.0f, 1.0f, 1.0f } },
};

const std::vector<uint16_t> indices = { 0, 1, 2, 2, 3, 0 };

template <typename A, typename B, typename C>
auto Clamp(A a, B mn, C mx)
{
    return std::min(mx, std::max(a, mn));
}

std::vector<char> ReadFile(const std::string &filename)
{
    std::ifstream f(filename, std::ios::ate | std::ios::binary);
    if(!f.is_open())
        throw std::runtime_error("Failed to open file " + filename + "!");
    size_t sz = static_cast<size_t>(f.tellg());
    std::vector<char> buff(sz);
    f.seekg(0);
    f.read(buff.data(), sz);
    f.close();
    return buff;
}

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT *createInfo,
    const VkAllocationCallbacks *allocator,
    VkDebugUtilsMessengerEXT *debugMessenger)
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if(func)
        return func(instance, createInfo, allocator, debugMessenger);
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void DestroyDebugUtilsMessengerEXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT debugMessenger,
    VkAllocationCallbacks *allocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if(func)
        func(instance, debugMessenger, allocator);
}

class TriangleApp
{
public:
    void Run()
    {
        InitWindow();
        InitVulkan();
        MainLoop();
        Cleanup();
    }

private:
    void InitWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        m_window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(m_window, this);
        glfwSetFramebufferSizeCallback(m_window, FramebufferResizeCallback);
    }

    void PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo)
    {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = DebugCallback;
        createInfo.pUserData = nullptr;
    }

    void SetupDebugMessenger()
    {
        if(!ENABLE_VALIDATION_LAYERS)
            return;
        VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
        PopulateDebugMessengerCreateInfo(createInfo);
        if(CreateDebugUtilsMessengerEXT(m_instance, &createInfo, nullptr, &m_debugMessenger) != VK_SUCCESS)
            throw std::runtime_error("Failed to setup debug messenger!");
    }

    struct SwapChainSupport
    {
        VkSurfaceCapabilitiesKHR m_capabilities;
        std::vector<VkSurfaceFormatKHR> m_formats;
        std::vector<VkPresentModeKHR> m_presentModes;
    };

    VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats)
    {
        for(const auto &format : availableFormats)
        {
            if(format.format == VK_FORMAT_B8G8R8A8_SRGB
               && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                return format;
            }
        }
        return availableFormats[0];
    }

    VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes)
    {
        for(const auto &presentMode : availablePresentModes)
        {
            if(presentMode == VK_PRESENT_MODE_MAILBOX_KHR)
                return presentMode;
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities)
    {
        if(capabilities.currentExtent.width != UINT32_MAX)
            return capabilities.currentExtent;

        int width, height;
        glfwGetFramebufferSize(m_window, &width, &height);
        VkExtent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
        actualExtent.width = Clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = Clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
        return actualExtent;
    }

    SwapChainSupport QuerySwapChainSupport(VkPhysicalDevice device)
    {
        SwapChainSupport support;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, m_surface, &support.m_capabilities);

        uint32_t nFormats;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &nFormats, nullptr);
        if(nFormats)
        {
            support.m_formats.resize(nFormats);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &nFormats, support.m_formats.data());
        }

        uint32_t nPresentModes;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_surface, &nPresentModes, nullptr);
        if(nPresentModes)
        {
            support.m_presentModes.resize(nPresentModes);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_surface, &nPresentModes, support.m_presentModes.data());
        }
        return support;
    }

    struct QueueFamilyIndices
    {
        std::optional<uint32_t> m_graphicsFamily;
        std::optional<uint32_t> m_presentFamily;

        bool IsComplete()
        {
            return m_graphicsFamily.has_value()
                && m_presentFamily.has_value();
        }
    };

    QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device)
    {
        QueueFamilyIndices indices;
        
        uint32_t nQueueFamilies;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &nQueueFamilies, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(nQueueFamilies);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &nQueueFamilies, queueFamilies.data());
        int i = 0;
        for(const auto &queueFamily : queueFamilies)
        {
            if(queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
                indices.m_graphicsFamily = i;

            VkBool32 presentSupport;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_surface, &presentSupport);
            if(presentSupport)
                indices.m_presentFamily = i;

            if(indices.IsComplete())
                break;
            i++;
        }
        return indices;
    }

    bool CheckDeviceExtensionSupport(VkPhysicalDevice device)
    {
        uint32_t nExtensions;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &nExtensions, nullptr);

        std::vector<VkExtensionProperties> extensions(nExtensions);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &nExtensions, extensions.data());

        std::set<std::string> requiredExtensions(m_deviceExtensions.begin(), m_deviceExtensions.end());
        for(const auto &extension : extensions)
            requiredExtensions.erase(extension.extensionName);

        return requiredExtensions.empty();
    }

    bool IsDeviceSuitable(VkPhysicalDevice device)
    {
#if 0
        // Example device suitability checks:
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);

        VkPhysicalDeviceFeatures features;
        vkGetPhysicalDeviceFeatures(device, &features);

        return properties.deviceType == VK_PHYSICAL_DEVICE_DISCRETE_GPU
            && features.geometryShader;
#endif
        QueueFamilyIndices indices = FindQueueFamilies(device);
        bool extensionsSupported = CheckDeviceExtensionSupport(device);
        bool swapChainAdequate = false;
        if(extensionsSupported)
        {
            SwapChainSupport swapChainSupport = QuerySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.m_formats.empty()
                && !swapChainSupport.m_presentModes.empty();
        }

        VkPhysicalDeviceFeatures features;
        vkGetPhysicalDeviceFeatures(device, &features);
        return indices.IsComplete()
            && extensionsSupported
            && swapChainAdequate
            && features.samplerAnisotropy;
    }

    void PickPhysicalDevice()
    {
        uint32_t nDevices;
        vkEnumeratePhysicalDevices(m_instance, &nDevices, nullptr);
        if(!nDevices)
            throw std::runtime_error("Failed to find a GPU with Vulkan support!");
        std::vector<VkPhysicalDevice> devices(nDevices);
        vkEnumeratePhysicalDevices(m_instance, &nDevices, devices.data());
        auto suitable = std::find_if(devices.begin(), devices.end(), [&](auto &x) { return IsDeviceSuitable(x); });
        if(suitable == devices.end())
            throw std::runtime_error("Failed to find a suitable GPU!");
        m_physicalDevice = *suitable;
    }
    
    uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProps);
        for(uint32_t i = 0; i < memProps.memoryTypeCount; i++)
            if(typeFilter & (1 << i)
               && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        throw std::runtime_error("Failed to find a suitable memory type!");
    }

    void CreateLogicalDevice()
    {
        QueueFamilyIndices indices = FindQueueFamilies(m_physicalDevice);
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.m_graphicsFamily.value(), indices.m_presentFamily.value() };
        float queuePriority = 1.0f;

        for(uint32_t queueFamily : uniqueQueueFamilies)
        {
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures = {};
        deviceFeatures.samplerAnisotropy = VK_TRUE;

        VkDeviceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(m_deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = m_deviceExtensions.data();
        if(ENABLE_VALIDATION_LAYERS)
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(m_validationLayers.size());
            createInfo.ppEnabledLayerNames = m_validationLayers.data();
        }
        else
        {
            createInfo.enabledLayerCount = 0;
            createInfo.ppEnabledLayerNames = nullptr;
        }
        
        if(vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device) != VK_SUCCESS)
            throw std::runtime_error("Failed to create logical device!");

        vkGetDeviceQueue(m_device, indices.m_graphicsFamily.value(), 0, &m_graphicsQueue);
        vkGetDeviceQueue(m_device, indices.m_presentFamily.value(), 0, &m_presentQueue);
    }

    void CreateSurface()
    {
        if(glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface) != VK_SUCCESS)
            throw std::runtime_error("Failed to create window surface!");
    }

    void CreateSwapChain()
    {
        SwapChainSupport support = QuerySwapChainSupport(m_physicalDevice);
        VkSurfaceFormatKHR surfaceFormat = ChooseSwapSurfaceFormat(support.m_formats);
        VkPresentModeKHR presentMode = ChooseSwapPresentMode(support.m_presentModes);
        VkExtent2D extent = ChooseSwapExtent(support.m_capabilities);

        uint32_t nImages = support.m_capabilities.minImageCount + 1;
        if(support.m_capabilities.maxImageCount > 0
           && nImages > support.m_capabilities.maxImageCount)
            nImages = support.m_capabilities.maxImageCount;

        VkSwapchainCreateInfoKHR createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = m_surface;
        createInfo.minImageCount = nImages;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = FindQueueFamilies(m_physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.m_graphicsFamily.value(), indices.m_presentFamily.value() };
        if(indices.m_graphicsFamily != indices.m_presentFamily)
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = nullptr;
        }
        createInfo.preTransform = support.m_capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;
        if(vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapChain) != VK_SUCCESS)
            throw std::runtime_error("Failed to create swap chain!");

        uint32_t nSwapChainImages;
        vkGetSwapchainImagesKHR(m_device, m_swapChain, &nSwapChainImages, nullptr);
        m_swapChainImages.resize(nSwapChainImages);
        vkGetSwapchainImagesKHR(m_device, m_swapChain, &nSwapChainImages, m_swapChainImages.data());
        m_swapChainImageFormat = surfaceFormat.format;
        m_swapChainExtent = extent;
    }

    void CreateImageViews()
    {
        m_swapChainImageViews.resize(m_swapChainImages.size());
        for(int i = 0; i < m_swapChainImages.size(); i++)
            m_swapChainImageViews[i] = CreateImageView(m_swapChainImages[i], m_swapChainImageFormat);
    }

    void CreateRenderPass()
    {
        VkAttachmentDescription colorAttachment = {};
        colorAttachment.format = m_swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if(vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_renderPass) != VK_SUCCESS)
            throw std::runtime_error("Failed to create render pass!");
    }

    VkShaderModule CreateShaderModule(const std::vector<char> &code)
    {
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

        VkShaderModule module;
        if(vkCreateShaderModule(m_device, &createInfo, nullptr, &module) != VK_SUCCESS)
            throw std::runtime_error("Failed to create shader module!");
        return module;
    }

    void CreateDescriptorSetLayout()
    {
        VkDescriptorSetLayoutBinding uboLayoutBinding = {};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        uboLayoutBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        createInfo.bindingCount = 1;
        createInfo.pBindings = &uboLayoutBinding;
        if(vkCreateDescriptorSetLayout(m_device, &createInfo, nullptr, &m_descriptorSetLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create descriptor set layout");
    }

    void CreateGraphicsPipeline()
    {
        auto vertCode = ReadFile("shaders/vert.spv");
        auto fragCode = ReadFile("shaders/frag.spv");

        VkShaderModule vertModule = CreateShaderModule(vertCode);
        VkShaderModule fragModule = CreateShaderModule(fragCode);

        VkPipelineShaderStageCreateInfo vertStageInfo = {};
        vertStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertStageInfo.module = vertModule;
        vertStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragStageInfo = {};
        fragStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragStageInfo.module = fragModule;
        fragStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo stages[] = { vertStageInfo, fragStageInfo };

        auto bindingDesc = Vertex::GetBindingDescription();
        auto attributeDescs = Vertex::GetAttributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescs.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescs.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(m_swapChainExtent.width);
        viewport.height = static_cast<float>(m_swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        
        VkRect2D scissor = {};
        scissor.offset = { 0, 0 };
        scissor.extent = m_swapChainExtent;

        VkPipelineViewportStateCreateInfo viewportState = {};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer = {};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;

        VkPipelineMultisampleStateCreateInfo multisampling = {};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT
            | VK_COLOR_COMPONENT_G_BIT
            | VK_COLOR_COMPONENT_B_BIT
            | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlend = {};
        colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlend.logicOpEnable = VK_FALSE;
        colorBlend.logicOp = VK_LOGIC_OP_COPY;
        colorBlend.attachmentCount = 1;
        colorBlend.pAttachments = &colorBlendAttachment;
        colorBlend.blendConstants[0] = 0.0f;
        colorBlend.blendConstants[1] = 0.0f;
        colorBlend.blendConstants[2] = 0.0f;
        colorBlend.blendConstants[3] = 0.0f;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;
        if(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create pipeline layout!");
        
        VkGraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = stages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr;
        pipelineInfo.pColorBlendState = &colorBlend;
        pipelineInfo.pDynamicState = nullptr;
        pipelineInfo.layout = m_pipelineLayout;
        pipelineInfo.renderPass = m_renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.basePipelineIndex = -1;

        if(vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_graphicsPipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create graphics pipeline!");

        vkDestroyShaderModule(m_device, fragModule, nullptr);
        vkDestroyShaderModule(m_device, vertModule, nullptr);
    }

    void CreateFrameBuffers()
    {
        m_swapChainFrameBuffers.resize(m_swapChainImageViews.size());
        for(int i = 0; i < m_swapChainImageViews.size(); i++)
        {
            VkImageView attachments[] = {
                m_swapChainImageViews[i]
            };
            VkFramebufferCreateInfo frameBufferInfo = {};
            frameBufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            frameBufferInfo.renderPass = m_renderPass;
            frameBufferInfo.attachmentCount = 1;
            frameBufferInfo.pAttachments = attachments;
            frameBufferInfo.width = m_swapChainExtent.width;
            frameBufferInfo.height = m_swapChainExtent.height;
            frameBufferInfo.layers = 1;
            if(vkCreateFramebuffer(m_device, &frameBufferInfo, nullptr, &m_swapChainFrameBuffers[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to create framebuffer!");
        }
    }

    void CreateCommandPool()
    {
        QueueFamilyIndices indices = FindQueueFamilies(m_physicalDevice);

        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = indices.m_graphicsFamily.value();
        poolInfo.flags = 0;
        if(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create command pool!");
    }

    VkCommandBuffer BeginSingleTimeCommands()
    {
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = m_commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        return commandBuffer;
    }

    void EndSingleTimeCommands(VkCommandBuffer commandBuffer)
    {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_graphicsQueue);
        vkFreeCommandBuffers(m_device, m_commandPool, 1, &commandBuffer);
    }
    
    void CopyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size)
    {
        VkCommandBuffer commandBuffer = BeginSingleTimeCommands();

        VkBufferCopy copyRegion = {};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, src, dst, 1, &copyRegion);

        EndSingleTimeCommands(commandBuffer);
    }

    void SetupBarrierAccessMasks(
        VkImageMemoryBarrier &barrier,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        VkPipelineStageFlags &sourceStage,
        VkPipelineStageFlags &destinationStage)
    {
        if(oldLayout == VK_IMAGE_LAYOUT_UNDEFINED
           && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if(oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else
        {
            throw std::runtime_error("Unsupported layout transition!");
        }
    }

    void TransitionImageLayout(
        VkImage image,
        VkFormat format,
        VkImageLayout oldLayout,
        VkImageLayout newLayout)
    {
        VkCommandBuffer commandBuffer = BeginSingleTimeCommands();

        VkImageMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = 0;

        VkPipelineStageFlags srcStage, dstStage;
        SetupBarrierAccessMasks(
            barrier,
            oldLayout,
            newLayout,
            srcStage,
            dstStage);

        vkCmdPipelineBarrier(
            commandBuffer,
            srcStage,
            dstStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier);
        
        EndSingleTimeCommands(commandBuffer);
    }

    void CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
    {
        VkCommandBuffer commandBuffer = BeginSingleTimeCommands();

        VkBufferImageCopy region = {};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;

        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = { width, height, 1 };

        vkCmdCopyBufferToImage(
            commandBuffer,
            buffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region);
        
        EndSingleTimeCommands(commandBuffer);
    }

    void CreateBuffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkBuffer &buffer,
        VkDeviceMemory &memory)
    {
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if(vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
            throw std::runtime_error("Failed to create buffer!");

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(m_device, buffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = FindMemoryType(
            memRequirements.memoryTypeBits,
            properties);

        if(vkAllocateMemory(m_device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate vertex buffer memory!");

        vkBindBufferMemory(m_device, buffer, memory, 0);
    }

    void CreateImage(
        uint32_t width,
        uint32_t height,
        VkFormat format,
        VkImageTiling tiling,
        VkImageUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkImage &image,
        VkDeviceMemory &imageMemory)
    {
        VkImageCreateInfo imageInfo = {};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling,
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.flags = 0;

        if(vkCreateImage(m_device, &imageInfo, nullptr, &image) != VK_SUCCESS)
            throw std::runtime_error("Failed to create image!");

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(m_device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = FindMemoryType(
            memRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if(vkAllocateMemory(m_device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate image memory!");

        vkBindImageMemory(m_device, image, imageMemory, 0);
    }

    VkImageView CreateImageView(VkImage image, VkFormat format)
    {
        VkImageViewCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = image;
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = format;

        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        VkImageView result;
        if(vkCreateImageView(m_device, &createInfo, nullptr, &result) != VK_SUCCESS)
            throw std::runtime_error("Failed to create texture image view!");

        return result;
    }

    void CreateTextureImage()
    {
        int texWidth, texHeight, texChannels;
        stbi_uc *pixels = stbi_load(
            "textures/texture.jpg",
            &texWidth,
            &texHeight,
            &texChannels,
            STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4;

        if(!pixels)
            throw std::runtime_error("Failed to load texture image!");

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        CreateBuffer(
            imageSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer,
            stagingBufferMemory);

        void *data;
        vkMapMemory(m_device, stagingBufferMemory, 0, imageSize, 0, &data);
        std::memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(m_device, stagingBufferMemory);
        stbi_image_free(pixels);

        CreateImage(
            texWidth,
            texHeight,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            m_textureImage,
            m_textureImageMemory);

        TransitionImageLayout(
            m_textureImage,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        CopyBufferToImage(stagingBuffer, m_textureImage, texWidth, texHeight);
        
        TransitionImageLayout(
            m_textureImage,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        vkFreeMemory(m_device, stagingBufferMemory, nullptr);
    }

    void CreateTextureImageView()
    {
        m_textureImageView = CreateImageView(m_textureImage, VK_FORMAT_R8G8B8A8_SRGB);
    }

    void CreateTextureSampler()
    {
        VkSamplerCreateInfo samplerInfo = {};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = 16;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;

        if(vkCreateSampler(m_device, &samplerInfo, nullptr, &m_textureSampler) != VK_SUCCESS)
            throw std::runtime_error("Failed to create texture sampler!");
    }

    void CreateVertexBuffer()
    {
        VkDeviceSize size = sizeof(vertices[0]) * vertices.size();
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        CreateBuffer(
            size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer,
            stagingBufferMemory);

        void *data;
        vkMapMemory(m_device, stagingBufferMemory, 0, size, 0, &data);
        std::memcpy(data, vertices.data(), static_cast<size_t>(size));
        vkUnmapMemory(m_device, stagingBufferMemory);

        CreateBuffer(
            size,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            m_vertexBuffer,
            m_vertexBufferMemory);
        CopyBuffer(stagingBuffer, m_vertexBuffer, size);

        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        vkFreeMemory(m_device, stagingBufferMemory, nullptr);
    }

    void CreateIndexBuffer()
    {
        VkDeviceSize size = sizeof(indices[0]) * indices.size();
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        CreateBuffer(
            size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer,
            stagingBufferMemory);

        void *data;
        vkMapMemory(m_device, stagingBufferMemory, 0, size, 0, &data);
        std::memcpy(data, indices.data(), static_cast<size_t>(size));
        vkUnmapMemory(m_device, stagingBufferMemory);

        CreateBuffer(
            size,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            m_indexBuffer,
            m_indexBufferMemory);
        CopyBuffer(stagingBuffer, m_indexBuffer, size);

        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        vkFreeMemory(m_device, stagingBufferMemory, nullptr);
    }

    void CreateUniformBuffers()
    {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        int numUniformBuffers = m_swapChainImages.size();
        m_uniformBuffers.resize(numUniformBuffers);
        m_uniformBuffersMemory.resize(numUniformBuffers);
        
        for(int i = 0; i < numUniformBuffers; i++)
        {
            CreateBuffer(
                bufferSize,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                m_uniformBuffers[i],
                m_uniformBuffersMemory[i]);
        }
    }

    void CreateDescriptorPool()
    {
        VkDescriptorPoolSize poolSize = {};
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(m_swapChainImages.size());
        
        VkDescriptorPoolCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        createInfo.poolSizeCount = 1;
        createInfo.pPoolSizes = &poolSize;
        createInfo.maxSets = static_cast<uint32_t>(m_swapChainImages.size());

        if(vkCreateDescriptorPool(m_device, &createInfo, nullptr, &m_descriptorPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create descriptor pool!");
    }

    void CreateDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> layouts(m_swapChainImages.size(), m_descriptorSetLayout);

        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(m_swapChainImages.size());
        allocInfo.pSetLayouts = layouts.data();

        m_descriptorSets.resize(m_swapChainImages.size());
        if(vkAllocateDescriptorSets(m_device, &allocInfo, m_descriptorSets.data()) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate descriptor set!");

        for(int i = 0; i < m_swapChainImages.size(); i++)
        {
            VkDescriptorBufferInfo bufferInfo = {};
            bufferInfo.buffer = m_uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkWriteDescriptorSet descriptorWrite = {};
            descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite.dstSet = m_descriptorSets[i];
            descriptorWrite.dstBinding = 0;
            descriptorWrite.dstArrayElement = 0;
            descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrite.descriptorCount = 1;
            descriptorWrite.pBufferInfo = &bufferInfo;
            descriptorWrite.pImageInfo = nullptr;
            descriptorWrite.pTexelBufferView = nullptr;
            vkUpdateDescriptorSets(m_device, 1, &descriptorWrite, 0, nullptr);
        }
    }

    void CreateCommandBuffers()
    {
        m_commandBuffers.resize(m_swapChainFrameBuffers.size());
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = m_commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(m_commandBuffers.size());
        if(vkAllocateCommandBuffers(m_device, &allocInfo, m_commandBuffers.data()) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate command buffers!");

        for(int i = 0; i < m_commandBuffers.size(); i++)
        {
            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = 0;
            beginInfo.pInheritanceInfo = nullptr;

            if(vkBeginCommandBuffer(m_commandBuffers[i], &beginInfo) != VK_SUCCESS)
                throw std::runtime_error("Failed to begin recording command buffer!");


            VkClearValue clearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
            VkRenderPassBeginInfo renderPassInfo = {};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass = m_renderPass;
            renderPassInfo.framebuffer = m_swapChainFrameBuffers[i];
            renderPassInfo.renderArea.offset = { 0, 0 };
            renderPassInfo.renderArea.extent = m_swapChainExtent;
            renderPassInfo.clearValueCount = 1;
            renderPassInfo.pClearValues = &clearColor;

            vkCmdBeginRenderPass(m_commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(m_commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
            VkBuffer vertexBuffers[] = { m_vertexBuffer };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(m_commandBuffers[i], 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(m_commandBuffers[i], m_indexBuffer, 0, VK_INDEX_TYPE_UINT16);
            vkCmdBindDescriptorSets(
                m_commandBuffers[i],
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                m_pipelineLayout,
                0,
                1,
                &m_descriptorSets[i],
                0,
                nullptr);

            vkCmdDrawIndexed(m_commandBuffers[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

            vkCmdEndRenderPass(m_commandBuffers[i]);
            
            if(vkEndCommandBuffer(m_commandBuffers[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to record command buffer!");
        }
    }

    void CreateSyncObjects()
    {
        m_imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        m_renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        m_inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        m_imagesInFlight.resize(m_swapChainImages.size(), VK_NULL_HANDLE);
        VkSemaphoreCreateInfo semaphoreInfo = {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        
        for(int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            if(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_imageAvailableSemaphores[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to create ImageAvailable Semaphore!");
            if(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_renderFinishedSemaphores[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to create ImageAvailable Semaphore!");
            if(vkCreateFence(m_device, &fenceInfo, nullptr, &m_inFlightFences[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to create fence!");
        }
    }

    void CleanupSwapChain()
    {
        for(int i = 0; i < m_swapChainFrameBuffers.size(); i++)
            vkDestroyFramebuffer(m_device, m_swapChainFrameBuffers[i], nullptr);

        vkFreeCommandBuffers(m_device, m_commandPool, static_cast<uint32_t>(m_commandBuffers.size()), m_commandBuffers.data());
        vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        vkDestroyRenderPass(m_device, m_renderPass, nullptr);
        for(int i = 0; i < m_swapChainImageViews.size(); i++)
            vkDestroyImageView(m_device, m_swapChainImageViews[i], nullptr);
        vkDestroySwapchainKHR(m_device, m_swapChain, nullptr);

        for(int i = 0; i < m_uniformBuffers.size(); i++)
        {
            vkDestroyBuffer(m_device, m_uniformBuffers[i], nullptr);
            vkFreeMemory(m_device, m_uniformBuffersMemory[i], nullptr);
        }

        vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    }

    void RecreateSwapChain()
    {
        int width, height;
        glfwGetFramebufferSize(m_window, &width, &height);
        while(width == 0 && height == 0)
        {
            glfwGetFramebufferSize(m_window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(m_device);

        CleanupSwapChain();

        CreateSwapChain();
        CreateImageViews();
        CreateRenderPass();
        CreateGraphicsPipeline();
        CreateFrameBuffers();
        CreateUniformBuffers();
        CreateDescriptorPool();
        CreateDescriptorSets();
        CreateCommandBuffers();
    }

    void InitVulkan()
    {
        CreateInstance();
        SetupDebugMessenger();
        CreateSurface();
        PickPhysicalDevice();
        CreateLogicalDevice();
        CreateSwapChain();
        CreateImageViews();
        CreateRenderPass();
        CreateDescriptorSetLayout();
        CreateGraphicsPipeline();
        CreateFrameBuffers();
        CreateCommandPool();
        CreateTextureImage();
        CreateTextureImageView();
        CreateTextureSampler();
        CreateVertexBuffer();
        CreateIndexBuffer();
        CreateUniformBuffers();
        CreateDescriptorPool();
        CreateDescriptorSets();
        CreateCommandBuffers();
        CreateSyncObjects();
    }

    std::vector<const char *> GetRequiredExtensions()
    {
        uint32_t nGlfwExtensions;
        const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&nGlfwExtensions);
        std::vector<const char *> extensions(glfwExtensions, glfwExtensions + nGlfwExtensions);
        if(ENABLE_VALIDATION_LAYERS)
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        return extensions;
    }

    void CreateInstance()
    {
        if(ENABLE_VALIDATION_LAYERS && !CheckValidationLayerSupport())
            throw std::runtime_error("Validation layers requested but not available!");
        
        VkApplicationInfo appInfo = {}; // Optional, used for optimizations
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo = {}; // Not optional
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
        if(ENABLE_VALIDATION_LAYERS)
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(m_validationLayers.size());
            createInfo.ppEnabledLayerNames = m_validationLayers.data();
            PopulateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = reinterpret_cast<VkDebugUtilsMessengerCreateInfoEXT *>(&debugCreateInfo);
        }
        else
        {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }

        auto extensions = GetRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();
        if(vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan instance!");

        uint32_t nAvailable;
        vkEnumerateInstanceExtensionProperties(nullptr, &nAvailable, nullptr);
        std::vector<VkExtensionProperties> available(nAvailable);
        vkEnumerateInstanceExtensionProperties(nullptr, &nAvailable, available.data());
        std::cout << "Available extensions:\n";
        std::set<std::string> availableSet;
        for(const auto &extension : available)
        {
            availableSet.insert(extension.extensionName);
            std::cout << '\t' << extension.extensionName << '\n';
        }

        bool allSupported = true;
        for(const auto &extension : extensions)
            if(availableSet.find(extension) == availableSet.end())
                std::cout << "Missing support for extension " << extension << '\n';
    }

    bool CheckValidationLayerSupport()
    {
        uint32_t nLayers;
        vkEnumerateInstanceLayerProperties(&nLayers, nullptr);

        std::vector<VkLayerProperties> layers(nLayers);
        vkEnumerateInstanceLayerProperties(&nLayers, layers.data());
        std::set<std::string> layerSet;
        std::cout << "Available layers:" << std::endl;
        for(const auto &layer : layers)
        {
            layerSet.insert(layer.layerName);
            std::cout << '\t' << layer.layerName << '\n';
        }
        std::cout << '\n';
        for(const char *layer : m_validationLayers)
        {
            if(layerSet.find(layer) == layerSet.end())
                return false;
        }

        return true;
    }

    void UpdateUniformBuffer(uint32_t imageIndex)
    {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        auto diff = currentTime - startTime;
        float time = std::chrono::duration<float, std::chrono::seconds::period>(diff).count();

        UniformBufferObject ubo = {};
        ubo.model = glm::rotate(
            glm::mat4(1.0f),
            time * glm::radians(90.0f),
            glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(
            glm::vec3(2.0f, 2.0f, 2.0f),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(
            glm::radians(45.0f),
            m_swapChainExtent.width / static_cast<float>(m_swapChainExtent.height),
            0.1f,
            10.0f);
        ubo.proj[1][1] *= -1;

        void *data;
        vkMapMemory(m_device, m_uniformBuffersMemory[imageIndex], 0, sizeof(ubo), 0, &data);
        std::memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(m_device, m_uniformBuffersMemory[imageIndex]);
    }

    void DrawFrame()
    {
        vkWaitForFences(m_device, 1, &m_inFlightFences[m_currentFrame], VK_TRUE, UINT64_MAX);
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(
            m_device,
            m_swapChain,
            UINT64_MAX,
            m_imageAvailableSemaphores[m_currentFrame],
            VK_NULL_HANDLE,
            &imageIndex);
        
        if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
            return RecreateSwapChain();
        else if(result != VK_SUCCESS)
            throw std::runtime_error("Failed to acquire swap chain image!");

        if(m_imagesInFlight[imageIndex] != VK_NULL_HANDLE)
            vkWaitForFences(m_device, 1, &m_imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        m_imagesInFlight[imageIndex] = m_inFlightFences[m_currentFrame];

        UpdateUniformBuffer(imageIndex);

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = { m_imageAvailableSemaphores[m_currentFrame] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_commandBuffers[imageIndex];

        VkSemaphore signalSemaphores[] = { m_renderFinishedSemaphores[m_currentFrame] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        vkResetFences(m_device, 1, &m_inFlightFences[m_currentFrame]);

        if(vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, m_inFlightFences[m_currentFrame]) != VK_SUCCESS)
            throw std::runtime_error("Failed to submit draw command buffer!");

        VkPresentInfoKHR presentInfo = {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = { m_swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr;

        result = vkQueuePresentKHR(m_presentQueue, &presentInfo);
        if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || m_frameBufferResized)
        {
            m_frameBufferResized = false;
            return RecreateSwapChain();
        }
        else if(result != VK_SUCCESS)
            throw std::runtime_error("Failed to present swap chain image!");

        m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
    
    void MainLoop()
    {
        while(!glfwWindowShouldClose(m_window))
        {
            glfwPollEvents();
            DrawFrame();
        }
        vkDeviceWaitIdle(m_device);
    }

    void Cleanup()
    {
        CleanupSwapChain();
        vkDestroySampler(m_device, m_textureSampler, nullptr);
        vkDestroyImageView(m_device, m_textureImageView, nullptr);
        vkDestroyImage(m_device, m_textureImage, nullptr);
        vkFreeMemory(m_device, m_textureImageMemory, nullptr);
        vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
        for(int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vkDestroySemaphore(m_device, m_renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(m_device, m_imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(m_device, m_inFlightFences[i], nullptr);
        }
        vkDestroyBuffer(m_device, m_vertexBuffer, nullptr);
        vkFreeMemory(m_device, m_vertexBufferMemory, nullptr);
        vkDestroyBuffer(m_device, m_indexBuffer, nullptr);
        vkFreeMemory(m_device, m_indexBufferMemory, nullptr);
        vkDestroyCommandPool(m_device, m_commandPool, nullptr);
        vkDestroyDevice(m_device, nullptr);
        if(ENABLE_VALIDATION_LAYERS)
            DestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        vkDestroyInstance(m_instance, nullptr);
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT type,
        const VkDebugUtilsMessengerCallbackDataEXT *callbackData,
        void *userData)
    {
        if((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) == 0)
            return VK_FALSE;

        switch(severity)
        {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
            std::cerr << "VERBOSE: ";
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            std::cerr << "WARNING: ";
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            std::cerr << "ERROR: ";
            break;
        default:
            std::cerr << "UNKNOWN: ";
        }
        std::cerr << "Validation layer: " << callbackData->pMessage << std::endl;
        return VK_FALSE;
    }

    static void FramebufferResizeCallback(GLFWwindow *window, int width, int height)
    {
        auto app = reinterpret_cast<TriangleApp *>(glfwGetWindowUserPointer(window));
        app->m_frameBufferResized = true;
    }

    static constexpr uint32_t WIDTH = 800;
    static constexpr uint32_t HEIGHT = 600;
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

    const std::vector<const char *> m_validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };

    const std::vector<const char *> m_deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

#ifndef NDEBUG
    static constexpr bool ENABLE_VALIDATION_LAYERS = true;
#else
    static constexpr bool ENABLE_VALIDATION_LAYERS = false;
#endif

    VkImage m_textureImage;
    VkDeviceMemory m_textureImageMemory;
    VkImageView m_textureImageView;
    VkSampler m_textureSampler;
    GLFWwindow *m_window;
    VkInstance m_instance;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device;
    VkSurfaceKHR m_surface;
    VkQueue m_graphicsQueue;
    VkQueue m_presentQueue;
    VkSwapchainKHR m_swapChain;
    std::vector<VkImage> m_swapChainImages;
    VkFormat m_swapChainImageFormat;
    VkExtent2D m_swapChainExtent;
    std::vector<VkImageView> m_swapChainImageViews;
    VkRenderPass m_renderPass;
    VkDescriptorSetLayout m_descriptorSetLayout;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_graphicsPipeline;
    std::vector<VkFramebuffer> m_swapChainFrameBuffers;
    VkBuffer m_vertexBuffer;
    VkDeviceMemory m_vertexBufferMemory;
    VkBuffer m_indexBuffer;
    VkDeviceMemory m_indexBufferMemory;
    VkDescriptorPool m_descriptorPool;
    std::vector<VkDescriptorSet> m_descriptorSets;
    std::vector<VkBuffer> m_uniformBuffers;
    std::vector<VkDeviceMemory> m_uniformBuffersMemory;
    VkCommandPool m_commandPool;
    std::vector<VkCommandBuffer> m_commandBuffers;
    std::vector<VkSemaphore> m_imageAvailableSemaphores;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    std::vector<VkFence> m_inFlightFences;
    std::vector<VkFence> m_imagesInFlight;
    int m_currentFrame = 0;
    bool m_frameBufferResized = false;
    VkDebugUtilsMessengerEXT m_debugMessenger;
};

int main()
{
    TriangleApp app;
    try
    {
        app.Run();
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
