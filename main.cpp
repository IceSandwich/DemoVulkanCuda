#include <iostream>
#include <algorithm>
#include <numeric>
#include <array>
#include <set>
#include <bitset>
#include <string_view>
#include <memory>
#include <functional>
#include <vulkan/vulkan_raii.hpp>

#include <Windows.h>
#include <vulkan/vulkan_win32.h>
#include "kernel.h"

// Ref: https://www.gpultra.com/blog/vulkan-cuda-memory-interoperability/
// Ref: https://developer.nvidia.com/getting-vulkan-ready-vr
// Ref: https://github.com/heterodb/toybox/blob/master/cuda_external_resources/extres_sample_01.cu
// Ref: https://github.com/KhronosGroup/VK-GL-CTS/blob/main/external/vulkancts/modules/vulkan/api/vktApiExternalMemoryTests.cpp

template <typename T>
void printArray(std::string_view name, T iterStart, T iterEnd) {
	auto printItem = [](float x) {
		std::cout << x << ", ";
	};

	std::cout << name << " array: " << std::endl;
	std::for_each(iterStart, iterStart + 10, printItem);
	std::cout << "..., ";
	std::for_each(iterEnd - 10, iterEnd - 1, printItem);
	std::cout << *(iterEnd-1);
	std::cout << std::endl << std::endl;
}

struct Data {
	static constexpr size_t N = 256;
	std::array<float, N> a;
	std::array<float, N> b;
	std::array<float, N> c;
};

Data prepareData() {
	Data ret{};

	std::iota(ret.a.begin(), ret.a.end(), 0);
	std::iota(ret.b.begin(), ret.b.end(), 2);

	printArray("a", ret.a.begin(), ret.a.end());
	printArray("b", ret.b.begin(), ret.b.end());

	return ret;
}

int main_cuda() {
	std::cout << "hello world!" << std::endl;

	Data data = prepareData();

	vectorAdd(data.a.data(), data.b.data(), data.c.data());

	printArray("c", data.c.begin(), data.c.end());

	return 0;
}

constexpr static std::array<const char *, 1> validationLayers{
	"VK_LAYER_KHRONOS_validation"
};


static constexpr std::array<const char *, 2> instanceExtensions{
	VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
	VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
};

static constexpr std::array<const char *, 5> deviceExtensions{
	VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
	VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,

	VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
	VK_KHR_VIDEO_QUEUE_EXTENSION_NAME,
	VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME,
};

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData) {
	if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
		// Message is important enough to show
		std::cout << "-- validation layer: " << pCallbackData->pMessage << std::endl;
	}
	return VK_FALSE; // return true to abort
}


uint32_t findQueue(std::string_view queueName, vk::raii::PhysicalDevice& physicalDevice, vk::QueueFlagBits requiredFlag) {
	std::vector<vk::QueueFamilyProperties> queueProps = physicalDevice.getQueueFamilyProperties();
	auto queuePropIter = std::find_if(queueProps.begin(), queueProps.end(), [requiredFlag](const vk::QueueFamilyProperties &prop) {
		return (prop.queueFlags & static_cast<vk::QueueFlags>(requiredFlag));
		});
	if (queuePropIter == queueProps.end()) {
		std::cerr << "Can't find " << queueName << "." << std::endl;
		exit(-1);
	}
	return std::distance(queueProps.begin(), queuePropIter);
}

struct Buffer {
	vk::raii::Buffer buffer;
	vk::raii::DeviceMemory deviceMemory;
	vk::DeviceSize deviceSize;

	static uint32_t findMemoryType(const vk::PhysicalDeviceMemoryProperties &memoryProperties, uint32_t typeBits, vk::MemoryPropertyFlags requirementsMask) {
		std::cout << "debug memory type for typeBits=" << std::bitset<sizeof(typeBits)>{typeBits} << " and requirementsMask=" << vk::to_string(requirementsMask) << std::endl;
		for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++, typeBits >>= 1) {
			if ((typeBits & 1) && ((memoryProperties.memoryTypes[i].propertyFlags & requirementsMask) == requirementsMask)) {
				return i;
			}
		}
		std::cout << "Can't find suitable memory type for typeBits=" << std::bitset<sizeof(typeBits)>{typeBits} << " and requirementsMask=" << vk::to_string(requirementsMask) << std::endl;
	}


	Buffer(const vk::raii::PhysicalDevice &physicalDevice, const vk::raii::Device &device, const vk::DeviceSize deviceSize, const vk::BufferUsageFlags usage, const vk::MemoryPropertyFlags properties) : buffer{ nullptr }, deviceMemory{ nullptr }, deviceSize(deviceSize) {
		//vk::PhysicalDeviceExternalBufferInfo physicalExternalInfo{ {}, usage, vk::FlagTraits<vk::ExternalMemoryHandleTypeFlagBits>::allFlags };
		//vk::ExternalBufferProperties externalProp = physicalDevice.getExternalBufferProperties(physicalExternalInfo);
		//vk::ExternalMemoryHandleTypeFlags compatiable = externalProp.externalMemoryProperties.compatibleHandleTypes;
		
	
		vk::ExportMemoryAllocateInfoKHR exportAllocInfo{ vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32 };

		vk::ExternalMemoryBufferCreateInfo externalCreateInfo{ vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32, nullptr };
		vk::BufferCreateInfo bufferCreateInfo{ {}, deviceSize, usage };
		bufferCreateInfo.pNext = &externalCreateInfo;
		buffer = vk::raii::Buffer{ device, bufferCreateInfo };
		vk::MemoryRequirements memRequirement = buffer.getMemoryRequirements();
		vk::MemoryAllocateInfo allocInfo{ memRequirement.size, findMemoryType(physicalDevice.getMemoryProperties(), memRequirement.memoryTypeBits, properties), &exportAllocInfo };
		deviceMemory = vk::raii::DeviceMemory{ device, allocInfo };

		buffer.bindMemory(*deviceMemory, 0);
	}

	Buffer &UploadData(const void *data) {
		void *mapped = deviceMemory.mapMemory(0, deviceSize);
		memcpy(mapped, data, deviceSize);
		deviceMemory.unmapMemory();

		return *this;
	}

	template <typename T>
	std::unique_ptr<T[], std::function<void(T *)> > MapFromDevice() {
		void *mapped = deviceMemory.mapMemory(0, deviceSize);
		return std::unique_ptr<T[], std::function<void(T *)> >(
			(T *)mapped,
			[mem = *deviceMemory, device = buffer.getDevice()](T *) {
				device.unmapMemory(mem);
			}
		);
	}
};

struct CommandBuffer {
	vk::raii::CommandBuffer instance;
	CommandBuffer(std::nullptr_t) : instance{nullptr} {

	}
	CommandBuffer(const vk::raii::Device &device, const vk::raii::CommandPool &commandPool) : instance{ nullptr } {
		vk::CommandBufferAllocateInfo commandBufferAllocateInfo{ *commandPool, vk::CommandBufferLevel::ePrimary, 1 };
		instance = std::move(device.allocateCommandBuffers(commandBufferAllocateInfo).front());
	}
	CommandBuffer &Record(std::function<void(const vk::raii::CommandBuffer &)> func, const vk::CommandBufferUsageFlags usage = vk::CommandBufferUsageFlagBits::eOneTimeSubmit) {
		vk::CommandBufferBeginInfo commandBufferBeginInfo{ usage };
		instance.begin(commandBufferBeginInfo);
		func(instance);
		instance.end();
		return *this;
	}
	void SubmitAndWait(const vk::raii::Queue &queue) {
		vk::SubmitInfo submitInfo{ nullptr, nullptr, *instance };
		queue.submit(submitInfo, nullptr);
		queue.waitIdle();
	}
	void Submit(const vk::raii::Queue &queue, const std::vector<vk::Semaphore> &waitSemaphores, const std::vector<vk::PipelineStageFlags> waitDstStageMask, const std::vector<vk::Semaphore> &signalSemaphore, const vk::raii::Fence &fence) {
		//static std::function<vk::Semaphore(const vk::raii::Semaphore&)> convertToVkSemaphore{ [](const vk::raii::Semaphore& semaphore) {
		//	return *semaphore;
		//}};
		//std::vector<vk::Semaphore> waitVkSemaphores{ waitSemaphores.size() }, signalVkSemaphores{ signalSemaphore.size() };
		//std::transform(waitSemaphores.begin(), waitSemaphores.end(), waitVkSemaphores.begin(), convertToVkSemaphore);
		//std::transform(signalSemaphore.begin(), signalSemaphore.end(), signalVkSemaphores.begin(), convertToVkSemaphore);
		vk::SubmitInfo submitInfo{ static_cast<uint32_t>(waitSemaphores.size()), waitSemaphores.data(), waitDstStageMask.size() == 0 ? nullptr : waitDstStageMask.data(), 1, &*instance, static_cast<uint32_t>(signalSemaphore.size()), signalSemaphore.data() };;
		queue.submit({ submitInfo }, *fence);
	}
	void Clear() {
		instance.reset();
	}
};

struct StagingBuffer {
	Buffer buffer;
	vk::Device device;
	vk::DeviceSize deviceSize;
	bool hasSubmitted;
	StagingBuffer(const vk::raii::PhysicalDevice &physicalDevice, const vk::raii::Device &device, const vk::DeviceSize deviceSize) :
		buffer{ physicalDevice, device, deviceSize, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible },
		device{ *device },
		deviceSize{ deviceSize },
		hasSubmitted{ false } {

	}

	~StagingBuffer() {
		if (!hasSubmitted) {
			std::cout << "[Warning] Staging buffer haven't been submitted. Is it intentional? Use StagingBuffer::NoSubmit() to disable this warning.";
		}
	}

	StagingBuffer &TranferToDevice(const void *data) {
		buffer.UploadData(data);

		return *this;
	}

	StagingBuffer &Submit(const vk::raii::CommandPool &commandPool, const vk::raii::Queue &queue, const vk::raii::Buffer &dst) {
		vk::CommandBufferAllocateInfo commandBufferAllocateInfo{ *commandPool, vk::CommandBufferLevel::ePrimary, 1 };
		vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(commandBufferAllocateInfo).front();

		vk::CommandBufferBeginInfo commandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit };
		commandBuffer.begin(commandBufferBeginInfo);

		std::array<vk::BufferCopy, 1> regions = {
			vk::BufferCopy{0, 0, deviceSize}
		};

		commandBuffer.copyBuffer(*buffer.buffer, *dst, regions);

		commandBuffer.end();

		vk::SubmitInfo submitInfo{ nullptr, nullptr, commandBuffer };
		queue.submit(submitInfo, nullptr);
		queue.waitIdle();

		device.freeCommandBuffers(*commandPool, 1, &commandBuffer);

		hasSubmitted = true;

		return *this;
	}

	StagingBuffer &TransferFromBuffer(const vk::raii::CommandPool &commandPool, const vk::raii::Queue &queue, const vk::raii::Buffer &src) {
		vk::CommandBufferAllocateInfo commandBufferAllocateInfo{ *commandPool, vk::CommandBufferLevel::ePrimary, 1 };
		vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(commandBufferAllocateInfo).front();

		vk::CommandBufferBeginInfo commandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit };
		commandBuffer.begin(commandBufferBeginInfo);

		std::array<vk::BufferCopy, 1> regions = {
			vk::BufferCopy{0, 0, deviceSize}
		};

		commandBuffer.copyBuffer(*src, *buffer.buffer, regions);

		commandBuffer.end();

		vk::SubmitInfo submitInfo{ nullptr, nullptr, commandBuffer };
		queue.submit(submitInfo, nullptr);
		queue.waitIdle();

		device.freeCommandBuffers(*commandPool, 1, &commandBuffer);
		return *this;
	}

	template <typename T>
	StagingBuffer &CopyToHost(T* output) {
		auto mapping = buffer.MapFromDevice<T>();
		std::memcpy(output, mapping.get(), deviceSize);
		
		return *this;
	}

	StagingBuffer &NoSubmit() {
		hasSubmitted = true;

		return *this;
	}
};

struct VkContext {
	vk::raii::Context context;
	vk::raii::Instance instance;
	vk::raii::PhysicalDevice physicalDevice;
	vk::raii::Device device;
	vk::raii::CommandPool commandPool;
	CommandBuffer commandBuffer;

	vk::raii::Queue videoDecodeQueue;
	vk::raii::Queue computeQueue;
	vk::raii::Queue transferQueue;

	uint32_t videoDecodeQueueFamilyIndex;
	uint32_t computeQueueFamilyIndex;
	uint32_t transferQueueFamilyIndex;

	VkContext() : instance{ nullptr }, physicalDevice{ nullptr }, device{ nullptr }, commandPool{ nullptr }, commandBuffer{ nullptr }, 
		videoDecodeQueue{ nullptr }, computeQueue{ nullptr }, transferQueue{ nullptr } {

	}
};


HANDLE getMemoryHandle(vk::Instance instance, vk::Device device, vk::DeviceMemory deviceMemory) {
	PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR = (PFN_vkGetMemoryWin32HandleKHR)instance.getProcAddr("vkGetMemoryWin32HandleKHR");
	if (vkGetMemoryWin32HandleKHR == nullptr) {
		throw std::exception{ "cannot find vk getmemorywin32handlekhr" };
	}

	VkMemoryGetWin32HandleInfoKHR info{ VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR, nullptr };
	info.memory = deviceMemory;
	info.handleType = VkExternalMemoryHandleTypeFlagBits::VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

	HANDLE handle;
	if (VkResult result = vkGetMemoryWin32HandleKHR(device, &info, &handle); result != VK_SUCCESS) {
		std::cerr << result << std::endl;
		throw std::exception{ "fail to call vk getmemorywin32handlekhr" };
	}

	return handle;
}

VkContext prepareVulkanContext(const char *appName) {
	VkContext context;

	vk::ApplicationInfo appInfo{ appName, VK_MAKE_VERSION(1, 0, 0), "No engine", VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_1, nullptr };;
	vk::InstanceCreateInfo instanceCreateInfo{ {}, &appInfo, validationLayers.size(), validationLayers.data(), instanceExtensions.size(), instanceExtensions.data(), nullptr };

	context.instance = vk::raii::Instance{ context.context, instanceCreateInfo };

	vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
	vk::DebugUtilsMessageTypeFlagsEXT    messageTypeFlags(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
	vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT({}, severityFlags, messageTypeFlags, &debugCallback);
	vk::raii::DebugUtilsMessengerEXT debugMessenger{ context.instance, debugUtilsMessengerCreateInfoEXT };

	context.physicalDevice = vk::raii::PhysicalDevices{ context.instance }.front();
	std::cout << "Using device: " << context.physicalDevice.getProperties().deviceName << std::endl;

	float queuePriority = 0;
	context.videoDecodeQueueFamilyIndex = findQueue("video decode", context.physicalDevice, vk::QueueFlagBits::eVideoDecodeKHR);
	context.computeQueueFamilyIndex = findQueue("compute queue", context.physicalDevice, vk::QueueFlagBits::eCompute);
	context.transferQueueFamilyIndex = findQueue("transfer queue", context.physicalDevice, vk::QueueFlagBits::eTransfer);

	std::vector<vk::DeviceQueueCreateInfo> queueCreateInfo;
	std::set<uint32_t> uniqueQueue{ context.videoDecodeQueueFamilyIndex, context.computeQueueFamilyIndex, context.transferQueueFamilyIndex };
	for (const uint32_t queueIdx : uniqueQueue) {
		queueCreateInfo.push_back(vk::DeviceQueueCreateInfo{ {}, queueIdx, 1, &queuePriority });
	}

	vk::PhysicalDeviceFeatures deviceFeatures;
	vk::DeviceCreateInfo deviceCreateInfo{ {}, static_cast<uint32_t>(queueCreateInfo.size()), queueCreateInfo.data(), static_cast<uint32_t>(validationLayers.size()), validationLayers.data(), deviceExtensions.size(), deviceExtensions.data(), &deviceFeatures };
	context.device = vk::raii::Device{ context.physicalDevice, deviceCreateInfo };

	vk::PhysicalDeviceMemoryProperties memProperties = context.physicalDevice.getMemoryProperties();
	context.videoDecodeQueue = vk::raii::Queue{ context.device, context.videoDecodeQueueFamilyIndex, 0 };
	context.computeQueue = vk::raii::Queue{ context.device, context.computeQueueFamilyIndex, 0 };
	context.transferQueue = vk::raii::Queue{ context.device, context.transferQueueFamilyIndex, 0 };

	vk::CommandPoolCreateInfo commandPoolCreateInfo{ vk::CommandPoolCreateFlagBits::eResetCommandBuffer, context.transferQueueFamilyIndex };
	context.commandPool = vk::raii::CommandPool{ context.device, commandPoolCreateInfo };
	context.commandBuffer = CommandBuffer{ context.device, context.commandPool };

	return context;
}

int main_vkcuda() {
	VkContext context = prepareVulkanContext("TestVulkanCuda");
	vk::raii::PhysicalDevice &physicalDevice = context.physicalDevice;
	vk::raii::Device &device = context.device;
	vk::raii::Queue &transferQueue = context.transferQueue;
	vk::raii::CommandPool &commandPool = context.commandPool;
	vk::raii::Instance &instance = context.instance;

	Data data = prepareData();
	vk::DeviceSize size{ sizeof(data.a[0]) * data.N };

	Buffer bufferA{ physicalDevice, device, size, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal };
	Buffer bufferB{ physicalDevice, device, size, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal };
	Buffer bufferC{ physicalDevice, device, size, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eDeviceLocal };

	StagingBuffer stagingBuffer{ physicalDevice, device, size };
	stagingBuffer.TranferToDevice(data.a.data()).Submit(commandPool, transferQueue, bufferA.buffer);
	stagingBuffer.TranferToDevice(data.b.data()).Submit(commandPool, transferQueue, bufferB.buffer);

	HANDLE handleA = getMemoryHandle(*instance, *device, *bufferA.deviceMemory);
	HANDLE handleB = getMemoryHandle(*instance, *device, *bufferB.deviceMemory);
	HANDLE handleC = getMemoryHandle(*instance, *device, *bufferC.deviceMemory);

	std::cout << "HandleA: " << handleA << ", HandleB: " << handleB << ", HandleC: " << handleC << std::endl;

	vectorAdd(handleA, handleB, handleC);

	stagingBuffer.TransferFromBuffer(commandPool, transferQueue, bufferC.buffer).CopyToHost<float>(data.c.data());

	printArray("c", data.c.begin(), data.c.end());

	return 0;
}

#include <cuda_runtime.h>
void printCudaInfos() {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	for (int idx = 0; idx < deviceCount; ++idx) {
		cudaDeviceProp deviceProp;
		if (cudaGetDeviceProperties(&deviceProp, idx) == cudaError::cudaSuccess) {
			std::cout << "==============================================================================" << std::endl;
			std::cout << "Device[" << idx << "]: " << deviceProp.name << std::endl;
			std::cout << "Number of multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
			std::cout << "Total amount of global memory: " << deviceProp.totalGlobalMem << std::endl;
			std::cout << "Total registers per block: " << deviceProp.regsPerBlock << std::endl;
			std::cout << "Warp size: " << deviceProp.warpSize << std::endl;
			std::cout << "Maximum number of thread per block: " << deviceProp.maxThreadsPerBlock << std::endl;
			std::cout << "Maximum size of each dimension of a grid: " << deviceProp.maxGridSize[0] << " x " << deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2] << std::endl;
			std::cout << "Is support zero copy: " << std::boolalpha << (deviceProp.canMapHostMemory == 1) << std::noboolalpha << std::endl;
			std::cout << "==============================================================================" << std::endl;
		}
	}
}

int main() {
	printCudaInfos();

	main_vkcuda();

	//runtest();


	//Ref: https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_KHR_video_queue.adoc#create-video-session-for-a-video-profile
	// 
	//VkContext context = prepareVulkanContext("TestVulkanCuda");

	//vk::VideoProfileInfoKHR profile;
	

	//vk::VideoSessionCreateInfoKHR videoSessionCreateInfo{ context.videoDecodeQueueFamilyIndex, {}, &profile, vk::Format::eB8G8R8A8Sint, vk::Extent2D{640, 480}, vk::Format::;


	return 0;
}
