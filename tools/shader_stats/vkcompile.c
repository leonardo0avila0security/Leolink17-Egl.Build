/* Compile one SPIR-V compute module through the Vulkan driver, so that
 * RADV_DEBUG=shaderstats has something to print. Nothing is dispatched: the
 * pipeline is created and thrown away. Used to read a shader's VGPR/SGPR/
 * scratch/occupancy numbers offline, without booting a game.
 *
 *   vkcompile <shader.spv> [image-binding] [push-constant-bytes] [spec-kind] [spec-refine]
 *
 * image-binding: which binding is a storage image, -1 (default) for none.
 * The second and third arguments only need to match the module's layout well
 * enough for the driver's shader key; they do not affect correctness here.
 */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        VkResult r_ = (x);                                                                         \
        if (r_ != VK_SUCCESS) {                                                                    \
            fprintf(stderr, "%s -> %d\n", #x, r_);                                                 \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static unsigned char* Slurp(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        perror(path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    *len = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* buf = malloc(*len);
    if (fread(buf, 1, *len, f) != *len) {
        fprintf(stderr, "short read\n");
        exit(1);
    }
    fclose(f);
    return buf;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <shader.spv> [image-binding] [push-bytes]\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    const int image_binding = argc > 2 ? atoi(argv[2]) : -1;
    const int push_bytes = argc > 3 ? atoi(argv[3]) : 128;
    const bool have_spec = argc > 4;
    unsigned spec_values[2] = {argc > 4 ? (unsigned)atoi(argv[4]) : 0u,
                               argc > 5 ? (unsigned)atoi(argv[5]) : 0u};
    const VkSpecializationMapEntry spec_entries[2] = {
        {0u, 0u, sizeof(unsigned)},
        {1u, sizeof(unsigned), sizeof(unsigned)},
    };
    const VkSpecializationInfo spec = {
        .mapEntryCount = 2,
        .pMapEntries = spec_entries,
        .dataSize = sizeof(spec_values),
        .pData = spec_values,
    };

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "vkcompile",
        .apiVersion = VK_API_VERSION_1_2,
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    VkInstance inst;
    CHECK(vkCreateInstance(&ici, NULL, &inst));

    uint32_t count = 0;
    CHECK(vkEnumeratePhysicalDevices(inst, &count, NULL));
    if (count == 0) {
        fprintf(stderr, "no Vulkan devices\n");
        return 1;
    }
    VkPhysicalDevice* pds = malloc(count * sizeof(*pds));
    CHECK(vkEnumeratePhysicalDevices(inst, &count, pds));
    VkPhysicalDevice pd = pds[0];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    fprintf(stderr, "[vkcompile] %s | %s\n", props.deviceName, path);

    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, NULL);
    VkQueueFamilyProperties* qs = malloc(qcount * sizeof(*qs));
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, qs);
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < qcount; i++) {
        if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            family = i;
            break;
        }
    }
    if (family == UINT32_MAX) {
        fprintf(stderr, "no compute queue\n");
        return 1;
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = family,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
    };
    VkDevice dev;
    CHECK(vkCreateDevice(pd, &dci, NULL, &dev));

    VkDescriptorSetLayoutBinding binds[2] = {
        {
            .binding = 0,
            .descriptorType = image_binding == 0 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                 : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = image_binding == 1 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                 : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2,
        .pBindings = binds,
    };
    VkDescriptorSetLayout dsl;
    CHECK(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl));

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = (uint32_t)push_bytes,
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &dsl,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr,
    };
    VkPipelineLayout layout;
    CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &layout));

    size_t len = 0;
    unsigned char* code = Slurp(path, &len);
    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = len,
        .pCode = (const uint32_t*)code,
    };
    VkShaderModule mod;
    CHECK(vkCreateShaderModule(dev, &smci, NULL, &mod));

    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = mod,
                .pName = "main",
                .pSpecializationInfo = have_spec ? &spec : NULL,
            },
        .layout = layout,
    };
    VkPipeline pipe = VK_NULL_HANDLE;
    VkResult r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "[vkcompile] pipeline creation failed: %d\n", r);
        return 1;
    }
    fprintf(stderr, "[vkcompile] ok\n");
    vkDestroyPipeline(dev, pipe, NULL);
    return 0;
}
