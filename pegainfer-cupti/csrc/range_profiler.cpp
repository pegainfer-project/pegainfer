#include <cuda.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_range_profiler.h>
#include <cupti_result.h>
#include <cupti_target.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using PegaCuptiCallback = int (*)(void *);

namespace {

std::once_flag g_cupti_init_once;
CUptiResult g_cupti_init_status = CUPTI_SUCCESS;

void write_error(char *error, size_t error_len, const std::string &message) {
    if (error == nullptr || error_len == 0) {
        return;
    }
    const size_t copy_len = std::min(error_len - 1, message.size());
    std::memcpy(error, message.data(), copy_len);
    error[copy_len] = '\0';
}

std::string cupti_error(CUptiResult status, const char *call) {
    const char *name = nullptr;
    const char *description = nullptr;
    cuptiGetResultString(status, &name);
    cuptiGetResultString(status, &description);
    std::ostringstream out;
    out << call << " failed";
    if (name != nullptr) {
        out << ": " << name;
    }
    if (description != nullptr && description != name) {
        out << " (" << description << ")";
    }
    return out.str();
}

std::string cuda_error(CUresult status, const char *call) {
    const char *name = nullptr;
    const char *description = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &description);
    std::ostringstream out;
    out << call << " failed";
    if (name != nullptr) {
        out << ": " << name;
    }
    if (description != nullptr) {
        out << " (" << description << ")";
    }
    return out.str();
}

void check_cupti(CUptiResult status, const char *call) {
    if (status != CUPTI_SUCCESS) {
        throw std::runtime_error(cupti_error(status, call));
    }
}

void check_cuda(CUresult status, const char *call) {
    if (status != CUDA_SUCCESS) {
        throw std::runtime_error(cuda_error(status, call));
    }
}

void initialize_cupti_once() {
    std::call_once(g_cupti_init_once, []() {
        CUpti_Profiler_Initialize_Params params = {
            CUpti_Profiler_Initialize_Params_STRUCT_SIZE
        };
        g_cupti_init_status = cuptiProfilerInitialize(&params);
    });
    check_cupti(g_cupti_init_status, "cuptiProfilerInitialize");
}

struct CurrentContextGuard {
    CUcontext previous = nullptr;

    explicit CurrentContextGuard(CUcontext context) {
        check_cuda(cuCtxGetCurrent(&previous), "cuCtxGetCurrent");
        check_cuda(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    }

    ~CurrentContextGuard() {
        cuCtxSetCurrent(previous);
    }
};

struct HostObject {
    CUpti_Profiler_Host_Object *ptr = nullptr;

    HostObject() = default;
    HostObject(const HostObject &) = delete;
    HostObject &operator=(const HostObject &) = delete;

    HostObject(HostObject &&other) noexcept : ptr(std::exchange(other.ptr, nullptr)) {}

    HostObject &operator=(HostObject &&other) noexcept {
        if (this != &other) {
            ptr = std::exchange(other.ptr, nullptr);
        }
        return *this;
    }

    ~HostObject() {
        if (ptr != nullptr) {
            CUpti_Profiler_Host_Deinitialize_Params params = {
                CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE
            };
            params.pHostObject = ptr;
            cuptiProfilerHostDeinitialize(&params);
        }
    }
};

struct RangeObject {
    CUpti_RangeProfiler_Object *ptr = nullptr;

    RangeObject() = default;
    RangeObject(const RangeObject &) = delete;
    RangeObject &operator=(const RangeObject &) = delete;

    RangeObject(RangeObject &&other) noexcept : ptr(std::exchange(other.ptr, nullptr)) {}

    RangeObject &operator=(RangeObject &&other) noexcept {
        if (this != &other) {
            ptr = std::exchange(other.ptr, nullptr);
        }
        return *this;
    }

    ~RangeObject() {
        if (ptr != nullptr) {
            CUpti_RangeProfiler_Disable_Params params = {
                CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE
            };
            params.pRangeProfilerObject = ptr;
            cuptiRangeProfilerDisable(&params);
        }
    }
};

std::string chip_name(size_t device_index) {
    CUpti_Device_GetChipName_Params params = {CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    params.deviceIndex = device_index;
    check_cupti(cuptiDeviceGetChipName(&params), "cuptiDeviceGetChipName");
    return params.pChipName;
}

std::vector<uint8_t> counter_availability_image(CUcontext context) {
    CUpti_Profiler_GetCounterAvailability_Params params = {
        CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE
    };
    params.ctx = context;
    check_cupti(cuptiProfilerGetCounterAvailability(&params),
                "cuptiProfilerGetCounterAvailability(size)");

    std::vector<uint8_t> image(params.counterAvailabilityImageSize);
    params.pCounterAvailabilityImage = image.data();
    check_cupti(cuptiProfilerGetCounterAvailability(&params),
                "cuptiProfilerGetCounterAvailability(data)");
    return image;
}

HostObject initialize_host(const std::string &chip,
                           const std::vector<uint8_t> &counter_availability) {
    HostObject host;
    CUpti_Profiler_Host_Initialize_Params params = {
        CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE
    };
    params.profilerType = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
    params.pChipName = chip.c_str();
    params.pCounterAvailabilityImage = counter_availability.data();
    check_cupti(cuptiProfilerHostInitialize(&params), "cuptiProfilerHostInitialize");
    host.ptr = params.pHostObject;
    return host;
}

std::vector<uint8_t> create_config_image(CUpti_Profiler_Host_Object *host,
                                         const char **metric_names,
                                         size_t metric_count) {
    CUpti_Profiler_Host_ConfigAddMetrics_Params add = {
        CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE
    };
    add.pHostObject = host;
    add.ppMetricNames = metric_names;
    add.numMetrics = metric_count;
    check_cupti(cuptiProfilerHostConfigAddMetrics(&add),
                "cuptiProfilerHostConfigAddMetrics");

    CUpti_Profiler_Host_GetConfigImageSize_Params get_size = {
        CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE
    };
    get_size.pHostObject = host;
    check_cupti(cuptiProfilerHostGetConfigImageSize(&get_size),
                "cuptiProfilerHostGetConfigImageSize");

    std::vector<uint8_t> config(get_size.configImageSize);
    CUpti_Profiler_Host_GetConfigImage_Params get_image = {
        CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE
    };
    get_image.pHostObject = host;
    get_image.pConfigImage = config.data();
    get_image.configImageSize = config.size();
    check_cupti(cuptiProfilerHostGetConfigImage(&get_image),
                "cuptiProfilerHostGetConfigImage");
    return config;
}

RangeObject enable_range_profiler(CUcontext context) {
    RangeObject range;
    CUpti_RangeProfiler_Enable_Params params = {CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE};
    params.ctx = context;
    check_cupti(cuptiRangeProfilerEnable(&params), "cuptiRangeProfilerEnable");
    range.ptr = params.pRangeProfilerObject;
    return range;
}

std::vector<uint8_t> create_counter_data_image(CUpti_RangeProfiler_Object *range,
                                               const char **metric_names,
                                               size_t metric_count) {
    CUpti_RangeProfiler_GetCounterDataSize_Params get_size = {
        CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE
    };
    get_size.pRangeProfilerObject = range;
    get_size.pMetricNames = metric_names;
    get_size.numMetrics = metric_count;
    get_size.maxNumOfRanges = 1;
    get_size.maxNumRangeTreeNodes = 1;
    check_cupti(cuptiRangeProfilerGetCounterDataSize(&get_size),
                "cuptiRangeProfilerGetCounterDataSize");

    std::vector<uint8_t> counter_data(get_size.counterDataSize, 0);
    CUpti_RangeProfiler_CounterDataImage_Initialize_Params init = {
        CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE
    };
    init.pRangeProfilerObject = range;
    init.pCounterData = counter_data.data();
    init.counterDataSize = counter_data.size();
    check_cupti(cuptiRangeProfilerCounterDataImageInitialize(&init),
                "cuptiRangeProfilerCounterDataImageInitialize");
    return counter_data;
}

void set_config(CUpti_RangeProfiler_Object *range, const std::vector<uint8_t> &config,
                std::vector<uint8_t> &counter_data) {
    CUpti_RangeProfiler_SetConfig_Params params = {
        CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE
    };
    params.pRangeProfilerObject = range;
    params.pConfig = config.data();
    params.configSize = config.size();
    params.pCounterDataImage = counter_data.data();
    params.counterDataImageSize = counter_data.size();
    params.maxRangesPerPass = 1;
    params.numNestingLevels = 1;
    params.minNestingLevel = 1;
    params.passIndex = 0;
    params.targetNestingLevel = 1;
    params.range = CUPTI_UserRange;
    params.replayMode = CUPTI_UserReplay;
    check_cupti(cuptiRangeProfilerSetConfig(&params), "cuptiRangeProfilerSetConfig");
}

void profile_passes(CUpti_RangeProfiler_Object *range, const char *range_name,
                    PegaCuptiCallback prepare_fn, PegaCuptiCallback launch_fn,
                    void *userdata) {
    bool all_passes_submitted = false;
    do {
        if (prepare_fn != nullptr && prepare_fn(userdata) != 0) {
            throw std::runtime_error("CUPTI prepare callback failed");
        }
        check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(prepare)");

        CUpti_RangeProfiler_Start_Params start = {CUpti_RangeProfiler_Start_Params_STRUCT_SIZE};
        start.pRangeProfilerObject = range;
        check_cupti(cuptiRangeProfilerStart(&start), "cuptiRangeProfilerStart");

        CUpti_RangeProfiler_PushRange_Params push = {
            CUpti_RangeProfiler_PushRange_Params_STRUCT_SIZE
        };
        push.pRangeProfilerObject = range;
        push.pRangeName = range_name;
        check_cupti(cuptiRangeProfilerPushRange(&push), "cuptiRangeProfilerPushRange");

        if (launch_fn == nullptr || launch_fn(userdata) != 0) {
            throw std::runtime_error("CUPTI launch callback failed");
        }
        check_cuda(cuCtxSynchronize(), "cuCtxSynchronize(profiled range)");

        CUpti_RangeProfiler_PopRange_Params pop = {
            CUpti_RangeProfiler_PopRange_Params_STRUCT_SIZE
        };
        pop.pRangeProfilerObject = range;
        check_cupti(cuptiRangeProfilerPopRange(&pop), "cuptiRangeProfilerPopRange");

        CUpti_RangeProfiler_Stop_Params stop = {CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE};
        stop.pRangeProfilerObject = range;
        check_cupti(cuptiRangeProfilerStop(&stop), "cuptiRangeProfilerStop");
        all_passes_submitted = stop.isAllPassSubmitted;
    } while (!all_passes_submitted);
}

void decode_data(CUpti_RangeProfiler_Object *range) {
    CUpti_RangeProfiler_DecodeData_Params params = {
        CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE
    };
    params.pRangeProfilerObject = range;
    check_cupti(cuptiRangeProfilerDecodeData(&params), "cuptiRangeProfilerDecodeData");
}

void evaluate(CUpti_Profiler_Host_Object *host, std::vector<uint8_t> &counter_data,
              const char **metric_names, size_t metric_count, double *metric_values,
              size_t metric_value_count) {
    if (metric_value_count < metric_count) {
        throw std::runtime_error("metric output buffer is smaller than metric list");
    }

    CUpti_RangeProfiler_GetCounterDataInfo_Params info = {
        CUpti_RangeProfiler_GetCounterDataInfo_Params_STRUCT_SIZE
    };
    info.pCounterDataImage = counter_data.data();
    info.counterDataImageSize = counter_data.size();
    check_cupti(cuptiRangeProfilerGetCounterDataInfo(&info),
                "cuptiRangeProfilerGetCounterDataInfo");
    if (info.numTotalRanges == 0) {
        throw std::runtime_error("CUPTI returned zero profiled ranges");
    }

    CUpti_Profiler_Host_EvaluateToGpuValues_Params params = {
        CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE
    };
    params.pHostObject = host;
    params.pCounterDataImage = counter_data.data();
    params.counterDataImageSize = counter_data.size();
    params.ppMetricNames = metric_names;
    params.numMetrics = metric_count;
    params.rangeIndex = 0;
    params.pMetricValues = metric_values;
    check_cupti(cuptiProfilerHostEvaluateToGpuValues(&params),
                "cuptiProfilerHostEvaluateToGpuValues");
}

} // namespace

extern "C" int pegainfer_cupti_profile_range(
    CUcontext context, size_t device_index, const char *range_name,
    const char **metric_names, size_t metric_count, PegaCuptiCallback prepare_fn,
    PegaCuptiCallback launch_fn, void *userdata, double *metric_values,
    size_t metric_value_count, char *error, size_t error_len) {
    try {
        if (context == nullptr) {
            throw std::runtime_error("CUDA context is null");
        }
        if (range_name == nullptr) {
            throw std::runtime_error("range name is null");
        }
        if (metric_count == 0) {
            throw std::runtime_error("metric list is empty");
        }
        if (metric_names == nullptr || metric_values == nullptr) {
            throw std::runtime_error("metric input or output buffer is null");
        }

        initialize_cupti_once();
        CurrentContextGuard current_context(context);
        const std::string chip = chip_name(device_index);
        const std::vector<uint8_t> availability = counter_availability_image(context);
        HostObject host = initialize_host(chip, availability);
        const std::vector<uint8_t> config =
            create_config_image(host.ptr, metric_names, metric_count);
        RangeObject range = enable_range_profiler(context);
        std::vector<uint8_t> counter_data =
            create_counter_data_image(range.ptr, metric_names, metric_count);

        set_config(range.ptr, config, counter_data);
        profile_passes(range.ptr, range_name, prepare_fn, launch_fn, userdata);
        decode_data(range.ptr);
        evaluate(host.ptr, counter_data, metric_names, metric_count, metric_values,
                 metric_value_count);
        write_error(error, error_len, "");
        return 0;
    } catch (const std::exception &ex) {
        write_error(error, error_len, ex.what());
        return 1;
    } catch (...) {
        write_error(error, error_len, "unknown CUPTI profiler error");
        return 1;
    }
}
