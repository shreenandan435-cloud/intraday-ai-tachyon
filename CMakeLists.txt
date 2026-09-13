cmake_minimum_required(VERSION 3.24)

project(tachyon_sidecar
    VERSION 0.1.0
    DESCRIPTION "TACHYON low-latency TensorRT inference sidecar"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT tachyon_ipo_supported OUTPUT tachyon_ipo_output)
if(tachyon_ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(STATUS "Link-time optimization unavailable: ${tachyon_ipo_output}")
endif()

set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures (sm for local GPU)")

add_compile_options(
    "$<$<CXX_COMPILER_ID:MSVC>:/O2>"
    "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3>"
)

enable_testing()

add_executable(test_spsc_ring
    tests/unit/test_spsc_ring.cpp
    src/tachyon/execution/SharedMemory.cpp)
target_include_directories(test_spsc_ring PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
set_target_properties(test_spsc_ring PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
add_test(NAME spsc_ring COMMAND test_spsc_ring)

set(TACHYON_MISSING_DEPS "")

set(TACHYON_CUDA_LINK "")
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Runtime : ${CUDAToolkit_VERSION} (${CUDAToolkit_LIBRARY_ROOT})")
    list(APPEND TACHYON_CUDA_LINK CUDA::cudart)
else()
    find_path(CUDA_RUNTIME_INCLUDE_DIR NAMES cuda_runtime_api.h
        HINTS "${CUDAToolkit_ROOT}" "$ENV{CUDA_PATH}" "$ENV{TACHYON_CUDA_ROOT}"
        PATH_SUFFIXES include include/nvidia/cuda_runtime)
    find_library(CUDART_LIBRARY NAMES cudart
        HINTS "${CUDAToolkit_ROOT}" "$ENV{CUDA_PATH}" "$ENV{TACHYON_CUDA_ROOT}"
        PATH_SUFFIXES lib/x64 lib lib64)
    if(CUDA_RUNTIME_INCLUDE_DIR AND CUDART_LIBRARY)
        message(STATUS "CUDA Runtime : manual discovery (${CUDA_RUNTIME_INCLUDE_DIR})")
        if(WIN32)
            add_library(cudart_rt SHARED IMPORTED)
            set_target_properties(cudart_rt PROPERTIES
                IMPORTED_IMPLIB "${CUDART_LIBRARY}"
                IMPORTED_LOCATION "${CUDART_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${CUDA_RUNTIME_INCLUDE_DIR}")
        else()
            add_library(cudart_rt UNKNOWN IMPORTED)
            set_target_properties(cudart_rt PROPERTIES
                IMPORTED_LOCATION "${CUDART_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${CUDA_RUNTIME_INCLUDE_DIR}")
        endif()
        list(APPEND TACHYON_CUDA_LINK cudart_rt)
    else()
        string(APPEND TACHYON_MISSING_DEPS
            "  [x] CUDA Toolkit (cudart) not found.\n"
            "      Install CUDA Toolkit 12.8+ (RTX 50xx needs sm_120) and/or pass -DCUDAToolkit_ROOT=<path>.\n")
    endif()
endif()

set(TENSORRT_ROOT "" CACHE PATH "TensorRT 10.x/11.x root directory (contains include/NvInfer.h)")
find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h
    HINTS "${TENSORRT_ROOT}" "$ENV{TENSORRT_ROOT}" "$ENV{NVIDIA_TENSORRT_ROOT}"
    PATHS "/usr/include/x86_64-linux-gnu" "/usr/lib/x86_64-linux-gnu"
    PATH_SUFFIXES include targets/x86_64-linux-gnu/include)
# Windows SDKs ship versioned import libraries (nvinfer_10.lib, nvinfer_11.lib);
# Linux and the minimal runtime extractions use the unversioned name.
find_library(TENSORRT_LIBRARY_NVINFER NAMES nvinfer libnvinfer nvinfer_11 nvinfer_10
    HINTS "${TENSORRT_ROOT}" "$ENV{TENSORRT_ROOT}" "$ENV{NVIDIA_TENSORRT_ROOT}"
    PATH_SUFFIXES lib lib64 lib/x64 targets/x86_64-linux-gnu/lib)

set(TACHYON_TRT_LINK "")
if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY_NVINFER)
    set(NV_TENSORRT_MAJOR "")
    if(EXISTS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h")
        file(READ "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" _nv_version_header)
        string(REGEX MATCH "NV_TENSORRT_MAJOR[ \t]+([0-9]+)" _nv_major_match "${_nv_version_header}")
        if(CMAKE_MATCH_1)
            set(NV_TENSORRT_MAJOR "${CMAKE_MATCH_1}")
        else()
            # TRT 11 indirection: NV_TENSORRT_MAJOR expands to TRT_MAJOR_ENTERPRISE.
            string(REGEX MATCH "TRT_MAJOR_ENTERPRISE[ \t]+([0-9]+)" _nv_major_match "${_nv_version_header}")
            if(CMAKE_MATCH_1)
                set(NV_TENSORRT_MAJOR "${CMAKE_MATCH_1}")
            endif()
        endif()
    endif()
    if(NV_TENSORRT_MAJOR AND NOT (NV_TENSORRT_MAJOR EQUAL 10 OR NV_TENSORRT_MAJOR EQUAL 11))
        string(APPEND TACHYON_MISSING_DEPS
            "  [x] TensorRT ${NV_TENSORRT_MAJOR}.x found at ${TENSORRT_INCLUDE_DIR}; TensorRT 10.x or 11.x required.\n"
            "      Point -DTENSORRT_ROOT at a 10.x/11.x install.\n")
    else()
        message(STATUS "TensorRT     : ${NV_TENSORRT_MAJOR}.x (${TENSORRT_INCLUDE_DIR})")
        if(WIN32)
            add_library(nvinfer SHARED IMPORTED)
            set_target_properties(nvinfer PROPERTIES
                IMPORTED_IMPLIB "${TENSORRT_LIBRARY_NVINFER}"
                IMPORTED_LOCATION "${TENSORRT_LIBRARY_NVINFER}"
                INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}")
        else()
            add_library(nvinfer UNKNOWN IMPORTED)
            set_target_properties(nvinfer PROPERTIES
                IMPORTED_LOCATION "${TENSORRT_LIBRARY_NVINFER}"
                INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}")
        endif()
        list(APPEND TACHYON_TRT_LINK nvinfer)
    endif()
elseif(TENSORRT_INCLUDE_DIR)
    string(APPEND TACHYON_MISSING_DEPS
        "  [x] TensorRT headers found but import library 'nvinfer' not found (searched under ${TENSORRT_ROOT}).\n")
elseif(TENSORRT_LIBRARY_NVINFER)
    string(APPEND TACHYON_MISSING_DEPS
        "  [x] TensorRT library found but NvInfer.h not found (searched under ${TENSORRT_ROOT}).\n")
else()
    string(APPEND TACHYON_MISSING_DEPS
        "  [x] TensorRT 10.x/11.x (nvinfer) not found.\n"
        "      Install TensorRT 10.x/11.x and pass -DTENSORRT_ROOT=<path> (or set TENSORRT_ROOT env var).\n")
endif()

find_package(cppzmq CONFIG QUIET)
set(TACHYON_ZMQ_LINK "")
if(cppzmq_FOUND)
    message(STATUS "ZeroMQ       : cppzmq config package (${cppzmq_DIR})")
    if(TARGET cppzmq::cppzmq)
        list(APPEND TACHYON_ZMQ_LINK cppzmq::cppzmq)
    elseif(TARGET cppzmq)
        list(APPEND TACHYON_ZMQ_LINK cppzmq)
    else()
        string(APPEND TACHYON_MISSING_DEPS
            "  [x] cppzmq config package found but no 'cppzmq' target defined.\n")
    endif()
else()
    find_path(LIBZMQ_INCLUDE_DIR NAMES zmq.h)
    find_path(CPPZMQ_INCLUDE_DIR NAMES zmq.hpp)
    find_library(LIBZMQ_LIBRARY NAMES zmq libzmq)
    if(LIBZMQ_INCLUDE_DIR AND CPPZMQ_INCLUDE_DIR AND LIBZMQ_LIBRARY)
        message(STATUS "ZeroMQ       : manual discovery (${LIBZMQ_LIBRARY})")
        add_library(libzmq UNKNOWN IMPORTED)
        set_target_properties(libzmq PROPERTIES
            IMPORTED_LOCATION "${LIBZMQ_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${LIBZMQ_INCLUDE_DIR};${CPPZMQ_INCLUDE_DIR}")
        list(APPEND TACHYON_ZMQ_LINK libzmq)
    else()
        string(APPEND TACHYON_MISSING_DEPS
            "  [x] ZeroMQ / cppzmq not found (zmq.h=${LIBZMQ_INCLUDE_DIR}, zmq.hpp=${CPPZMQ_INCLUDE_DIR}, lib=${LIBZMQ_LIBRARY}).\n"
            "      Install via vcpkg ('vcpkg install zeromq cppzmq') or system packages, then pass -DZEROMQ_ROOT=<path>.\n")
    endif()
endif()

if(TACHYON_MISSING_DEPS)
    message(FATAL_ERROR
        "tachyon_sidecar configuration failed; unresolved dependencies:\n"
        "${TACHYON_MISSING_DEPS}"
        "Fix the entries above and re-run: cmake -B build -S .\n")
endif()

add_executable(tachyon_sidecar
    src/tachyon/execution/EngineSidecar.cpp
    src/tachyon/execution/SharedMemory.cpp)
add_executable(tachyon_plan_builder src/tachyon/execution/PlanBuilder.cpp)

target_include_directories(tachyon_sidecar PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(tachyon_sidecar PRIVATE
    ${TACHYON_TRT_LINK}
    ${TACHYON_CUDA_LINK}
    ${TACHYON_ZMQ_LINK})

target_link_libraries(tachyon_plan_builder PRIVATE
    ${TACHYON_TRT_LINK}
    ${TACHYON_CUDA_LINK})

set_target_properties(tachyon_sidecar tachyon_plan_builder PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION TRUE
    CUDA_ARCHITECTURES "native"
    MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

set(TACHYON_EXTRA_DLLS "" CACHE STRING "Runtime DLLs copied next to the executables after build")
foreach(_tachyon_dll IN LISTS TACHYON_EXTRA_DLLS)
    if(EXISTS "${_tachyon_dll}")
        add_custom_command(TARGET tachyon_sidecar POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_tachyon_dll}" "$<TARGET_FILE_DIR:tachyon_sidecar>")
        add_custom_command(TARGET tachyon_plan_builder POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_tachyon_dll}" "$<TARGET_FILE_DIR:tachyon_plan_builder>")
    else()
        message(WARNING "TACHYON_EXTRA_DLLS entry does not exist: ${_tachyon_dll}")
    endif()
endforeach()

install(TARGETS tachyon_sidecar tachyon_plan_builder RUNTIME DESTINATION bin)
