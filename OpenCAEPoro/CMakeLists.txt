# Main CMakeLists file for OpenCAEPoroX
#
# Sample usages:
#   cmake .                             // build in default configuration
#   cmake -DCMAKE_BUILD_TYPE=Release .  // build in Release configuration
#   cmake -DCMAKE_BUILD_TYPE=Debug .    // build in Debug configuration
#   cmake -DCMAKE_CXX_COMPILER=g++ .    // build with GNU C++ compiler
#   cmake -DCMAKE_VERBOSE_MAKEFILE=ON . // build with verbose on
#   cmake -DUSE_FASPCPR=ON .            // build with FASPCPR support
#   cmake -DUSE_FASP4BLKOIL=ON .        // build with FASP4BLKOIL support
#   cmake -DUSE_FASP4CUDA=ON .          // build with FASP4CUDA support
#   cmake -DUSE_UMFPACK=ON .            // build with UMFPACK support
#   cmake -DUSE_OPENMP=ON .             // build with OpenMP support

###############################################################################
## General environment setting
###############################################################################

# Minimum cmake version needed
cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0076 NEW)

# Name of the project: should appear after find_program
project(OpenCAEPoroX C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)

# Set module lookup path
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/modules)

# Add math and stdc++ libraries if not included by default
if (UNIX OR LINUX)
    set(ADD_STDLIBS m stdc++)
endif()

# Do not skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)
 
# When building, don't use the install RPATH already
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
 
# Add the automatically determined parts of the RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# LIBNAME is used in the external dependencies, to make it easier if we 
# decide to change the name in the future
set(LIBNAME OpenCAEPoroX CACHE INTERNAL "The library name")
add_library(OpenCAEPoroX STATIC)

###############################################################################
## External dependencies
###############################################################################

add_subdirectory(external)

###############################################################################
## Set configuration types and default compiler flags
###############################################################################

set(CMAKE_CONFIGURATION_TYPES "Release;Debug" CACHE STRING "Configs" FORCE)
if (MSVC)
    # Compiler flags for Windows MSVC
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O3")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /DDEBUG /W4")
else()
    # Compiler flags for Linux or Mac OS X
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 ")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -Wall -g")
endif()

###############################################################################
## Project specific parameters
###############################################################################

add_subdirectory(src)
add_subdirectory(include)
add_subdirectory(main)

# set default build type: Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "the build type" FORCE)
endif()
message(STATUS "Set build type to ${CMAKE_BUILD_TYPE} configuration")

option(BUILD_TEST "Build test" OFF)
if(BUILD_TEST)
    include(CTest)
endif()


if(WIN32)
	MESSAGE(STATUS "Windows Environments Now")
	find_package(MPI REQUIRED)
	target_link_libraries(${LIBNAME} PUBLIC MPI::MPI_CXX)
endif()

###############################################################################
## Installtion targets for lib and executable files
###############################################################################

# Set install location for OpenCAEPoroX library
set(CMAKE_INSTALL_LIBDIR "${PROJECT_SOURCE_DIR}/lib/")
install(TARGETS OpenCAEPoroX
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# CMake uninstall target
if(NOT TARGET uninstall)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/modules/cmake_uninstall.cmake.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
                   IMMEDIATE @ONLY)
    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
