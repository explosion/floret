#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

cmake_minimum_required(VERSION 2.8.9)
project(floret)

# The version number.
set (floret_VERSION_MAJOR 0)
set (floret_VERSION_MINOR 1)

include_directories(fasttext)

set(CMAKE_CXX_FLAGS " -pthread -std=c++11 -funroll-loops -O3")

set(HEADER_FILES
    src/MurmurHash3.h
    src/args.h
    src/autotune.h
    src/densematrix.h
    src/dictionary.h
    src/fasttext.h
    src/loss.h
    src/matrix.h
    src/meter.h
    src/model.h
    src/productquantizer.h
    src/quantmatrix.h
    src/real.h
    src/utils.h
    src/vector.h)

set(SOURCE_FILES
    src/MurmurHash3.cpp
    src/args.cc
    src/autotune.cc
    src/densematrix.cc
    src/dictionary.cc
    src/fasttext.cc
    src/loss.cc
    src/main.cc
    src/matrix.cc
    src/meter.cc
    src/model.cc
    src/productquantizer.cc
    src/quantmatrix.cc
    src/utils.cc
    src/vector.cc)


# std::atomic<int64_t> needs libatomic on 32-bit targets without 64-bit
# atomic instructions, e.g. riscv32. The variable is global so the compiler
# cannot optimize the atomic operations away.
include(CheckCXXSourceCompiles)
set(ATOMIC64_TEST "
#include <atomic>
#include <cstdint>
std::atomic<int64_t> v;
int main() {
  v.fetch_add(1);
  return static_cast<int>(v.load());
}")
check_cxx_source_compiles("${ATOMIC64_TEST}" HAVE_ATOMIC64)
if (NOT HAVE_ATOMIC64)
  set(CMAKE_REQUIRED_LIBRARIES atomic)
  check_cxx_source_compiles("${ATOMIC64_TEST}" HAVE_ATOMIC64_WITH_LIBATOMIC)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if (HAVE_ATOMIC64_WITH_LIBATOMIC)
    set(ATOMIC_LIBRARY atomic)
    set(PC_LIBS_PRIVATE "-latomic")
  endif()
endif()

if (NOT MSVC)
  include(GNUInstallDirs)
  configure_file("floret.pc.in" "floret.pc" @ONLY)
  install(FILES "${CMAKE_BINARY_DIR}/floret.pc" DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()

add_library(floret-shared SHARED ${SOURCE_FILES} ${HEADER_FILES})
add_library(floret-static STATIC ${SOURCE_FILES} ${HEADER_FILES})
add_library(floret-static_pic STATIC ${SOURCE_FILES} ${HEADER_FILES})
set_target_properties(floret-shared PROPERTIES OUTPUT_NAME floret
  SOVERSION "${floret_VERSION_MAJOR}")
set_target_properties(floret-static PROPERTIES OUTPUT_NAME floret)
set_target_properties(floret-static_pic PROPERTIES OUTPUT_NAME floret_pic
  POSITION_INDEPENDENT_CODE True)
target_link_libraries(floret-shared ${ATOMIC_LIBRARY})
target_link_libraries(floret-static ${ATOMIC_LIBRARY})
target_link_libraries(floret-static_pic ${ATOMIC_LIBRARY})

add_executable(floret-bin src/main.cc)
target_link_libraries(floret-bin pthread floret-static)
set_target_properties(floret-bin PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" OUTPUT_NAME floret)
install (TARGETS floret-shared
    LIBRARY DESTINATION lib)
install (TARGETS floret-static
    ARCHIVE DESTINATION lib)
install (TARGETS floret-static_pic
    ARCHIVE DESTINATION lib)
install (TARGETS floret-bin
    RUNTIME DESTINATION bin
 PUBLIC_HEADER DESTINATION include/floret)
