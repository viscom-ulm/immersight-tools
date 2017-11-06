cmake_minimum_required(VERSION 3.2)
project(ImmersightTools)

find_package( OpenCV REQUIRED )
find_package( OpenGL REQUIRED )
find_package(ImmersightTools REQUIRED)

set(HEADER_FILES 
include/tools.hpp
include/defines.hpp
include/opencv.hpp
include/util.h
)
set(SRC_FILES 
src/tools.cpp
src/opencv.cpp
src/util.cpp
)

include_directories("extern/glm")


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../lib)

add_library(ImmersightTools ${HEADER_FILES} ${SRC_FILES})

#add_executable(ImmersightToolsTests tests/main.cpp)

target_link_libraries( ImmersightTools ${OpenCV_LIBS} ${OPENGL_LIBRARY}) #glfw3 libglew32 glew32)
#target_link_libraries( ImmersightToolsTests ${OpenCV_LIBS} ${OPENGL_LIBRARY} glfw3 libglew32 glew32 ImmersightTools)