# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vlsilab/Documents/opencv3src/svm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vlsilab/Documents/opencv3src/svm

# Include any dependencies generated for this target.
include CMakeFiles/svmtrainer.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/svmtrainer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/svmtrainer.dir/flags.make

CMakeFiles/svmtrainer.dir/svmtrainer.cpp.o: CMakeFiles/svmtrainer.dir/flags.make
CMakeFiles/svmtrainer.dir/svmtrainer.cpp.o: svmtrainer.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/vlsilab/Documents/opencv3src/svm/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/svmtrainer.dir/svmtrainer.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/svmtrainer.dir/svmtrainer.cpp.o -c /home/vlsilab/Documents/opencv3src/svm/svmtrainer.cpp

CMakeFiles/svmtrainer.dir/svmtrainer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/svmtrainer.dir/svmtrainer.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/vlsilab/Documents/opencv3src/svm/svmtrainer.cpp > CMakeFiles/svmtrainer.dir/svmtrainer.cpp.i

CMakeFiles/svmtrainer.dir/svmtrainer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/svmtrainer.dir/svmtrainer.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/vlsilab/Documents/opencv3src/svm/svmtrainer.cpp -o CMakeFiles/svmtrainer.dir/svmtrainer.cpp.s

CMakeFiles/svmtrainer.dir/svmtrainer.cpp.o.requires:
.PHONY : CMakeFiles/svmtrainer.dir/svmtrainer.cpp.o.requires

CMakeFiles/svmtrainer.dir/svmtrainer.cpp.o.provides: CMakeFiles/svmtrainer.dir/svmtrainer.cpp.o.requires
	$(MAKE) -f CMakeFiles/svmtrainer.dir/build.make CMakeFiles/svmtrainer.dir/svmtrainer.cpp.o.provides.build
.PHONY : CMakeFiles/svmtrainer.dir/svmtrainer.cpp.o.provides

CMakeFiles/svmtrainer.dir/svmtrainer.cpp.o.provides.build: CMakeFiles/svmtrainer.dir/svmtrainer.cpp.o

# Object files for target svmtrainer
svmtrainer_OBJECTS = \
"CMakeFiles/svmtrainer.dir/svmtrainer.cpp.o"

# External object files for target svmtrainer
svmtrainer_EXTERNAL_OBJECTS =

svmtrainer: CMakeFiles/svmtrainer.dir/svmtrainer.cpp.o
svmtrainer: CMakeFiles/svmtrainer.dir/build.make
svmtrainer: /usr/local/lib/libopencv_viz.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_videostab.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_videoio.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_video.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_superres.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_stitching.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_shape.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_photo.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_objdetect.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_ml.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_imgproc.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_highgui.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_flann.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_features2d.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_core.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_calib3d.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_features2d.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_ml.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_highgui.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_videoio.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_flann.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_video.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_imgproc.so.3.1.0
svmtrainer: /usr/local/lib/libopencv_core.so.3.1.0
svmtrainer: CMakeFiles/svmtrainer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable svmtrainer"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/svmtrainer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/svmtrainer.dir/build: svmtrainer
.PHONY : CMakeFiles/svmtrainer.dir/build

CMakeFiles/svmtrainer.dir/requires: CMakeFiles/svmtrainer.dir/svmtrainer.cpp.o.requires
.PHONY : CMakeFiles/svmtrainer.dir/requires

CMakeFiles/svmtrainer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/svmtrainer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/svmtrainer.dir/clean

CMakeFiles/svmtrainer.dir/depend:
	cd /home/vlsilab/Documents/opencv3src/svm && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vlsilab/Documents/opencv3src/svm /home/vlsilab/Documents/opencv3src/svm /home/vlsilab/Documents/opencv3src/svm /home/vlsilab/Documents/opencv3src/svm /home/vlsilab/Documents/opencv3src/svm/CMakeFiles/svmtrainer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/svmtrainer.dir/depend

