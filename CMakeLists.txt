cmake_minimum_required(VERSION 3.13.4)

project(PulmoVision)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/JamesMascarenhas/SlicerPulmoVision")   
set(EXTENSION_CATEGORY "Radiology")                                         
set(EXTENSION_CONTRIBUTORS "James Mascarenhas (Queen's University)")
set(EXTENSION_DESCRIPTION "PulmoVision is a 3D Slicer extension providing an end-to-end pipeline for CT lung tumor preprocessing, 3D U-Net segmentation, and quality assessment.")
# These URLs only matter if you host an icon/screenshot on GitHub later
set(EXTENSION_ICONURL "")           
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "NA")       # no extra Slicer extensions required

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules (directories)
add_subdirectory(slicer_module/PulmoVision)   # folder that contains PulmoVision.py + its CMakeLists

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
