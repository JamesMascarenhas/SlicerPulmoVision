set(MODULE_NAME PulmoVision)

set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py

  PulmoBackend/__init__.py
  PulmoBackend/inference.py
  PulmoBackend/pipeline.py
  PulmoBackend/postprocessing.py
  PulmoBackend/preprocessing.py
  PulmoBackend/radiomics.py
  PulmoBackend/training.py
  PulmoBackend/unet3d.py
  PulmoBackend/utils.py
)

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
)

slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
)

if(BUILD_TESTING)
  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  # Can remove this if want to keep all testing in the Testing folder
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)
  # Additional build-time testing
  add_subdirectory(Testing)
endif()
