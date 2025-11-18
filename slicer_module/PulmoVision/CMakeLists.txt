#-----------------------------------------------------------------------------
set(MODULE_NAME PulmoVision)

#-----------------------------------------------------------------------------
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

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
)

# Optional: keep only if you want the basic wizard unittest
if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)
endif()
