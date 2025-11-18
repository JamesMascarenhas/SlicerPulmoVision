#-----------------------------------------------------------------------------
set(MODULE_NAME PulmoVision)

# Python files that define your module
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/Logic.py
  ${MODULE_NAME}Lib/Utils.py
)

# Non-code resources: STILL NEED TO CREATE THESE
set(MODULE_PYTHON_RESOURCES
  # Resources/Icons/${MODULE_NAME}.png
  # Resources/UI/${MODULE_NAME}.ui
)

# Build the scripted module
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
)

# Optional: only if we add a PulmoVisionTest class in PulmoVision.py
#f if(BUILD_TESTING)
#f  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)
#f endif()
