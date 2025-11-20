import logging
import os
from typing import Annotated, Optional

import vtk

import slicer
from slicer.i18n import tr as _
import qt
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
    Choice,
)

from slicer import vtkMRMLScalarVolumeNode, vtkMRMLTableNode 


#
# PulmoVision
#


class PulmoVision(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("PulmoVision")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#PulmoVision">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # PulmoVision1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="PulmoVision",
        sampleName="PulmoVision1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "PulmoVision1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="PulmoVision1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="PulmoVision1",
    )

    # PulmoVision2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="PulmoVision",
        sampleName="PulmoVision2",
        thumbnailFileName=os.path.join(iconsPath, "PulmoVision2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="PulmoVision2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="PulmoVision2",
    )


#
# PulmoVisionParameterNode
#


@parameterNodeWrapper
class PulmoVisionParameterNode:
    """
    The parameters needed by module.

    inputVolume - CT input volume
    windowCenter/windowWidth - CT windowing parameters forwarded to the backend
    segmentationMethod - Backend segmentation algoirthm (percentile, unet3d, or auto)
    segmentationPercentile - Percent threshold used by the lightweight method
    segmentationWeightsPath - Optional path to a UNet3D checkpoint file
    segmentationDevice - Compute device hint for UNet3D inference
    segmentationThreshold - Binarization threshold for UNet3D output probabilities
    postprocessEnabled - Toggle cleanup stage
    keepLargestComponent/minComponentSize - Postprocessing hints (placeholder in v1)
    outputMaskVolume - Segmentation mask destination
    outputFeatureTable -Table node for radiomics summary
    """

    inputVolume: vtkMRMLScalarVolumeNode
    windowCenter: Annotated[float, WithinRange(-1200.0, 400.0)] = -600.0
    windowWidth: Annotated[float, WithinRange(100.0, 4000.0)] = 1500.0
    segmentationMethod: Annotated[str, Choice(["auto", "percentile", "unet3d"])] = "auto"
    segmentationPercentile: Annotated[float, WithinRange(80.0, 100.0)] = 99.0
    segmentationDevice: str = ""
    segmentationThreshold: Annotated[float, WithinRange(0.0, 1.0)] = 0.5
    postprocessEnabled: bool = True
    keepLargestComponent: bool = False
    minComponentSize: int = 0
    radiomicsEnabled: bool = True
    outputMaskVolume: vtkMRMLScalarVolumeNode
    outputFeatureTable: vtkMRMLTableNode


#
# PulmoVisionWidget
#


class PulmoVisionWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/PulmoVision.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)
        for mrmlWidgetName in [
            "inputSelector",
            "outputMaskSelector",
            "featureTableSelector",
        ]:
            mrmlWidget = getattr(self.ui, mrmlWidgetName, None)
            if mrmlWidget:
                mrmlWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = PulmoVisionLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        if hasattr(self.ui, "featureTableView"):
            self.ui.featureTableView.setMRMLScene(slicer.mrmlScene)

        self.segmentationInfoLabel = qt.QLabel(_("Segmentation: not run"))
        self.segmentationInfoLabel.setToolTip(
            _("Displays which backend ran and which checkpoint path was used for segmentation.")
        )
        if hasattr(self.ui, "statusLayout"):
            self.ui.statusLayout.addWidget(self.segmentationInfoLabel)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input node if nothing is selected yet
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

        # Create default mask volume if needed
        if not self._parameterNode.outputMaskVolume:
            maskNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "PulmoVisionMask")
            self._parameterNode.outputMaskVolume = maskNode

        # Create default feature table node if needed
        if not self._parameterNode.outputFeatureTable:
            tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "PulmoVisionRadiomics")
            self._parameterNode.outputFeatureTable = tableNode
            if hasattr(self.ui, "featureTableView"):
                self.ui.featureTableView.setMRMLTableNode(tableNode)

    def setParameterNode(self, inputParameterNode: Optional[PulmoVisionParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()
            if hasattr(self.ui, "featureTableView"):
                self.ui.featureTableView.setMRMLTableNode(self._parameterNode.outputFeatureTable)

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.outputMaskVolume:
            self.ui.applyButton.toolTip = _("Run PulmoVision segmentation and radiomics")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output nodes")
            self.ui.applyButton.enabled = False
        if hasattr(self.ui, "featureTableView") and self._parameterNode:
            self.ui.featureTableView.setMRMLTableNode(self._parameterNode.outputFeatureTable)

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        self.ui.statusLabel.setText(_("Running PulmoVision pipeline…"))
        result = None
        try:
            with slicer.util.tryWithErrorDisplay(_("PulmoVision processing failed."), waitCursor=True):
                 result = self.logic.process(self._parameterNode, showResult=True)
        except Exception:
            self.ui.statusLabel.setText(_("PulmoVision processing failed"))
            raise
        else:
            self.ui.statusLabel.setText(_("PulmoVision completed successfully"))
            self._updateSegmentationInfoLabel(result)

    def _updateSegmentationInfoLabel(self, result) -> None:
        if not hasattr(self, "segmentationInfoLabel") or self.segmentationInfoLabel is None:
            return

        metadata = None
        if isinstance(result, dict):
            metadata = result.get("segmentation_metadata")

        if not metadata:
            self.segmentationInfoLabel.setText(_("Segmentation: unknown"))
            self.segmentationInfoLabel.setToolTip(
                _("No segmentation metadata was returned by the pipeline.")
            )
            return

        used_method = metadata.get("used_method") or metadata.get("requested_method") or "?"
        checkpoint_path = metadata.get("weights_path") or _("Default checkpoint")
        checkpoint_loaded = metadata.get("checkpoint_loaded")
        checkpoint_exists = metadata.get("checkpoint_exists")

        self.segmentationInfoLabel.setText(_(f"Segmentation: {used_method}"))
        tooltip_lines = [
            _(f"Requested: {metadata.get('requested_method', 'n/a')}")
            if metadata.get("requested_method")
            else _("Requested: n/a"),
            _(f"Checkpoint path: {checkpoint_path}"),
            _(
                f"Checkpoint status: {'available' if checkpoint_exists else 'missing'}; "
                f"{'loaded' if checkpoint_loaded else 'not loaded'}"
            ),
        ]
        if metadata.get("messages"):
            tooltip_lines.append("\n".join(metadata["messages"]))
        self.segmentationInfoLabel.setToolTip("\n".join(tooltip_lines))


# 
# PulmoVisionLogic
#


class PulmoVisionLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return PulmoVisionParameterNode(super().getParameterNode())

    def process(self, parameterNode: PulmoVisionParameterNode, showResult: bool = True):
        """
        Run the PulmoVision pipeline based on the parameter node values

        This executes CT windowing, segmentation, optional postprocessing, and 
        updates both the mask volume and the radiomics feature table
        Returns a dictionary with the computed mask and (if avaliable) features
        """ 

        if not parameterNode.inputVolume:
            raise ValueError("Input volume is invalid")
        if not parameterNode.outputMaskVolume:
            raise ValueError("Output mask volume is invalid")
        
        import time
        import numpy as np
        from PulmoBackend.pipeline import run_pulmo_pipeline

        startTime = time.time()
        logging.info("PulmoVision: processing started")

        inputVolume = parameterNode.inputVolume
        outputMaskVolume = parameterNode.outputMaskVolume

        # Slicer represents scalar volumes as (D, H, W) in numpy
        inputArray_DHW = slicer.util.arrayFromVolume(inputVolume)
        if inputArray_DHW.ndim != 3:
            raise ValueError(
                f"Expected 3D volume, got array with shape {inputArray_DHW.shape}"
            )

        # Backend convention: (H, W, D)
        volume_HWD = np.transpose(inputArray_DHW, (1, 2, 0)).astype(np.float32)

        segmentation_method = parameterNode.segmentationMethod.lower().strip()
        seg_kwargs = {}

        if "percentile" in segmentation_method:
            seg_kwargs["percentile"] = float(parameterNode.segmentationPercentile)
            segmentation_method = "percentile"

        elif "unet3d" in segmentation_method:
            seg_kwargs["device"] = parameterNode.segmentationDevice or None
            seg_kwargs["threshold"] = float(parameterNode.segmentationThreshold)
            segmentation_method = "unet3d"

        elif not segmentation_method or "auto" in segmentation_method:
            segmentation_method = "auto"
            seg_kwargs = {
                "device": parameterNode.segmentationDevice or None,
                "threshold": float(parameterNode.segmentationThreshold),
                "percentile": float(parameterNode.segmentationPercentile),
            }

        post_kwargs = {
            "keep_largest_component": bool(parameterNode.keepLargestComponent),
            "min_size_voxels": int(parameterNode.minComponentSize),
        }

        spacing_ijk = inputVolume.GetSpacing()
        voxel_spacing_hwd = (float(spacing_ijk[1]), float(spacing_ijk[0]), float(spacing_ijk[2]))

        pipeline_result = run_pulmo_pipeline(
            volume_HWD,
            window_center=float(parameterNode.windowCenter),
            window_width=float(parameterNode.windowWidth),
            normalize=True,
            segmentation_method=segmentation_method,
            segmentation_kwargs=seg_kwargs,
            return_metadata = True,
            postprocess=bool(parameterNode.postprocessEnabled),
            postprocess_kwargs=post_kwargs,
            return_intermediates=False,
            compute_features=bool(parameterNode.radiomicsEnabled),
            voxel_spacing=voxel_spacing_hwd,
        )

        segmentation_metadata = None
        if isinstance(pipeline_result, dict):
            mask_HWD = pipeline_result.get("mask")
            features = pipeline_result.get("features") if parameterNode.radiomicsEnabled else None
            segmentation_metadata = pipeline_result.get("segmentation_metadata")
        else:
            mask_HWD = pipeline_result
            features = None

        if mask_HWD.shape != volume_HWD.shape:
            raise RuntimeError(
                f"PulmoVision pipeline returned mask with shape {mask_HWD.shape}, "
                f"expected {volume_HWD.shape}"
            )

        mask_DHW = np.transpose(mask_HWD.astype(np.float32), (2, 0, 1))

        slicer.util.updateVolumeFromArray(outputMaskVolume, mask_DHW)
        self._copyVolumeGeometry(inputVolume, outputMaskVolume)

        if showResult:
            slicer.util.setSliceViewerLayers(
                background=inputVolume,
                foreground=outputMaskVolume,
                foregroundOpacity=0.5,
            )

        feature_results = None
        if parameterNode.outputFeatureTable and parameterNode.radiomicsEnabled:
            feature_results = self._updateFeatureTable(
                parameterNode.outputFeatureTable, features or {}
            )

        self._displaySegmentationWarnings(segmentation_metadata)

        stopTime = time.time()
        logging.info(f"PulmoVision: processing completed in {stopTime - startTime:.2f} seconds")

        return {
            "mask": mask_DHW,
            "features": feature_results,
            "segmentation_metadata": segmentation_metadata,
        }

    def _displaySegmentationWarnings(self, segmentation_metadata):
        if not segmentation_metadata:
            return

        requested = segmentation_metadata.get("requested_method")
        used = segmentation_metadata.get("used_method") or requested
        messages = list(segmentation_metadata.get("messages") or [])

        if requested and used and requested != used:
            messages.append(
                _(
                    f"Segmentation method '{requested}' fell back to '{used}'. Checkpoints may be missing or unusable."
                )
            )

        if messages:
            slicer.util.warningDisplay(
                "\n".join(dict.fromkeys(messages)),
                windowTitle=_("PulmoVision segmentation warning"),
            )

    @staticmethod
    def _copyVolumeGeometry(sourceVolume, targetVolume):
        """Copy geometry (origin, spacing, direction) from source to target volume node."""
        ijkToRas = vtk.vtkMatrix4x4()
        sourceVolume.GetIJKToRASMatrix(ijkToRas)
        targetVolume.SetIJKToRASMatrix(ijkToRas)

        targetVolume.SetSpacing(sourceVolume.GetSpacing())
        targetVolume.SetOrigin(sourceVolume.GetOrigin())

    def _updateFeatureTable(self, tableNode, features):
        """Populate a MRML table node with radiomics-style summaries."""

        features = features or {}

        table = tableNode.GetTable()
        table.Initialize()

        nameColumn = vtk.vtkStringArray()
        nameColumn.SetName("Feature")
        valueColumn = vtk.vtkDoubleArray()
        valueColumn.SetName("Value")
        table.AddColumn(nameColumn)
        table.AddColumn(valueColumn)

        for feature_name, value in features.items():
            row_idx = table.InsertNextBlankRow()
            table.SetValue(row_idx, 0, vtk.vtkVariant(feature_name))
            table.SetValue(row_idx, 1, vtk.vtkVariant(float(value)))

        tableNode.Modified()
        return features

#
# PulmoVisionTest
#


class PulmoVisionTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_PulmoVision1()

    def test_PulmoVision1(self):
        """
        End-to-end test for PulmoVision.

        Loads sample data, runs the PulmoVisionLogic.process(), and verifies:
        - Output volume geometry matches input.
        - Output contains a binary mask (values in {0, 1}).
        - There is at least one foreground voxel.
        - Radiomics table is populated with summary metrics
        """

        self.delayDisplay("Starting PulmoVision1 test")

        import SampleData
        import numpy as np

        registerSampleData()
        inputVolume = SampleData.downloadSample("PulmoVision1")
        self.delayDisplay("Loaded PulmoVision1 test data set")

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        outputVolume.SetName("PulmoVisionTestOutput")
        featureTable = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
        featureTable.SetName("PulmoVisionTestFeatures")

        logic = PulmoVisionLogic()
        parameterNode = logic.getParameterNode()
        parameterNode.inputVolume = inputVolume
        parameterNode.outputMaskVolume = outputVolume
        parameterNode.outputFeatureTable = featureTable
        parameterNode.segmentationMethod = "auto"
        parameterNode.segmentationPercentile = 99.0
        parameterNode.postprocessEnabled = True

        results = logic.process(parameterNode=parameterNode, showResult=False)

        inputArray = slicer.util.arrayFromVolume(inputVolume)
        outputArray = slicer.util.arrayFromVolume(outputVolume)

        self.assertEqual(
            inputArray.shape,
            outputArray.shape,
            msg=f"Output shape {outputArray.shape} does not match input shape {inputArray.shape}",
        )

        unique_vals = np.unique(outputArray)
        self.assertTrue(
            np.all(np.isin(unique_vals, [0.0, 1.0])),
            msg=f"Output volume has non-binary values: {unique_vals}",
        )

        self.assertGreater(
            np.count_nonzero(outputArray),
            0,
            msg="Output mask is entirely empty (no foreground voxels).",
        )

        # Radiomics table should contain feature rows and match returned dict
        table = featureTable.GetTable()
        self.assertGreater(table.GetNumberOfRows(), 0, msg="Feature table is empty")
        returned_features = results["features"] or {}
        for feature_name in ["Voxels", "Volume (mm^3)", "Mean HU"]:
            self.assertIn(feature_name, returned_features)

        self.delayDisplay("PulmoVision1 test passed")