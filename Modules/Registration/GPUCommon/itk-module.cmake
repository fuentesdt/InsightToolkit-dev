set(DOCUMENTATION "This module contains some common components to support GPU-based
registrations")

itk_module(ITKGPURegistrationCommon
  DEPENDS
    ITKCommon
    ITKRegistrationCommon
    ITKGPUCommon
    ITKGPUFiniteDifference
    ITKGPUImageFilterBase
    ITKSmoothing
  TEST_DEPENDS
    ITKTestKernel
  DESCRIPTION
    "${DOCUMENTATION}"
)
