if (ITK_USE_GPU)
  set(ITKGPURegistrationCommon_SRC
    )

  set(ITKGPURegistrationCommon_Kernels
    GPUBlockMatchingImageFilter.cl
    )

  write_gpu_kernels("${ITKGPURegistrationCommon_Kernels}" ITKGPURegistrationCommon_SRC)

  add_library(ITKGPURegistrationCommon ${ITKGPURegistrationCommon_SRC})
  target_link_libraries(ITKGPURegistrationCommon
  ${ITKGPURegistrationCommon_LIBRARIES} ${ITKGPUCommon_LIBRARIES} ${ITKCommon_LIBRARIES} ${OPENCL_LIBRARIES} )
  itk_module_target(ITKGPURegistrationCommon)
endif()
