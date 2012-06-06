if (ITK_USE_GPU)

# With MS compilers on Win64, we need the /bigobj switch, else generated
# code results in objects with number of sections exceeding object file
# format.
# see http://msdn.microsoft.com/en-us/library/ms173499.aspx
if(CMAKE_CL_64 OR CMAKE_COMPILER_2005)
add_definitions(/bigobj)
endif()

itk_module_test()

set(ITKGPURegistrationCommon-tests
 itkGPUBlockMatchingImageFilterTest.cxx
)

CreateTestDriver(ITKGPURegistrationCommon "${ITKGPURegistrationCommon-Test_LIBRARIES}" "${ITKGPURegistrationCommon-tests}")

itk_add_test(NAME itkGPUBlockMatchingImageFilterTest3D
        COMMAND ITKGPURegistrationCommonTestDriver itkGPUBlockMatchingImageFilterTest DATA{${ITK_DATA_ROOT}/Input/HeadMRVolume.mha}
        ${ITK_TEST_OUTPUT_DIR}/itkBlockMatchingImageFilterTest.mha)

endif(ITK_USE_GPU)
