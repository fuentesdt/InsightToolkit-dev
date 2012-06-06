/*=========================================================================
*
*  Copyright Insight Software Consortium
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*         http://www.apache.org/licenses/LICENSE-2.0.txt
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*=========================================================================*/

/**
 * Test program for itkGPUBlockMatchingImageFilter class
 *
 * This program creates a GPU Mean filter test pipelining.
 */

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkBlockMatchingImageFilter.h"

#include "itkGPUImage.h"
#include "itkGPUKernelManager.h"
#include "itkGPUContextManager.h"
#include "itkGPUImageToImageFilter.h"
#include "itkGPUBlockMatchingImageFilter.h"

#include "itkRegionOfInterestImageFilter.h"
#include "itkMaskFeaturePointSelectionFilter.h"
#include "itkBlockMatchingImageFilter.h"
#include "itkTranslationTransform.h"
#include "itkResampleImageFilter.h"


#include "itkTimeProbe.h"


int itkGPUBlockMatchingImageFilterTest(int argc, char *argv[])
{
  if(!itk::IsGPUAvailable())
  {
    std::cerr << "OpenCL-enabled GPU is not present." << std::endl;
    return EXIT_FAILURE;
  }

  if( argc <  3 )
  {
    std::cerr << "Error: missing arguments" << std::endl;
    std::cerr << "fixed moving mask" << std::endl;
    return EXIT_FAILURE;
  }

  const double selectFraction = 0.01;

  typedef unsigned char  InputPixelType;

  static const unsigned int Dimension = 3;

  typedef itk::Image< InputPixelType,  Dimension >  InputImageType;

  // Parameters used for FS and BM
  typedef InputImageType::SizeType RadiusType;
  RadiusType blockRadius;
  blockRadius.Fill( 1 );

  RadiusType searchRadius;
  searchRadius.Fill( 5 );

  typedef itk::ImageFileReader< InputImageType >  ReaderType;

  //Set up the reader
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( argv[1] );
  try
    {
    reader->Update();
    }
  catch( itk::ExceptionObject & e )
    {
    std::cerr << "Error in reading the input image: " << e << std::endl;
    return EXIT_FAILURE;
    }

  // Reduce region of interest by SEARCH_RADIUS
  typedef itk::RegionOfInterestImageFilter< InputImageType, InputImageType >  RegionOfInterestFilterType;

  RegionOfInterestFilterType::Pointer regionOfInterestFilter = RegionOfInterestFilterType::New();

  regionOfInterestFilter->SetInput( reader->GetOutput() );

  RegionOfInterestFilterType::RegionType regionOfInterest = reader->GetOutput()->GetLargestPossibleRegion();

  RegionOfInterestFilterType::RegionType::IndexType regionOfInterestIndex = regionOfInterest.GetIndex();
  regionOfInterestIndex += searchRadius;
  regionOfInterest.SetIndex( regionOfInterestIndex );

  RegionOfInterestFilterType::RegionType::SizeType regionOfInterestSize = regionOfInterest.GetSize();
  regionOfInterestSize -= searchRadius + searchRadius;
  regionOfInterest.SetSize( regionOfInterestSize );

  regionOfInterestFilter->SetRegionOfInterest( regionOfInterest );
  regionOfInterestFilter->Update();

  typedef itk::MaskFeaturePointSelectionFilter< InputImageType >  FeatureSelectionFilterType;
  typedef FeatureSelectionFilterType::FeaturePointsType           PointSetType;

  typedef FeatureSelectionFilterType::PointType       PointType;
  typedef FeatureSelectionFilterType::InputImageType  ImageType;

  // Feature Selection
  FeatureSelectionFilterType::Pointer featureSelectionFilter = FeatureSelectionFilterType::New();

  featureSelectionFilter->SetInput( regionOfInterestFilter->GetOutput() );
  featureSelectionFilter->SetSelectFraction( selectFraction );
  featureSelectionFilter->SetBlockRadius( blockRadius );
  featureSelectionFilter->ComputeStructureTensorsOff();

  // Create transformed image from input to match with
  typedef itk::TranslationTransform< double, Dimension > TranslationTransformType;
  TranslationTransformType::Pointer transform = TranslationTransformType::New();
  TranslationTransformType::OutputVectorType translation;
  // move each pixel in input image 5 pixels along first(0) dimension
  translation[0] = 5.0;
  translation[1] = 0.0;
  translation[2] = 0.0;
  transform->Translate(translation);

  typedef itk::ResampleImageFilter< InputImageType, InputImageType > ResampleImageFilterType;
  ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();
  resampleFilter->SetTransform( transform.GetPointer() );
  resampleFilter->SetInput( reader->GetOutput() );
  resampleFilter->SetReferenceImage( reader->GetOutput() );
  resampleFilter->UseReferenceImageOn();

  typedef itk::GPUBlockMatchingImageFilter< InputImageType >  BlockMatchingFilterType;
  BlockMatchingFilterType::Pointer blockMatchingFilter = BlockMatchingFilterType::New();

  // inputs (all required)
  blockMatchingFilter->SetFixedImage( resampleFilter->GetOutput() );
  blockMatchingFilter->SetMovingImage( reader->GetOutput() );
  blockMatchingFilter->SetFeaturePoints( featureSelectionFilter->GetOutput() );

  // parameters (all optional)
  blockMatchingFilter->SetBlockRadius( blockRadius );
  blockMatchingFilter->SetSearchRadius( searchRadius );

  std::cout << "Block matching: " << blockMatchingFilter << std::endl;
  try
    {
    blockMatchingFilter->Update();
    }
  catch ( itk::ExceptionObject &err )
    {
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }

  // Exercise the following methods
  BlockMatchingFilterType::DisplacementsType * displacements = blockMatchingFilter->GetDisplacements();
  if( displacements == NULL )
    {
    std::cerr << "GetDisplacements() failed." << std::endl;
    return EXIT_FAILURE;
    }
  BlockMatchingFilterType::SimilaritiesType * similarities = blockMatchingFilter->GetSimilarities();
  if( similarities == NULL )
    {
    std::cerr << "GetSimilarities() failed." << std::endl;
    return EXIT_FAILURE;
    }

  std::cout << "DONE" << std::endl;

//  // test 1~8 threads for CPU
//  for(int nThreads = 1; nThreads <= 8; nThreads++)
//  {
//    typename MeanFilterType::Pointer CPUFilter = MeanFilterType::New();
//
//    itk::TimeProbe cputimer;
//    cputimer.Start();
//
//    CPUFilter->SetNumberOfThreads( nThreads );
//
//    CPUFilter->SetInput( reader->GetOutput() );
//    CPUFilter->SetRadius( indexRadius );
//    CPUFilter->Update();
//
//    cputimer.Stop();
//
//    std::cout << "CPU mean filter took " << cputimer.GetMeanTime() << " seconds with "
//              << CPUFilter->GetNumberOfThreads() << " threads.\n" << std::endl;
//
//    // -------
//
//    typename GPUMeanFilterType::Pointer GPUFilter = GPUMeanFilterType::New();


//    if( nThreads == 8 )
//    {
//      typename GPUMeanFilterType::Pointer GPUFilter = GPUMeanFilterType::New();
//
//      itk::TimeProbe gputimer;
//      gputimer.Start();
//
//      GPUFilter->SetInput( reader->GetOutput() );
//      GPUFilter->Update();
//      GPUFilter->GetOutput()->UpdateBuffers(); // synchronization point (GPU->CPU memcpy)
//
//      gputimer.Stop();
//      std::cout << "GPU mean filter took " << gputimer.GetMeanTime() << " seconds.\n" << std::endl;
//
//      // ---------------
//      // RMS Error check
//      // ---------------
//
//      double diff = 0;
//      unsigned int nPix = 0;
//      itk::ImageRegionIterator<OutputImageType> cit(CPUFilter->GetOutput(), CPUFilter->GetOutput()->GetLargestPossibleRegion());
//      itk::ImageRegionIterator<OutputImageType> git(GPUFilter->GetOutput(), GPUFilter->GetOutput()->GetLargestPossibleRegion());
//
//      for(cit.GoToBegin(), git.GoToBegin(); !cit.IsAtEnd(); ++cit, ++git)
//      {
//        //std::cout << "CPU : " << (double)(cit.Get()) << ", GPU : " << (double)(git.Get()) << std::endl;
//        double err = (double)(cit.Get()) - (double)(git.Get());
//        diff += err*err;
//        nPix++;
//      }
//      if (nPix > 0)
//      {
//        double RMSError = sqrt( diff / (double)nPix );
//        std::cout << "RMS Error : " << RMSError << std::endl;
//        double RMSThreshold = 0;
//        if (vnl_math_isnan(RMSError))
//        {
//          std::cout << "RMS Error is NaN! nPix: " << nPix << std::endl;
//          return EXIT_FAILURE;
//        }
//        if (RMSError > RMSThreshold)
//        {
//          std::cout << "RMS Error exceeds threshold (" << RMSThreshold << ")" << std::endl;
//          return EXIT_FAILURE;
//        }
//        writer->SetInput( GPUFilter->GetOutput() );
//        writer->Update();
//      }
//      else
//      {
//        std::cout << "No pixels in output!" << std::endl;
//        return EXIT_FAILURE;
//      }
//    }

  return EXIT_SUCCESS;
}
