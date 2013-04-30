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
#ifndef __itkGPUBlockMatchingImageFilter_hxx
#define __itkGPUBlockMatchingImageFilter_hxx

#include "itkGPUBlockMatchingImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkCastImageFilter.h"
#include "itkTimeProbe.h"


namespace itk
{

template< class TFixedImage, class TMovingImage, class TFeatures, class TDisplacements, class TSimilarities >
GPUBlockMatchingImageFilter< TFixedImage, TMovingImage, TFeatures, TDisplacements, TSimilarities >
::GPUBlockMatchingImageFilter()
{
  m_GPUEnabled = true;
  m_GPUKernelManager = GPUKernelManager::New();

  std::ostringstream defines;

  defines << "#define FIXEDTYPE ";
  GetTypenameInString( typeid ( typename TFixedImage::PixelType ), defines );
  std::cout << "Defines: " << defines.str() << std::endl;

  defines << "#define MOVINGTYPE ";
  GetTypenameInString( typeid ( typename TMovingImage::PixelType ), defines );
  std::cout << "Defines: " << defines.str() << std::endl;

  defines << "#define SIMTYPE ";
  GetTypenameInString( typeid ( typename Superclass::SimilaritiesValue ), defines );
  std::cout << "Defines: " << defines.str() << std::endl;

  const char* GPUSource = GPUBlockMatchingImageFilter::GetOpenCLSource();

  // load and build program
  this->m_GPUKernelManager->LoadProgramFromString( GPUSource, defines.str().c_str() );

  // create kernel
  m_GPUKernelHandle = this->m_GPUKernelManager->CreateKernel("BlockMatchingFilter");
}

template< class TFixedImage, class TMovingImage, class TFeatures, class TDisplacements, class TSimilarities >
GPUBlockMatchingImageFilter< TFixedImage, TMovingImage, TFeatures, TDisplacements, TSimilarities >
::~GPUBlockMatchingImageFilter()
{

}

template< class TFixedImage, class TMovingImage, class TFeatures, class TDisplacements, class TSimilarities >
void
GPUBlockMatchingImageFilter< TFixedImage, TMovingImage, TFeatures, TDisplacements, TSimilarities >
::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "GPU: " << ( m_GPUEnabled ? "Enabled" : "Disabled" ) << std::endl;
}


template< class TFixedImage, class TMovingImage, class TFeatures, class TDisplacements, class TSimilarities >
void
GPUBlockMatchingImageFilter< TFixedImage, TMovingImage, TFeatures, TDisplacements, TSimilarities >
::GenerateData()
{
if( !m_GPUEnabled ) // call CPU update function
  {
  Superclass::GenerateData();
  }
else // call GPU update function
  {
  // Call a method to allocate memory for the filter's outputs
//  this->AllocateOutputs();

  GPUGenerateData();
  }
}


template< class TFixedImage, class TMovingImage, class TFeatures, class TDisplacements, class TSimilarities >
void
GPUBlockMatchingImageFilter< TFixedImage, TMovingImage, TFeatures, TDisplacements, TSimilarities >
::GPUGenerateData()
{
TimeProbe clock;
clock.Start();

  typename Superclass::FixedImageConstPointer fixedImage = this->GetFixedImage();
  typename Superclass::MovingImageConstPointer movingImage = this->GetMovingImage();
  typename Superclass::ImageSizeType imageSize = fixedImage->GetLargestPossibleRegion().GetSize();

  // 1D images for passing feature points and retrieving displacements
  typedef GPUImage< int, 1 >                                     IndexImage;
  typedef GPUImage< typename Superclass::SimilaritiesValue, 1 >  SimilaritiesImage;

  typename Superclass::FeaturePointsConstPointer featurePoints = this->GetFeaturePoints();
  int numberOfPoints = featurePoints->GetNumberOfPoints();

  IndexImage::IndexType start;
  start[0] = 0;

  IndexImage::SizeType size;
  size[0] = numberOfPoints;

  IndexImage::RegionType region;
  region.SetIndex( start );
  region.SetSize( size );

  IndexImage::Pointer featuresX = IndexImage::New();
  featuresX->SetRegions( region );
  featuresX->Allocate();

  IndexImage::Pointer featuresY = IndexImage::New();
  featuresY->SetRegions( region );
  featuresY->Allocate();

  IndexImage::Pointer featuresZ = IndexImage::New();
  featuresZ->SetRegions( region );
  featuresZ->Allocate();

  IndexImage::Pointer displacementsX = IndexImage::New();
  displacementsX->SetRegions( region );
  displacementsX->Allocate();

  IndexImage::Pointer displacementsY = IndexImage::New();
  displacementsY->SetRegions( region );
  displacementsY->Allocate();

  IndexImage::Pointer displacementsZ = IndexImage::New();
  displacementsZ->SetRegions( region );
  displacementsZ->Allocate();

  typename SimilaritiesImage::Pointer similarities = SimilaritiesImage::New();
  similarities->SetRegions( region );
  similarities->Allocate();


  typedef typename Superclass::FeaturePointsType::PointsContainer::ConstIterator FeaturestIteratorType;
  FeaturestIteratorType pointItr = featurePoints->GetPoints()->Begin();
  FeaturestIteratorType pointEnd = featurePoints->GetPoints()->End();

  ImageRegionIterator<IndexImage> iterX( featuresX, region );
  ImageRegionIterator<IndexImage> iterY( featuresY, region );
  ImageRegionIterator<IndexImage> iterZ( featuresZ, region );

  // copy feature points index coordinates into 3 separate arrays
  while ( pointItr != pointEnd )
    {
    typename Superclass::ImageIndexType voxelIndex;
    fixedImage->TransformPhysicalPointToIndex( pointItr.Value(), voxelIndex );

    iterX.Set( voxelIndex[0] );
    iterY.Set( voxelIndex[1] );
    iterZ.Set( voxelIndex[2] );

//    std::cout << "X:" << iterX.Value()
//              << ", Y:" << iterY.Value()
//              << ", Z:" << iterZ.Value() << std::endl;

    pointItr++;
    ++iterX;
    ++iterY;
    ++iterZ;
    }

  // cast Image to GPUImage, dynamic_cast gives null
  typedef typename itk::GPUTraits< TFixedImage >::Type GPUFixedImage;
  typedef CastImageFilter< typename Superclass::FixedImageType, GPUFixedImage > FixedCasterType;
  typename FixedCasterType::Pointer fixedCaster = FixedCasterType::New();
  fixedCaster->SetInput( fixedImage );
  fixedCaster->Update();
  typename GPUFixedImage::Pointer fixedGPUImage = fixedCaster->GetOutput();

  typedef typename itk::GPUTraits< TMovingImage >::Type GPUMovingImage;
  typedef CastImageFilter< typename Superclass::MovingImageType, GPUMovingImage > MovingCasterType;
  typename MovingCasterType::Pointer movingCaster = MovingCasterType::New();
  movingCaster->SetInput( movingImage );
  movingCaster->Update();
  typename GPUFixedImage::Pointer movingGPUImage = movingCaster->GetOutput();

  // pass images to kernel
  int argidx = 0;

  this->m_GPUKernelManager->SetKernelArgWithImage(m_GPUKernelHandle, argidx++, fixedGPUImage->GetGPUDataManager() );
  this->m_GPUKernelManager->SetKernelArgWithImage(m_GPUKernelHandle, argidx++, movingGPUImage->GetGPUDataManager() );

  this->m_GPUKernelManager->SetKernelArgWithImage(m_GPUKernelHandle, argidx++, featuresX->GetGPUDataManager() );
  this->m_GPUKernelManager->SetKernelArgWithImage(m_GPUKernelHandle, argidx++, featuresY->GetGPUDataManager() );
  this->m_GPUKernelManager->SetKernelArgWithImage(m_GPUKernelHandle, argidx++, featuresZ->GetGPUDataManager() );

  this->m_GPUKernelManager->SetKernelArgWithImage(m_GPUKernelHandle, argidx++, displacementsX->GetGPUDataManager() );
  this->m_GPUKernelManager->SetKernelArgWithImage(m_GPUKernelHandle, argidx++, displacementsY->GetGPUDataManager() );
  this->m_GPUKernelManager->SetKernelArgWithImage(m_GPUKernelHandle, argidx++, displacementsZ->GetGPUDataManager() );

  this->m_GPUKernelManager->SetKernelArgWithImage(m_GPUKernelHandle, argidx++, similarities->GetGPUDataManager() );

  // numberOfPoints
  this->m_GPUKernelManager->SetKernelArg(m_GPUKernelHandle, argidx++, sizeof( int ), &( numberOfPoints ) );

  // imageSize
  int width = imageSize[0];
  int height = imageSize[1];
  int depth = imageSize[2];

  this->m_GPUKernelManager->SetKernelArg(m_GPUKernelHandle, argidx++, sizeof( int ), &( width ) );
  this->m_GPUKernelManager->SetKernelArg(m_GPUKernelHandle, argidx++, sizeof( int ), &( height ) );
  this->m_GPUKernelManager->SetKernelArg(m_GPUKernelHandle, argidx++, sizeof( int ), &( depth ) );


  // block radius
  int blockRadiusX = this->GetBlockRadius()[0];
  int blockRadiusY = this->GetBlockRadius()[1];
  int blockRadiusZ = this->GetBlockRadius()[2];

  this->m_GPUKernelManager->SetKernelArg(m_GPUKernelHandle, argidx++, sizeof( int ), &( blockRadiusX ) );
  this->m_GPUKernelManager->SetKernelArg(m_GPUKernelHandle, argidx++, sizeof( int ), &( blockRadiusY ) );
  this->m_GPUKernelManager->SetKernelArg(m_GPUKernelHandle, argidx++, sizeof( int ), &( blockRadiusZ ) );

  // search radius
  int searchRadiusX = this->GetSearchRadius()[0];
  int searchRadiusY = this->GetSearchRadius()[1];
  int searchRadiusZ = this->GetSearchRadius()[2];

  this->m_GPUKernelManager->SetKernelArg(m_GPUKernelHandle, argidx++, sizeof( int ), &( searchRadiusX ) );
  this->m_GPUKernelManager->SetKernelArg(m_GPUKernelHandle, argidx++, sizeof( int ), &( searchRadiusY ) );
  this->m_GPUKernelManager->SetKernelArg(m_GPUKernelHandle, argidx++, sizeof( int ), &( searchRadiusZ ) );

  size_t localSize[1];
  localSize[0] = OpenCLGetLocalBlockSize( 1 );
  size_t globalSize[1];
  globalSize[0] = ceil( static_cast<float>( numberOfPoints ) / localSize[0] ) * localSize[0];
std::cout << "numberOfPoints: " << numberOfPoints
          << ", globalSize: "   << globalSize[0]
          <<", localSize: " << localSize[0] << std::endl;

  // launch kernel
  this->m_GPUKernelManager->LaunchKernel( m_GPUKernelHandle, 1, globalSize, localSize );

  // copy displacements and similarities to the output
  displacementsX->UpdateBuffers();
  displacementsY->UpdateBuffers();
  displacementsZ->UpdateBuffers();
  similarities->UpdateBuffers();

  ImageRegionIterator<IndexImage> dispIterX( displacementsX, region );
  ImageRegionIterator<IndexImage> dispIterY( displacementsY, region );
  ImageRegionIterator<IndexImage> dispIterZ( displacementsZ, region );
  ImageRegionIterator<SimilaritiesImage> similaritiesIter( similarities, region );

  const typename Superclass::FeaturePointsType::PointsContainer *points = featurePoints->GetPoints();

  typename Superclass::DisplacementsPointer displacementsPointSet = Superclass::GetDisplacements();

  typedef typename Superclass::DisplacementsType::PointsContainerPointer  DisplacementsPointsContainerPointerType;
  typedef typename Superclass::DisplacementsType::PointsContainer         DisplacementsPointsContainerType;
  DisplacementsPointsContainerPointerType displacementsPoints = DisplacementsPointsContainerType::New();

  typedef typename Superclass::DisplacementsType::PointDataContainerPointer  DisplacementsPointDataContainerPointerType;
  typedef typename Superclass::DisplacementsType::PointDataContainer         DisplacementsPointDataContainerType;
  DisplacementsPointDataContainerPointerType displacementsData = DisplacementsPointDataContainerType::New();

  typename Superclass::SimilaritiesPointer similaritiesPointSet = Superclass::GetSimilarities();

  typedef typename Superclass::SimilaritiesType::PointsContainerPointer  SimilaritiesPointsContainerPointerType;
  typedef typename Superclass::SimilaritiesType::PointsContainer         SimilaritiesPointsContainerType;
  SimilaritiesPointsContainerPointerType similaritiesPoints = SimilaritiesPointsContainerType::New();

  typedef typename Superclass::SimilaritiesType::PointDataContainerPointer  SimilaritiesPointDataContainerPointerType;
  typedef typename Superclass::SimilaritiesType::PointDataContainer         SimilaritiesPointDataContainerType;
  SimilaritiesPointDataContainerPointerType similaritiesData = SimilaritiesPointDataContainerType::New();

  SizeValueType i = 0;
  while ( !similaritiesIter.IsAtEnd() )
    {
//      std::cout << "X:" << dispIterX.Value()
//                << ", Y:" << dispIterY.Value()
//                << ", Z:" << dispIterZ.Value()
//                << ", sim: " << similaritiesIter.Value() << std::endl;

    typename Superclass::FeaturePointsPhysicalCoordinates originalLocation = points->GetElement( i );
    typename Superclass::DisplacementsVector displacementVector;

    typename Superclass::ImageIndexType newLocationIdx;
    newLocationIdx[0] = dispIterX.Value();
    newLocationIdx[1] = dispIterY.Value();
    newLocationIdx[2] = dispIterZ.Value();

    typename Superclass::FeaturePointsPhysicalCoordinates newLocation;
    fixedImage->TransformIndexToPhysicalPoint( newLocationIdx, newLocation );
    displacementVector = newLocation - originalLocation;

    displacementsPoints->InsertElement( i, points->GetElement( i ) );
    similaritiesPoints->InsertElement( i, points->GetElement( i ) );
    displacementsData->InsertElement( i, displacementVector );
    similaritiesData->InsertElement( i, similaritiesIter.Value() );

    ++i;
    ++similaritiesIter;
    ++dispIterX;
    ++dispIterY;
    ++dispIterZ;
    }

    displacementsPointSet->SetPoints( displacementsPoints );
    displacementsPointSet->SetPointData( displacementsData );
    similaritiesPointSet->SetPoints( similaritiesPoints );
    similaritiesPointSet->SetPointData( similaritiesData );

clock.Stop();
std::cout << "GPUGenerateData: " << clock.GetTotal() << std::endl;

}

} // end namespace itk

#endif
