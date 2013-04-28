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
#ifndef __itkGPUBlockMatchingImageFilter_h
#define __itkGPUBlockMatchingImageFilter_h

#include "itkBlockMatchingImageFilter.h"
#include "itkGPUBoxImageFilter.h"
#include "itkVersion.h"
#include "itkObjectFactoryBase.h"
#include "itkOpenCLUtil.h"

#include "itkMeanImageFilter.h"

#include "itkConceptChecking.h"


namespace itk
{
/** \class GPUBlockMatchingImageFilter
 *
 * \brief GPU-enabled implementation of BlockMatchingImageFilter.
 *
 * ...
 *
 * \ingroup ...
 */

/** Create a helper GPU Kernel class for GPUBlockMatchingImageFilter */
itkGPUKernelClassMacro(GPUBlockMatchingImageFilterKernel);


template<
  class TFixedImage,
  class TMovingImage = TFixedImage,
  class TFeatures = PointSet< Matrix< double, TFixedImage::ImageDimension, TFixedImage::ImageDimension>, TFixedImage::ImageDimension >,
  class TDisplacements = PointSet< Vector< typename TFeatures::PointType::ValueType, TFeatures::PointDimension >, TFeatures::PointDimension >,
  class TSimilarities = PointSet< double, TDisplacements::PointDimension > >
class ITK_EXPORT GPUBlockMatchingImageFilter
  : public BlockMatchingImageFilter< TFixedImage, TMovingImage, TFeatures, TDisplacements, TSimilarities >
{
public:
  /** Standard class typedefs. */
  typedef GPUBlockMatchingImageFilter                                                                      Self;
  typedef BlockMatchingImageFilter< TFixedImage, TMovingImage, TFeatures, TDisplacements, TSimilarities >  Superclass;
  typedef SmartPointer< Self >                                                                             Pointer;
  typedef SmartPointer< const Self >                                                                       ConstPointer;

  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUBlockMatchingImageFilter, Superclass);

  /** Get OpenCL Kernel source as a string, creates a GetOpenCLSource method */
  itkGetOpenCLSourceFromKernelMacro(GPUBlockMatchingImageFilterKernel);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro( FixedImageDimensionShouldBe3,
                   ( Concept::SameDimension< TFixedImage::ImageDimension, 3u > ) );
  itkConceptMacro( MovingImageDimensionShouldBe3,
                   ( Concept::SameDimension< TMovingImage::ImageDimension, 3u > ) );
  itkConceptMacro( PointDimensionShouldBe3,
                   ( Concept::SameDimension< TFeatures::PointType::PointDimension, 3u > ) );
  /** End concept checking */
#endif

  // macro to set if GPU is used
  itkSetMacro(GPUEnabled, bool);
  itkGetConstMacro(GPUEnabled, bool);
  itkBooleanMacro(GPUEnabled);

  virtual void GenerateData();

protected:
  GPUBlockMatchingImageFilter();
  ~GPUBlockMatchingImageFilter();

  virtual void PrintSelf(std::ostream & os, Indent indent) const;

  virtual void GPUGenerateData();

  typename GPUKernelManager::Pointer m_GPUKernelManager;

private:
  GPUBlockMatchingImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &);     //purposely not implemented

  int m_GPUKernelHandle;
  bool m_GPUEnabled;
};

} // end namespace itk

#if ITK_TEMPLATE_TXX
#include "itkGPUBlockMatchingImageFilter.hxx"
#endif

#endif
