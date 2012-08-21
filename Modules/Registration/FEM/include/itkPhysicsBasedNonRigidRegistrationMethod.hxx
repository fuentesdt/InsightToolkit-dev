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
#ifndef __itkPhysicsBasedNonRigidRegistrationMethod_hxx
#define __itkPhysicsBasedNonRigidRegistrationMethod_hxx

#include <iostream>
#include "itkTimeProbe.h"
#include "itkPhysicsBasedNonRigidRegistrationMethod.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkImageFileWriter.h"

namespace itk
{

namespace fem
{

template <class TFixedImage, class TMovingImage, class TMaskImage, class TMesh, class TDeformationField>
PhysicsBasedNonRigidRegistrationMethod<TFixedImage, TMovingImage, TMaskImage, TMesh, TDeformationField>
::PhysicsBasedNonRigidRegistrationMethod()
{
  // defaults
  this->m_NonConnectivity = 0; // VERTEX_CONNECTIVITY
  this->m_SelectFraction = 0.1;
  this->m_BlockRadius.Fill( 2 );
  this->m_SearchRadius.Fill( 5 );
  this->m_ApproximationSteps = 10;
  this->m_OutlierRejectionSteps = 10;

  // setup internal pipeline
  this->m_FeatureSelectionFilter = FeatureSelectionFilterType::New();
  this->m_FeatureSelectionFilter->ComputeStructureTensorsOn();
  this->m_BlockMatchingFilter = BlockMatchingFilterType::New();
  this->m_BlockMatchingFilter->SetFeaturePoints( this->m_FeatureSelectionFilter->GetOutput() );
  this->m_FEMFilter = FEMFilterType::New();
  this->m_FEMFilter->SetConfidencePointSet( this->m_BlockMatchingFilter->GetSimilarities() );
  this->m_FEMFilter->SetTensorPointSet( this->m_FeatureSelectionFilter->GetOutput() );

  // all inputs are required
  this->SetPrimaryInputName("FixedImage");
  this->AddRequiredInputName("FixedImage");
  this->AddRequiredInputName("MovingImage");
  this->AddRequiredInputName("MaskImage");
  this->AddRequiredInputName("Mesh");
}


template <class TFixedImage, class TMovingImage, class TMaskImage, class TMesh, class TDeformationField>
PhysicsBasedNonRigidRegistrationMethod<TFixedImage, TMovingImage, TMaskImage, TMesh, TDeformationField>
::~PhysicsBasedNonRigidRegistrationMethod()
{
}


template <class TFixedImage, class TMovingImage, class TMaskImage, class TMesh, class TDeformationField>
void
PhysicsBasedNonRigidRegistrationMethod<TFixedImage, TMovingImage, TMaskImage, TMesh, TDeformationField>
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "m_BlockRadius: " << m_BlockRadius << std::endl
     << indent << "m_SearchRadius: " << m_SearchRadius << std::endl
     << indent << "m_SelectFraction: " << m_SelectFraction << std::endl
     << indent << "m_NonConnectivity: " << m_NonConnectivity << std::endl
     << indent << "m_ApproximationSteps: " << m_ApproximationSteps << std::endl
     << indent << "m_OutlierRejectionSteps: " << m_OutlierRejectionSteps << std::endl;
}

template <class TFixedImage, class TMovingImage, class TMaskImage, class TMesh, class TDeformationField>
void
PhysicsBasedNonRigidRegistrationMethod<TFixedImage, TMovingImage, TMaskImage, TMesh, TDeformationField>
::GenerateData()
{
  // feature selection
  this->m_FeatureSelectionFilter->SetInput( this->GetMovingImage() );
  this->m_FeatureSelectionFilter->SetMaskImage( this->GetMaskImage() );
  this->m_FeatureSelectionFilter->SetSelectFraction( this->m_SelectFraction );
  this->m_FeatureSelectionFilter->SetNonConnectivity( this->m_NonConnectivity );
  this->m_FeatureSelectionFilter->SetBlockRadius( this->m_BlockRadius );

  // block matching
  this->m_BlockMatchingFilter->SetFixedImage( this->GetFixedImage() );
  this->m_BlockMatchingFilter->SetMovingImage( this->GetMovingImage() );
  this->m_BlockMatchingFilter->SetBlockRadius( this->m_BlockRadius );
  this->m_BlockMatchingFilter->SetSearchRadius( this->m_SearchRadius );

  // assembly and solver
  typename BlockMatchingFilterType::DisplacementsType * displacements = this->m_BlockMatchingFilter->GetDisplacements();
  this->m_FEMFilter->SetInput( displacements );
  this->m_FEMFilter->SetMesh( const_cast< MeshType * >( this->GetMesh() ) );
  const FixedImageType * fixedImage = this->GetFixedImage();
  this->m_FEMFilter->SetSpacing( fixedImage->GetSpacing() );
  this->m_FEMFilter->SetOrigin( fixedImage->GetOrigin() );
  this->m_FEMFilter->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );

  typename FEMFilterType::FEMSolverType * femSolver = this->m_FEMFilter->GetFEMSolver();
  femSolver->SetApproximationSteps( this->m_ApproximationSteps );
  femSolver->SetOutlierRejectionSteps( this->m_OutlierRejectionSteps );

  // graft our output to the filter to force the proper regions to be generated
  this->m_FEMFilter->GraftOutput( this->GetOutput() );

  this->m_FEMFilter->Update();

  // graft the output of the subtract filter back onto this filter's output
  // this is needed to get the appropriate regions passed back
  this->GraftOutput( this->m_FEMFilter->GetOutput() );
}

template <class TFixedImage, class TMovingImage, class TMaskImage, class TMesh, class TDeformationField>
void
PhysicsBasedNonRigidRegistrationMethod<TFixedImage, TMovingImage, TMaskImage, TMesh, TDeformationField>
::CreateDeformedImage(typename MovingImageType::Pointer& pDeformedImage)
{
  if(this->GetMovingImage() == NULL)
    {
    itkExceptionMacro("pMoving Image is NULL!");
    }
    if(this->m_FEMFilter.IsNull())
    {
    itkExceptionMacro("pFilter is NULL!");
    }
  std::cout << "Creating Deformed Moving Image... " << std::endl;

  itk::fem::Element::VectorType vMin,vMax,VoxelDispl,nodeDispl,vGlobal,vLocal,shapeF,dVec,pos;
  itk::fem::Element::Pointer elem;
  typename MovingImageType::RegionType region;
  typename MovingImageType::SizeType   regionSize;
  typename MovingImageType::PixelType  PixelValue;
  std::vector<itk::fem::Element::VectorType> nodeDisplVector;
  unsigned int numDofs = 0;
  itk::fem::Element::DegreeOfFreedomIDType id;
  typename MovingImageType::PointType global_pt;
  typename MovingImageType::PointType point;

  // Allocate space for deformed image
  pDeformedImage = MovingImageType::New();
  // Set region at deformed image
  region.SetIndex(this->GetMovingImage()->GetLargestPossibleRegion().GetIndex());
  region.SetSize(this->GetMovingImage()->GetLargestPossibleRegion().GetSize());
  pDeformedImage->SetRegions(region);
  // Set origini at deformed image
  pDeformedImage->SetOrigin(this->GetMovingImage()->GetOrigin());
  // Set spacing at deformed image
  pDeformedImage->SetSpacing(this->GetMovingImage()->GetSpacing());
  // Set Direction!
  pDeformedImage->SetDirection(this->GetMovingImage()->GetDirection());
  // Allocate the image memory
  pDeformedImage->Allocate();
  // fill the buffer with zeros
  pDeformedImage->FillBuffer(0);
  pDeformedImage->Update();

  // Set the interpolator as linear
  typedef itk::LinearInterpolateImageFunction<MovingImageType, double> InterpolatorType;
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
  interpolator->SetInputImage(this->GetMovingImage());
  unsigned int numberOfElements = this->m_FEMFilter->GetFEMSolver()->GetOutput()->GetElementContainer()->Size();

  // Loop through all the cells of mesh
  typename itk::fem::FEMObject<ImageDimension>::ElementContainerType::Iterator it;
  for(it = this->m_FEMFilter->GetFEMSolver()->GetOutput()->GetElementContainer()->Begin(); it != this->m_FEMFilter->GetFEMSolver()->GetOutput()->GetElementContainer()->End(); ++it)
    {
    elem = it.Value();
    // Initialize the bounding box
    vMin = elem->GetNodeCoordinates(0); // get the deformed coordinates of the mesh.
    vMax = vMin;
    const unsigned int NumberOfDimensions = elem->GetNumberOfSpatialDimensions();
    // Find the bounding box of the cell
    for( unsigned int i = 1; i < elem->GetNumberOfNodes(); i++ )
      {
      const itk::fem::Element::VectorType& v = elem->GetNodeCoordinates(i);
      for( unsigned int j = 0; j < NumberOfDimensions; j++ )
        {
        if( v[j] < vMin[j] )
          {
          vMin[j] = v[j];
          }
        if( v[j] > vMax[j] )
          {
          vMax[j] = v[j];
          }
        }
      }

    numDofs = elem->GetNumberOfDegreesOfFreedomPerNode();
    nodeDispl.set_size(numDofs);
    nodeDisplVector.clear();

    // Take the displacements of each node
    for( unsigned int i = 0; i < elem->GetNumberOfNodes(); i++ )
      {
      nodeDispl.fill(0);
      for(int j=0; j<numDofs; j++)
        {
        id = elem->GetNode(i)->GetDegreeOfFreedom(j);
        nodeDispl[j] = this->m_FEMFilter->GetFEMSolver()->GetSolution(id,0);
        }
      nodeDisplVector.push_back(nodeDispl);
      }
    typename MovingImageType::PointType minPoint,maxPoint;
    for( unsigned int j = 0; j < ImageDimension; j++ )
      {
      minPoint[j] = vMin[j];
      maxPoint[j] = vMax[j];
      }

    // Check if the two corners of the bounding box is inside the moving image.
    typename MovingImageType::IndexType vMinIndex,vMaxIndex,vDirectionIndex;
    if(this->GetMovingImage()->TransformPhysicalPointToIndex(minPoint,vMinIndex) == false)
      {
      continue;
      }
    if(this->GetMovingImage()->TransformPhysicalPointToIndex(maxPoint,vMaxIndex) == false)
      {
      continue;
      }

    // Set the region size of the bounding box
    int indexCheck;
    for( unsigned int i = 0; i < NumberOfDimensions; i++ )
      {
      if(this->GetMovingImage()->GetDirection()(i,i) == 1)
        {
        indexCheck = vMaxIndex[i] - vMinIndex[i] + 1;
        }
      else if(this->GetMovingImage()->GetDirection()(i,i) == -1)
        {
        indexCheck = vMinIndex[i] - vMaxIndex[i] + 1;
        }
      else
        {
        itkExceptionMacro("Unsupported direction value for image!");
        }
      if(indexCheck <= 0)
        {
        itkExceptionMacro("Invalide region size!");
        }
      regionSize[i] = indexCheck;
      }
    // Find the appropriate start index for iteration.
    for( unsigned int i = 0; i < NumberOfDimensions; i++ )
      {
      if(this->GetMovingImage()->GetDirection()(i,i) == 1)
        {
        vDirectionIndex[i] = vMinIndex[i];
        }
      else if(this->GetMovingImage()->GetDirection()(i,i) == -1)
        {
        vDirectionIndex[i] = vMaxIndex[i];
        }
      else
        {
        itkExceptionMacro("Unsupported direction value for image!");
        }
      }
    region.SetSize(regionSize);
    region.SetIndex(vDirectionIndex);

    itk::ImageRegionIterator<MovingImageType> iter((MovingImageType*)this->GetMovingImage(),region);
    typename MovingImageType::IndexType index;
    vGlobal.set_size(numDofs);
    dVec.set_size(numDofs);
    dVec.fill(0);

    // Step over all voxels of the region size
    for( iter.GoToBegin(); !iter.IsAtEnd(); ++iter )
      {
      index = iter.GetIndex();
      global_pt.Fill(0);
      // Get the physical coordinates of the voxel
      this->GetMovingImage()->TransformIndexToPhysicalPoint(index,global_pt);
      for( unsigned int j = 0; j < ImageDimension; j++ )
        {
        vGlobal[j] = global_pt[j];
        }
      // Check if the point is within the element...
      if(elem->GetLocalFromGlobalCoordinates(vGlobal,vLocal))
        {
        // get the shape functions
        shapeF.fill(0);
        shapeF = elem->ShapeFunctions(vLocal);
        dVec.fill(0);

        // u = u1*L1 + u2*L2 + u3*L3 + u4*L4
        for(int k=0; k<elem->GetNumberOfNodes(); k++)
          {
          dVec = dVec + (shapeF[k] * nodeDisplVector.at(k));
          }

        pos = vGlobal - dVec;

        for(int k=0; k<ImageDimension; k++)
          {
          point[k] = pos[k];
          }
        // Point inside the image buffer
        if( interpolator->IsInsideBuffer(point) )
          {
          // Get the pixel value with interpolation
          PixelValue = interpolator->Evaluate(point);
          }
        // Set the pixel value to deformed image
        pDeformedImage->SetPixel(index, PixelValue);
      }
     }
    }
}

}
}  // end namespace itk::fem

#endif
