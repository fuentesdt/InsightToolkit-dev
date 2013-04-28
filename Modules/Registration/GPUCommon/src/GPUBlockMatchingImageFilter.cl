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

__kernel void BlockMatchingFilter(const __global FIXEDTYPE* fixedImage,
                                  const __global MOVINGTYPE* movingImage,
                                  const __global int* featuresX,
		                          const __global int* featuresY,
		                          const __global int* featuresZ,
		                          __global int* displacementsX,
		                          __global int* displacementsY,
		                          __global int* displacementsZ,
		                          __global SIMTYPE* similarities,
                                          int numberOfPoints, int width, int height, int depth,
		                          int blockRadiusX, int blockRadiusY, int blockRadiusZ,
		                          int searchRadiusX, int searchRadiusY, int searchRadiusZ)
{
  int gix = get_global_id(0);

  if ( gix < numberOfPoints )
  {

  int newLocationX = 0;
  int newLocationY = 0;
  int newLocationZ = 0;

  // feature block in moving image
  int movingBlockFromX = featuresX[gix] - blockRadiusX;
  int movingBlockToX = featuresX[gix] + blockRadiusX;

  int movingBlockFromY = featuresY[gix] - blockRadiusY;
  int movingBlockToY = featuresY[gix] + blockRadiusY;

  int movingBlockFromZ = featuresZ[gix] - blockRadiusZ;
  int movingBlockToZ = featuresZ[gix] + blockRadiusZ;

  // search window in fixed image
  int windowFromX = featuresX[gix] - searchRadiusX;
  int windowToX = featuresX[gix] + searchRadiusX;

  int windowFromY = featuresY[gix] - searchRadiusY;
  int windowToY = featuresY[gix] + searchRadiusY;

  int windowFromZ = featuresZ[gix] - searchRadiusZ;
  int windowToZ = featuresZ[gix] + searchRadiusZ;

  SIMTYPE numberOfVoxelInBlock = ( blockRadiusX * 2 + 1 )
                               * ( blockRadiusY * 2 + 1 )
                               * ( blockRadiusZ * 2 + 1 );
  SIMTYPE similarity = 0;

  // loop over window
  for ( int windowZ = windowFromZ; windowZ <= windowToZ; windowZ++ )
  {
    for ( int windowY = windowFromY; windowY <= windowToY; windowY++ )
    {
      for ( int windowX = windowFromX; windowX <= windowToX; windowX++ )
      {

        // new location block in fixed image
        int fixedBlockFromX = windowX - blockRadiusX;
        int fixedBlockToX = windowX + blockRadiusX;

        int fixedBlockFromY = windowY - blockRadiusY;
        int fixedBlockToY = windowY + blockRadiusY;

        int fixedBlockFromZ = windowZ - blockRadiusZ;
        int fixedBlockToZ = windowZ + blockRadiusZ;

        SIMTYPE fixedSum = 0.0;
        SIMTYPE fixedSumOfSquares = 0.0;
        SIMTYPE movingSum = 0.0;
        SIMTYPE movingSumOfSquares = 0.0;
        SIMTYPE covariance = 0.0;

        // loop over blocks
        for ( int movingBlockZ = movingBlockFromZ, fixedBlockZ = fixedBlockFromZ;
              movingBlockZ <= movingBlockToZ; movingBlockZ++, fixedBlockZ++ )
        {
          for ( int movingBlockY = movingBlockFromY, fixedBlockY = fixedBlockFromY;
                movingBlockY <= movingBlockToY; movingBlockY++, fixedBlockY++ )
          {
            for ( int movingBlockX = movingBlockFromX, fixedBlockX = fixedBlockFromX;
                  movingBlockX <= movingBlockToX; movingBlockX++, fixedBlockX++ )
            {
              // inside block loop
              int fixedIdx = fixedBlockX + width * ( fixedBlockY + height * fixedBlockZ );
              int movingIdx = movingBlockX + width * ( movingBlockY + height * movingBlockZ );

              SIMTYPE fixedValue = fixedImage[ fixedIdx ];
              SIMTYPE movingValue = movingImage[ movingIdx ];

              fixedSum += fixedValue;
              fixedSumOfSquares += fixedValue * fixedValue;

              movingSum += movingValue;
              movingSumOfSquares += movingValue * movingValue;

              covariance += fixedValue * movingValue;
            }
          }
        }

        SIMTYPE fixedMean = fixedSum / numberOfVoxelInBlock;
        SIMTYPE movingMean = movingSum / numberOfVoxelInBlock;
        SIMTYPE fixedVariance = fixedSumOfSquares - numberOfVoxelInBlock * fixedMean * fixedMean;
        SIMTYPE movingVariance = movingSumOfSquares - numberOfVoxelInBlock * movingMean * movingMean;
        covariance -= numberOfVoxelInBlock * fixedMean * movingMean;

        SIMTYPE sim = 0.0;
        if ( fixedVariance * movingVariance )
        {
          sim = ( covariance * covariance ) / ( fixedVariance * movingVariance );
        }

        if ( sim >= similarity )
        {
          newLocationX = windowX;
          newLocationY = windowY;
          newLocationZ = windowZ;
          similarity = sim;
        }

      }
    }
  }

  displacementsX[gix] = newLocationX;
  displacementsY[gix] = newLocationY;
  displacementsZ[gix] = newLocationZ;
  similarities[gix] = similarity;


  }

}
