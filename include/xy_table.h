#ifndef INCLUDE_XY_TABLE_H_
#define INCLUDE_XY_TABLE_H_

#include <opencv2/core.hpp>
#include <k4a/k4a.h>
#include "G4RotationMatrix.hh"
#include "G4AffineTransform.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"

#include <math.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

void create_xy_table(int height, int width) {
	ofstream ofs("xytable.txt");
	k4a::device device;
	k4a::transformation main_depth_to_main_color;
	k4a_device_configuration_t deviceConfig = get_default_config();
	deviceConfig.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
	deviceConfig.camera_fps = K4A_FRAMES_PER_SECOND_15;
	deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_720P;
	deviceConfig.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;// no need to have a master cable if it's standalone

	device = k4a::device::open(0);

	// Get calibration information

	k4a::calibration main_calibration = device.get_calibration(deviceConfig.depth_mode,deviceConfig.color_resolution);
	main_depth_to_main_color = k4a::transformation(main_calibration);

	// xy table
	k4a_float2_t p;
	k4a_float3_t ray;
	cv::Mat xy_table=cv::Mat::zeros(height, width, CV_32FC2);
	float* xy_data = (float*)xy_table.data;

	//uchar
	for (int y = 0, idx = 0; y < height; y++)
	{
		p.xy.y = (float)y;
		for (int x = 0; x < width; x++, idx++)
		{
			p.xy.x = (float)x;

			if(main_calibration.convert_2d_to_3d(p,1.f,K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR, &ray))
			{
				xy_data[idx*2] = ray.xyz.x;
				xy_data[idx*2+1] = ray.xyz.y;
				ofs << xy_data[idx*2] << " " << xy_data[idx*2+1] << endl;
			}
			else {
				xy_data[idx*2] = nanf("");
				xy_data[idx*2+1] = nanf("");
				ofs << xy_data[idx*2] << " " << xy_data[idx*2+1] << endl;
			}

		}
	}
}

vector<G4ThreeVector> transform_xyz(vector<G4ThreeVector>& pcdVec)
{
	G4RotationMatrix rotM = G4RotationMatrix();
	G4ThreeVector pos = G4ThreeVector();
	G4AffineTransform affT = G4AffineTransform(rotM, pos);

	cout << pos << endl;
	cout << rotM << endl;
	cout << affT << endl;

	rotM.rotateX(180*deg);
	cout << rotM << endl;

	for (auto &itr:pcdVec) {
		itr = rotM * itr;
	}

	return pcdVec;
}




#endif
