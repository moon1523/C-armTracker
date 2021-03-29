#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc_c.h> // cvFindContours
#include <opencv2/imgproc.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/viz/widget_accessor.hpp>

#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPLYReader.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkTransform.h>
#include <vtkActor.h>
#include <vtkPLYWriter.h>

#include <iterator>
#include <set>
#include <cstdio>
#include <fstream>
#include <iostream>
#include "MultiDeviceCapturer.h"
#include "transformation.h"
#include "AzureKinect.h"
#include "labeldata.h"
#include "xy_table.h"
#include <k4a/k4a.hpp>

#include <termio.h>
#include <stdio.h>

/**
 * @class TriangleWidget
 * @brief Defining our own 3D Triangle widget
 */
using namespace cv;
using namespace std;

int frameIdx(0);
bool printPLY(false);
int count(0);

ofstream res("linemod_res.txt");



class WPoly : public viz::Widget3D
{
public:
    WPoly(){}
    WPoly(const string & fileName, const labeldata & label);
    vtkAlgorithmOutput* GetPolyDataPort() {return reader->GetOutputPort();}
    void Initialize(const string & fileName, const labeldata & label);
    void Transform(int class_id, int template_id);
    void Transform(int class_id, int template_id, double pos[3]);
private:
    labeldata label;
    vtkPLYReader* reader;
    vtkSmartPointer<vtkPolyDataMapper> mapper;
    vtkSmartPointer<vtkTransform> transform;
    vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter;
    vtkSmartPointer<vtkPLYWriter> plyWriter;
    vtkSmartPointer<vtkMatrix4x4> affineT;
};

/**
 * @function TriangleWidget::TriangleWidget
 * @brief Constructor
 */
WPoly::WPoly(const string & fileName, const labeldata & _label)
{
    Initialize(fileName, _label);
}
void WPoly::Initialize(const string &fileName, const labeldata &_label){
    label = _label;
    transform = vtkSmartPointer<vtkTransform>::New();
    transformFilter =
            vtkSmartPointer<vtkTransformPolyDataFilter>::New();

    vtkSmartPointer<vtkPolyData> polyData;
    reader = vtkPLYReader::New ();
    reader->SetFileName (fileName.c_str());
    reader->Update ();
    polyData = reader->GetOutput ();
    // Create mapper and actor
    mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(polyData);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    // Store this actor in the widget in order that visualizer can access it
    viz::WidgetAccessor::setProp(*this, actor);

    plyWriter = vtkSmartPointer<vtkPLYWriter>::New();
}

void WPoly::Transform(int class_id, int template_id){
    transform->SetMatrix(label.GetAffineTransformMatrix(class_id,template_id));
    // transform->RotateX(180);
    
    transformFilter->SetTransform(transform);
    transformFilter->SetInputConnection(reader->GetOutputPort());
    transformFilter->Update();
    mapper->SetInputData(transformFilter->GetOutput());

    if (printPLY) {
    	cout << "Print PLY  " << frameIdx++ << endl;
    	string path = "ply_set";
        // system(("mkdir " + path).c_str());
    	string fileName = "./" + path + "/frame" + to_string(frameIdx);
    	plyWriter->SetFileName(fileName.c_str());
    	plyWriter->SetInputData(transformFilter->GetOutput());
    	plyWriter->Write();
    }
}

void WPoly::Transform(int class_id, int template_id, double point[3]){
    cout << point[0] << " " << point[1] << " " << point[2] << endl;
    transform->SetMatrix(label.GetAffineTransformMatrix(class_id, template_id, point));
    // transform->RotateX(180);

    transformFilter->SetTransform(transform);
    transformFilter->SetInputConnection(reader->GetOutputPort());
    transformFilter->Update();
    mapper->SetInputData(transformFilter->GetOutput());

    if (printPLY) {
    	cout << "Print PLY  " << frameIdx++ << endl;
    	string path = "ply_set";
    	string fileName = "./" + path + "/frame" + to_string(frameIdx);
    	plyWriter->SetFileName(fileName.c_str());
    	plyWriter->SetInputData(transformFilter->GetOutput());
    	plyWriter->Write();
    }
}



void drawResponse(const std::vector<cv::linemod::Template>& templates,
                  int num_modalities, cv::Mat& dst, cv::Point offset, int T, 
                  cv::Mat depth, vector<pair<float,float>> xy_table);

cv::Mat displayQuantized(const cv::Mat& quantized);

// Copy of cv_mouse from cv_utilities
class Mouse
{
public:
    static void start(const std::string& a_img_name)
    {
        cv::setMouseCallback(a_img_name.c_str(), Mouse::cv_on_mouse, 0);
    }
    static int event(void)
    {
        int l_event = m_event;
        m_event = -1;
        return l_event;
    }
    static int x(void)
    {
        return m_x;
    }
    static int y(void)
    {
        return m_y;
    }

private:
    static void cv_on_mouse(int a_event, int a_x, int a_y, int, void*)
    {
        m_event = a_event;
        m_x = a_x;
        m_y = a_y;
    }

    static int m_event;
    static int m_x;
    static int m_y;
};
int Mouse::m_event;
int Mouse::m_x;
int Mouse::m_y;

static void help()
{
    printf("Usage: example_rgbd_linemod [templates.yml]\n\n"
           "Place your object on a planar, featureless surface. With the mouse,\n"
           "frame it in the 'color' window and right click to learn a first template.\n"
           "Then press 'l' to enter online learning mode, and move the camera around.\n"
           "When the match score falls between 90-95%% the demo will add a new template.\n\n"
           "Keys:\n"
           "\t h   -- This help page\n"
           "\t l   -- Toggle online learning\n"
           "\t m   -- Toggle printing match result\n"
           "\t t   -- Toggle printing timings\n"
           "\t [ ] -- Adjust matching threshold: '[' down,  ']' up\n"
           "\t c   -- capture (for device-connected mode)\n"
           "\t r   -- record till pressed again (for device-connected mode)\n"
           "\t q   -- Quit\n\n");
}

// Adapted from cv_timer in cv_utilities
class Timer
{
public:
    Timer() : start_(0), time_(0) {}

    void start()
    {
        start_ = cv::getTickCount();
    }

    void stop()
    {
        CV_Assert(start_ != 0);
        int64 end = cv::getTickCount();
        time_ += end - start_;
        start_ = 0;
    }

    double time()
    {
        double ret = time_ / cv::getTickFrequency();
        time_ = 0;
        return ret;
    }

private:
    int64 start_, time_;
};

// Functions to store detector and templates in single XML/YAML file
static cv::Ptr<cv::linemod::Detector> readLinemod(const std::string& filename)
{
    cv::Ptr<cv::linemod::Detector> detector = cv::makePtr<cv::linemod::Detector>();
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    detector->read(fs.root());

    cv::FileNode fn = fs["classes"];
    for (cv::FileNodeIterator i = fn.begin(), iend = fn.end(); i != iend; ++i)
        detector->readClass(*i);

    return detector;
}

void PrintUsage(){
    cout<<"Usage(1): ./C-armTracker [dir_for_tracking] [*_c/d.xml]"<<endl;
    cout<<"1 argc -> no tracking (maybe for recording)"<<endl;
    cout<<"2 argc -> tracking"<<endl;
    cout<<"3 argc -> tracking with recorded file"<<endl;
    cout<<"Usage(1): ./C-armTracker -r [record_file_name]"<<endl;
}

int main(int argc, char* argv[])
{
    // Arguments
    bool tracking = true;
    bool fromRecorded = false;
    bool xyTable = false;
    string path;
    string recordFileName("record");
    labeldata label;
    if(argc==1) tracking = false;
    else if(argc==2 && string(argv[2])=="-xy") { xyTable = true; }
    else if(argc==3 && string(argv[1])=="-r" ) { tracking = false; recordFileName = string(argv[2]); }
    else if(argc>1){
        path = argv[1];
        label = labeldata(path+"/labels.txt");
        if(argc==3){
            fromRecorded = true;
            recordFileName = string(argv[2]);
        }
    }


    // Read C-arm model
    viz::Viz3d myWindow("PLY viewer");
    WPoly poly;
    if(tracking){
        poly.Initialize(path+"/model.ply",label);
        myWindow.setWindowSize(Size(1280, 720));
        myWindow.showWidget("Coordinate", viz::WCoordinateSystem(500.));
        myWindow.showWidget("model PLY", poly);
        Vec3f cam_pos(0,0,-3000), cam_focal_point(0,0,1), cam_y_dir(0,-1.,0);
        Affine3f cam_pose = viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
        myWindow.setViewerPose(cam_pose);
        // poly.Transform(0,0);
        myWindow.spin();
    }
    cout << endl;

    // Various settings and flags
    bool show_match_result = true;
    bool show_timings = false;
    int num_classes = 0;
    int matching_threshold = 70;

    // Timers
    Timer match_timer;

    // Initialize HighGUI
    help();
    cv::namedWindow("color");
    cv::namedWindow("normals");
    Mouse::start("color");

    // Initialize LINEMOD data structures
    cv::Ptr<cv::linemod::Detector> detector;
    if(tracking){
        detector = readLinemod(path+"/templates.yml");
        num_classes = detector->numClasses();
        printf("Loaded %s with %d classes and %d templates\n",
               argv[1], num_classes, detector->numTemplates());
    }
    else{
        detector = cv::linemod::getDefaultLINEMOD();
    }
    //    if (!ids.empty())
    //    {
    //        printf("Class ids:\n");
    //        std::copy(ids.begin(), ids.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
    //    }

    int num_modalities = (int)detector->getModalities().size();

    //**** Azure Kinect sensor
    std::vector<uint32_t> device_indices{ 0 };
    int32_t color_exposure_usec = 8000;  // somewhat reasonable default exposure time
    int32_t powerline_freq = 2;          // default to a 60 Hz powerline
    MultiDeviceCapturer capturer;
    k4a_device_configuration_t main_config, secondary_config;
    k4a::transformation main_depth_to_main_color;
    if(!fromRecorded){
        capturer = MultiDeviceCapturer(device_indices, color_exposure_usec, powerline_freq);
        // Create configurations for devices
        main_config = get_master_config();
        main_config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
        main_config.camera_fps = K4A_FRAMES_PER_SECOND_15;
        main_config.color_resolution = K4A_COLOR_RESOLUTION_720P;
        main_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;// no need to have a master cable if it's standalone
        secondary_config = get_subordinate_config(); // not used - currently standalone mode
        // Construct all the things that we'll need whether or not we are running with 1 or 2 cameras
        k4a::calibration main_calibration = capturer.get_master_device().get_calibration(main_config.depth_mode,main_config.color_resolution);
        // Set up a transformation. DO THIS OUTSIDE OF YOUR MAIN LOOP! Constructing transformations involves time-intensive
        // hardware setup and should not change once you have a rigid setup, so only call it once or it will run very
        // slowly.
        main_depth_to_main_color = k4a::transformation(main_calibration);
        capturer.start_devices(main_config, secondary_config);

        cout << "calibration" << endl;
        cout << main_calibration.color_resolution << endl;
    }


    cv::Mat color, depth;
    bool recording(false), capt(false);
    int recordID(0);
    cv::FileStorage fs_d, fs_c;
    vector<cv::Mat> colorVec, depthVec;
    if(fromRecorded){
		cout << path+"/"+recordFileName+"_c.xml" << endl;
		cout << path+"/"+recordFileName+"_d.xml" << endl;
		fs_c.open(path+"/"+recordFileName+"_c.xml", cv::FileStorage::READ);
		fs_d.open(path+"/"+recordFileName+"_d.xml", cv::FileStorage::READ);
        string matName = "frame"+to_string(recordID);
        while(1){
            cv::Mat color1, depth1;
            fs_c[matName]>>color1;
            if(color1.empty())break;
            cout<<"\rReading frame "<<recordID<<flush;
            colorVec.push_back(color1);
            fs_d[matName]>>depth1;
            depthVec.push_back(depth1);
            matName = "frame"+to_string(++recordID);
        }
        fs_c.release();
        fs_d.release();
        cout<<"\rImported "<<--recordID<<" frames"<<endl;
        recordID = 0;
    }


	// CHECK THE DEPTH IMAGE AND ROTATED MODEL
	cv::FileStorage fs_d_pcd;
    vector<cv::Mat> depthPCDVec;

    int width_d = 1280; // cols
    int height_d = 720; // rows

    // Azure Kinect xy_table
//	if (xyTable)  create_xy_table(height_d, width_d);

    cout << "Read XY table" << flush;
    ifstream ifs("xytable.txt");
    string xxx, yyy;
    vector<pair<float, float>> xy_table;
    vector<tuple<float, float, float>> pcd_data;;
    for (int y=0; y<height_d; y++) {
        for (int x=0; x<width_d; x++) {
            ifs >> xxx >> yyy;
            if (xxx == "nan" && yyy == "nan") { xy_table.push_back({0,0}); continue; }
            xy_table.push_back({stof(xxx),stof(yyy)});
        }
    } cout << "...done" << endl;


    // Print PCD =======================================================
    cout << "Print Point Cloud Data" << flush;
    float xx, yy; 
    fs_d_pcd.open(path+"/"+recordFileName+"_d.xml", cv::FileStorage::READ);
    for (int i=0; i<depthVec.size(); i++) {
        string name = "frame" + to_string(i);
        cv::Mat depth2;
        fs_d_pcd[name] >> depth2;
        depthPCDVec.push_back(depth2);

        
        int idx(-1);
        for (int y=0; y<height_d; y++) {
            for (int x=0; x<width_d; x++) {
                idx++;
                float z = depthPCDVec[i].at<ushort>(y,x);
                // ifs >> xxx >> yyy;
                // if (xxx == "nan" && yyy == "nan") { xy_table.push_back({0,0}); continue; }
                // xy_table.push_back({stof(xxx),stof(yyy)});
                    if (z==0) continue;
                xx = xy_table[idx].first*z; yy = xy_table[idx].second*z;
                pcd_data.push_back(make_tuple(xx,yy,z));
            }
    	}

        ofstream pcd("pcd_set/frame" + to_string(i) +"_pcd.ply");
        pcd << "ply" << std::endl;
        pcd << "format ascii 1.0" << std::endl;
        pcd << "element vertex " << pcd_data.size() << std::endl;
        pcd << "property float x" << std::endl;
        pcd << "property float y" << std::endl;
        pcd << "property float z" << std::endl;
        pcd << "end_header" << std::endl;

        for (auto itr:pcd_data) {
            pcd << get<0>(itr) << " " << get<1>(itr) << " " << get<2>(itr) << endl;
        }
        if (i==depthVec.size()-1) continue;
        pcd_data.clear();
    }
    cout << "...done" << endl;
    // ======================================================= PCD

//     ofstream ofs1("linemod_xyz+depth.txt");
//     ofstream ofs2("linemod_sim+ID.txt");
//     ofstream ofs3("linemod_3D.txt");
//     ofstream ofs4("linemod_2D.txt");
//     ofstream ofs_xyz("source_xyz.txt");

//     bool endXML(false);
//     printPLY = true;

//     double source[3] = {0,0,-810};
//     double origin[3] = {0,0,0};
//     vtkSmartPointer<vtkTransform> transform  = vtkSmartPointer<vtkTransform>::New();

//     // Main Loop
//     for (;;)
//     {
//         if(fromRecorded){
//             if(recordID==depthVec.size()) break;
//             depth = depthVec[recordID];
//             color = colorVec[recordID++];
//         }
//         else{
//             vector<k4a::capture> captures;
//             captures = capturer.get_synchronized_captures(secondary_config, true);
//             k4a::image main_color_image = captures[0].get_color_image();
//             k4a::image main_depth_image = captures[0].get_depth_image();

//             // let's green screen out things that are far away.
//             // first: let's get the main depth image into the color camera space
//             k4a::image main_depth_in_main_color = create_depth_image_like(main_color_image);
//             main_depth_to_main_color.depth_image_to_color_camera(main_depth_image, &main_depth_in_main_color);
//             depth = depth_to_opencv(main_depth_in_main_color);
//             color = color_to_opencv(main_color_image);
// //            cv::imshow("masked", depth);
//             /*color2depth - dose not work for linemod*/
// //            k4a_image_t transformed = color_to_depth(transformation,main_depth_image.handle(),main_color_image.handle());
// //            depth = depth_to_opencv(main_depth_image);
// //            color = color_to_opencv(transformed);

//             if(recording || capt){
//                 if(!fs_d.isOpened()){
//                     fs_c.open(recordFileName+"_c.xml", cv::FileStorage::WRITE);
//                     fs_d.open(recordFileName+"_d.xml", cv::FileStorage::WRITE);
//                 }
//                 fs_d << "frame"+to_string(recordID)<< depth;
//                 fs_c << "frame"+to_string(recordID)<< color;

//                 cout<<"Recorded frame "<<recordID++<< endl;
//                 if(capt==true) cout<<endl;
//                 capt = false;
//             }
//         }

//         if (endXML) {
//         	fs_d.release();
// 			fs_c.release();
//         }



//         std::vector<cv::Mat> sources;
//         cv::Mat depthOrigin = depth.clone();
//         depth = (depth-label.GetMinDist())*label.GetDepthFactor();
//         sources.push_back(color);
//         sources.push_back(depth);

//         cv::Mat display = color.clone();

//         // Perform matching
//         std::vector<cv::linemod::Match> matches;
//         std::vector<cv::String> class_ids;
//         std::vector<cv::Mat> quantized_images;
//         match_timer.start();
//         detector->match(sources, (float)matching_threshold, matches, class_ids, quantized_images);
//         match_timer.stop();

//         int classes_visited = 0;
//         std::set<std::string> visited;
//         if(tracking){
//             int maxID; double maxSim(-1); int match_count(0);
//             for (int i = 0; (i < (int)matches.size()) && (classes_visited < num_classes); ++i)
//             {
//                 cv::linemod::Match m = matches[i];
//                 if (visited.insert(m.class_id).second)
//                 {
//                     ++classes_visited;

//                     if (show_match_result)
//                     {
//                         printf("Similarity: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
//                                m.similarity, m.x, m.y, m.class_id.c_str(), m.template_id);
//                                match_count++;
//                     }
//                     if(m.similarity>maxSim){
//                         maxSim = m.similarity;
//                         maxID = i;
//                     }

//                 }
//             }

            
//             if(maxSim>0){
//                 cv::linemod::Match m = matches[maxID];

//                 // Draw matching template
//                 const std::vector<cv::linemod::Template>& templates = detector->getTemplates(m.class_id, m.template_id);
//                 drawResponse(templates, num_modalities, display, cv::Point(m.x, m.y), detector->getT(0), depthOrigin, xy_table);


//                 // Debugging zone =================================================================================
//                 // ofstream ofs1("linemod_xyz+depth.txt");
//                 // ofstream ofs2("linemod_sim+ID.txt");
//                 // ofstream ofs3("linemod_3D.txt");
//                 // ofstream ofs4("linemod_2D.txt");
//                 // ofstream ofs_xyz("source_xyz.txt");

//                 ofs1 << m.x << " " << m.y << " " << depth.at<ushort>(m.y,m.x) << " " << depth.cols*m.y + m.x << "|"
//                 		<< xy_table[depth.cols*m.y + m.x].first   << " "
// 						<< xy_table[depth.cols*m.y + m.x].second  << "|"
// 						<< xy_table[depth.cols*m.y + m.x].first  * depth.at<ushort>(m.y,m.x) << " "
// 						<< xy_table[depth.cols*m.y + m.x].second * depth.at<ushort>(m.y,m.x) << endl;
//                 ofs2 << m.similarity << " " << m.class_id.c_str() << " " << m.template_id << endl;

//                 int iC = m.x + label.GetPosition(stoi(m.class_id), m.template_id).first;
//                 int jC = m.y + label.GetPosition(stoi(m.class_id), m.template_id).second;
//                 double kC = label.GetDistance(stoi(m.class_id), m.template_id);

//                 double posX = xy_table[depth.cols*jC + iC].first  * kC;
//                 double posY = xy_table[depth.cols*jC + iC].second * kC;
//                 double posZ = kC;


//                 ofs3 << xy_table[depth.cols*m.y + m.x].first * kC<< " " 
//                      << xy_table[depth.cols*m.y + m.x].second * kC<< " " 
//                      << kC << "\t" << posX << " "	<< posY << " "	<< posZ << endl;

//                 ofs4 << m.x << "\t" << m.y << "\t" << iC << "\t" << jC << endl;
//                 // Debugging zone =================================================================================

//                 double source_rot[3], origin_rot[3];
//                 transform->Identity();
//                 transform->SetMatrix(label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id));
//                 // transform->RotateX(180);
//                 transform->TransformPoint(origin, origin_rot);
//                 transform->TransformPoint(source, source_rot);
                
//                 origin_rot[0] += posX; origin_rot[1] += posY;
//                 source_rot[0] += posX; source_rot[1] += posY;

//                 // Print Results
//                 ofs_xyz << "Frame " << recordID-1 << endl;
//                 ofs_xyz << "Index " << m.class_id << " " << m.template_id << " " << endl;
//                 ofs_xyz << "AffineT" << endl;
//                 ofs_xyz << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[0] << " "
//                         << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[1] << " "
//                         << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[2] << " "
//                         << posX << endl;
//                 ofs_xyz << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[4] << " "
//                         << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[5] << " "
//                         << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[6] << " "
//                         << posY << endl;
//                 ofs_xyz << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[8] << " "
//                         << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[9] << " "
//                         << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[10] << " "
//                         << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[11] << endl;
//                 ofs_xyz << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[12] << " "
//                         << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[13] << " "
//                         << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[14] << " "
//                         << label.GetAffineTransformMatrix(stoi(m.class_id), m.template_id)[15] << endl;                                        
                
//                 ofs_xyz << "Isocenter " << origin_rot[0] << " " << origin_rot[1] << " " << origin_rot[2] << endl;
//                 ofs_xyz << "Source    " << source_rot[0] << " " << source_rot[1] << " " << source_rot[2] << endl;

//                 ofs_xyz << "Power     " << 1 << endl;
//                 ofs_xyz << "Time      " << 0 << endl;
//                 ofs_xyz << "DAP_rate  " << 0.0010 << " Gy-cm2/s" << endl;
//                 ofs_xyz << "T_Voltage " << 62     << " kV"       << endl;
//                 ofs_xyz << "T_Current " << 1.2    << " mA"       << endl << endl;
                               

//                 poly.Transform(atoi(m.class_id.c_str()), m.template_id, origin_rot);
//                 myWindow.spinOnce(1, true);
//             } // max Sim

//             if (show_match_result && matches.empty())
//                 printf("No matches found...\n");
//             if (show_timings)
//             {
//                 printf("Matching: %.2fs\n", match_timer.time());
//             }
//             if (show_match_result || show_timings)

//                 cout << "Match count: " << match_count << endl;
//                 printf("Matching: %.2fs\n", match_timer.time());
//                 printf("------------------------------------------------------------\n");
//         } // tracking
//         cv::imshow("color", display);
//         cv::imshow("normals", quantized_images[1]);

//         cv::imwrite("./capture/color/color" + to_string(recordID) +".png", display);
//         cv::imwrite("./capture/normal/normal" + to_string(recordID) +".png", quantized_images[1]);




//         char key = (char)cv::waitKey(10);
//         if (key == 'q')
//             break;

//         switch (key)
//         {
// 			case 'h':
// 				help();
// 				break;
// 			case 'm':
// 				// toggle printing match result
// 				show_match_result = !show_match_result;
// 				printf("Show match result %s\n", show_match_result ? "ON" : "OFF");
// 				break;
// 			case 't':
// 				// toggle printing timings
// 				show_timings = !show_timings;
// 				printf("Show timings %s\n", show_timings ? "ON" : "OFF");
// 				break;
// 			case '[':
// 				// decrement threshold
// 				matching_threshold = std::max(matching_threshold - 1, -100);
// 				printf("New threshold: %d\n", matching_threshold);
// 				break;
// 			case ']':
// 				// increment threshold
// 				matching_threshold = std::min(matching_threshold + 1, +100);
// 				printf("New threshold: %d\n", matching_threshold);
// 				break;
// 			case 'c':
// 				//capture switch
// 				cout<<"Capture switch ON"<<endl;
// 				capt = true;
// 				break;
// 			case 'r':
// 				//record switch
// 				if(recording) cout<<endl<<"Recording switch OFF"<<endl;
// 				else cout<<"Recording switch ON"<<endl;
// 				recording = !recording;
// 				if(!recording) endXML=true;
// 				break;
// 			default:
// 				;
//         } // switch



//     } // Main loop


    return 0;
}

// Adapted from cv_show_angles
cv::Mat displayQuantized(const cv::Mat& quantized)
{
    cv::Mat color(quantized.size(), CV_8UC3);
    for (int r = 0; r < quantized.rows; ++r)
    {
        const uchar* quant_r = quantized.ptr(r);
        cv::Vec3b* color_r = color.ptr<cv::Vec3b>(r);

        for (int c = 0; c < quantized.cols; ++c)
        {
            cv::Vec3b& bgr = color_r[c];
            switch (quant_r[c])
            {
            case 0:   bgr[0] = 0; bgr[1] = 0; bgr[2] = 0;    break;
            case 1:   bgr[0] = 55; bgr[1] = 55; bgr[2] = 55;    break;
            case 2:   bgr[0] = 80; bgr[1] = 80; bgr[2] = 80;    break;
            case 4:   bgr[0] = 105; bgr[1] = 105; bgr[2] = 105;    break;
            case 8:   bgr[0] = 130; bgr[1] = 130; bgr[2] = 130;    break;
            case 16:  bgr[0] = 155; bgr[1] = 155; bgr[2] = 155;    break;
            case 32:  bgr[0] = 180; bgr[1] = 180; bgr[2] = 180;    break;
            case 64:  bgr[0] = 205; bgr[1] = 205; bgr[2] = 205;    break;
            case 128: bgr[0] = 230; bgr[1] = 230; bgr[2] = 230;    break;
            case 255: bgr[0] = 0; bgr[1] = 0; bgr[2] = 255;    break;
            default:  bgr[0] = 0; bgr[1] = 255; bgr[2] = 0;    break;
            }
        }
    }

    return color;
}

void drawResponse(const std::vector<cv::linemod::Template>& templates,
                  int num_modalities, cv::Mat& dst, cv::Point offset, int T, 
                  cv::Mat depth, vector<pair<float,float>> xy_table)
{
    static const cv::Scalar COLORS[5] = { CV_RGB(0, 0, 255),
                                          CV_RGB(0, 255, 0),
                                          CV_RGB(255, 255, 0),
                                          CV_RGB(255, 140, 0),
                                          CV_RGB(255, 0, 0) };

    res << endl;
    for (int m = 0; m < num_modalities; ++m)
    {
        // NOTE: Original demo recalculated max response for each feature in the TxT
        // box around it and chose the display color based on that response. Here
        // the display color just depends on the modality.
        cv::Scalar color = COLORS[m];

        for (int i = 0; i < (int)templates[m].features.size(); ++i)
        {
            cv::linemod::Feature f = templates[m].features[i];
            cv::Point pt(f.x + offset.x, f.y + offset.y);
            cv::circle(dst, pt, T / 2, color);
            res << xy_table[depth.cols*pt.y + pt.x].first  * depth.at<ushort>(pt.y,pt.x) << " "
                << xy_table[depth.cols*pt.y + pt.x].second * depth.at<ushort>(pt.y,pt.x) << " "
                << depth.at<ushort>(pt.y,pt.x) << "    " << pt.x << " " << pt.y
				 	 	 	 	 	 	 	   << "    " << offset.x << " " << offset.y << endl;
        }
        res << "---------" << endl;
    }
}
