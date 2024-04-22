#include <vector>
#include <tuple>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/io/wkt/wkt.hpp>
#include <list>
#include <boost/assign/list_of.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/algorithms/within.hpp>
using namespace std;
using namespace cv;
namespace bg = boost::geometry;



void show(std::string name, cv::Mat img_show)
{
    float img_resize=0.3;
    cv::Mat img_show_resize;
    cv::resize(img_show, img_show_resize, cv::Size(), img_resize, img_resize);
    cv::imshow(name, img_show_resize);
    cv::waitKey(0);
}

//特征提取
void detectKeypointsAndDescriptors(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors,int flag)
{
    if(flag==0)
    {
        // 初始化SIFT检测器
        Ptr<SIFT> sift = SIFT::create();
        sift->detectAndCompute(image, noArray(), keypoints, descriptors);


    }
    else if(flag==1)
    {
        // 初始化SURF检测器
        Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create();
        surf->detectAndCompute(image, noArray(), keypoints, descriptors);


    }
    else if(flag==2)
    {

        // 初始化ORB检测器
        Ptr<ORB> orb = ORB::create();
        orb->detectAndCompute(image, noArray(), keypoints, descriptors);

    }
    else if(flag==3)
    {
        // 初始化AKAZE检测器
        Ptr<AKAZE> akaze = AKAZE::create();
        // 检测关键点和计算描述子
        akaze->detectAndCompute(image, noArray(), keypoints, descriptors);
    }

    // 打印关键点数量
    cout << "关键点数量: " << keypoints.size() << endl;
    cout << "描述子数量: " << descriptors.rows << endl;

    // 绘制特征点的位置并显示描述子
//    Mat image_with_keypoints;
//    drawKeypoints(image, keypoints, image_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DEFAULT);


//    // 显示图像和检测到的关键点
//    imshow("AKAZE特征检测", image_with_keypoints);
//    waitKey(0);
}

//特征匹配
////1 粗匹配：暴力匹配
vector<DMatch> feature_match_ByFLANN(Mat& des1,Mat& des2,double threshold,int flag,int flag1)
{
    // 对匹配结果进行处理
    std::vector<cv::DMatch> good_matches;
    if(flag == 1)
    {
        if(flag1==1)
        {
            BFMatcher bf(cv::NORM_L2); // 使用欧氏距离进行匹配
            vector<vector<DMatch>> matches;
            bf.knnMatch(des1, des2, matches, 2);


            // 第一步筛选：
            // 应用比率测试，保留好的匹配
            for (size_t i = 0; i < matches.size(); ++i)
            {
                if (matches[i][0].distance < threshold * matches[i][1].distance)
                {
                    good_matches.push_back(matches[i][0]);
                }
            }
        }
        else
        {
            BFMatcher bf(NORM_HAMMING);//使用汉明距离
            vector<vector<DMatch>> matches;
            bf.knnMatch(des1, des2, matches, 2);


            // 第一步筛选：
            // 应用比率测试，保留好的匹配
            for (size_t i = 0; i < matches.size(); ++i)
            {
                if (matches[i][0].distance < threshold * matches[i][1].distance)
                {
                    good_matches.push_back(matches[i][0]);
                }
            }
        }



    }
    else
    {
        //FLANN
        if(flag1 ==1)
        {
            //SIFT /SURF
            FlannBasedMatcher flann;
            vector<vector<DMatch>> matches;
            flann.knnMatch(des1, des2, matches, 2);
            // 第一步筛选：
            // 应用比率测试，保留好的匹配
            for (size_t i = 0; i < matches.size(); ++i)
            {
                if (matches[i][0].distance < threshold * matches[i][1].distance)
                {
                    good_matches.push_back(matches[i][0]);
                }
            }
        }

    }

    // 先按照距离对matches中的匹配对进行排序。按照距离的升序对它们进行排序，以使最佳匹配（低距离）
    // 这个距离具体指的是什么？汉明距离吗？
    sort(good_matches.begin(), good_matches.end(), [](DMatch& a, DMatch& b)
    {
        return a.distance < b.distance;
    });

    return good_matches;
}
//1 粗匹配：暴力匹配
vector<DMatch> feature_match_ByBF(Mat& des1,Mat& des2,double threshold,int flag)
{
    // 对匹配结果进行处理
    std::vector<cv::DMatch> good_matches;

    if(flag==1)
    {
        BFMatcher bf(cv::NORM_L2); // 使用欧氏距离进行匹配
        vector<vector<DMatch>> matches;
        bf.knnMatch(des1, des2, matches, 2);


        // 第一步筛选：
        // 应用比率测试，保留好的匹配
        for (size_t i = 0; i < matches.size(); ++i)
        {
            if (matches[i][0].distance < threshold * matches[i][1].distance)
            {
                good_matches.push_back(matches[i][0]);
            }
        }
    }
    else
    {
        BFMatcher bf(NORM_HAMMING);//使用汉明距离
        vector<vector<DMatch>> matches;
        bf.knnMatch(des1, des2, matches, 2);


        // 第一步筛选：
        // 应用比率测试，保留好的匹配
        for (size_t i = 0; i < matches.size(); ++i)
        {
            if (matches[i][0].distance < threshold * matches[i][1].distance)
            {
                good_matches.push_back(matches[i][0]);
            }
        }
    }





    // 先按照距离对matches中的匹配对进行排序。按照距离的升序对它们进行排序，以使最佳匹配（低距离）
    // 这个距离具体指的是什么？汉明距离吗？
    sort(good_matches.begin(), good_matches.end(), [](DMatch& a, DMatch& b)
    {
        return a.distance < b.distance;
    });

    return good_matches;
}
//细匹配
cv::Mat feature_match_ByRANSAC(cv::Mat img1, std::vector<cv::KeyPoint> kp1, cv::Mat img2, std::vector<cv::KeyPoint> kp2, std::vector<cv::DMatch> good_matches)
{
    // Minimum number of matches should be greater than 3
    int MIN_MATCH_COUNT = 3;

    cv::Mat T;
    cv::Mat T_3x3;
    if (good_matches.size() > MIN_MATCH_COUNT)
    {
        cv::Mat src_pts(good_matches.size(), 1, CV_32FC2);
        cv::Mat dst_pts(good_matches.size(), 1, CV_32FC2);

        for (int i = 0; i < good_matches.size(); i++)
        {
            src_pts.at<cv::Point2f>(i) = kp1[good_matches[i].queryIdx].pt;
            dst_pts.at<cv::Point2f>(i) = kp2[good_matches[i].trainIdx].pt;
        }

        // Run RANSAC and estimate affine matrix
        cv::Mat mask; // RANSAC输出的掩码，指示哪些点被认为是内点
        T = cv::estimateAffinePartial2D(src_pts, dst_pts, mask, cv::RANSAC);


        std::cout << "T: " <<T << std::endl;

        // 将 2x3 矩阵 T 转换为 3x3 矩阵
        T_3x3 = cv::Mat::zeros(3, 3, CV_64FC1);
        T.rowRange(0, 2).copyTo(T_3x3.rowRange(0, 2));

        // 将 T_3x3 的最后一行设置为 [0, 0, 1]
        T_3x3.at<double>(2, 2) = 1;

        std::cout << "转换后的仿射变换矩阵 T：\n" << T_3x3 << std::endl;
        // 计算内点数
        std::cout << "Total Points: " << mask.total() << std::endl;
        std::cout << "Inliers Number: " << countNonZero(mask) << std::endl;
        std::cout << "Inliers Ratio: " << static_cast<double>(cv::countNonZero(mask)) / mask.total() << std::endl;

        //全部匹配
        cv::Mat img_matches;
        cv::drawMatches( img1, kp1, img2, kp2, good_matches, img_matches,cv::Scalar(0,0,255),cv::Scalar(0,0,255),vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


        // 绘制内点
        cv::Mat img_matches_ransac;
        cv::drawMatches(img1, kp1, img2, kp2, good_matches, img_matches_ransac, cv::Scalar(0,0,255),cv::Scalar(0,0,255), mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//        imshow("Matches", img_matches );
//        imshow("Inlier Matches", img_matches_ransac);
        show("Matches", img_matches );
        show("Inlier Matches", img_matches_ransac);
        cv::waitKey(0);
        return T_3x3;

    }
    else
    {
        std::cout << "Not enough matches are found - " << good_matches.size() << "/" << MIN_MATCH_COUNT << std::endl;
        return T_3x3;
    }

}
tuple<int, int, Mat> ROI(Mat img1, Mat img2, Mat T)
{
    // 计算 img1 的边界角点
    int height = img1.rows;
    int width = img1.cols;
    cout << height << endl;
    cout << width << endl;

    vector<Point2f> corners =
    {
        Point2f(0, 0),
        Point2f(width, 0),
        Point2f(width, height),
        Point2f(0, height)
    };

    cout << "原始角点：" << endl;
    for (const auto &corner : corners)
    {
        cout << corner << endl;
    }

    // 获取变换后的角点坐标
    vector<Point2f> transformed_corners;
    perspectiveTransform(corners, transformed_corners, T);


    vector<Point> transformed_corners_int;
    for (const auto &corner : transformed_corners)
    {
        cout << corner << endl;
        transformed_corners_int.push_back(Point(int(corner.x), int(corner.y)));
    }

    cout << "变换后角点为：" << endl;
    for (const auto &corner : transformed_corners_int)
    {
        cout << corner << endl;
    }
    // 计算最左
    int merged_left = min(min_element(transformed_corners_int.begin(), transformed_corners_int.end(), [](const Point2f &p1, const Point2f &p2)
    {
        return p1.x < p2.x;
    })->x, 0);
    // 计算最右
    int merged_right = max(max_element(transformed_corners_int.begin(), transformed_corners_int.end(), [](const Point2f &p1, const Point2f &p2)
    {
        return p1.x < p2.x;
    })->x, img2.cols);
    // 计算最上
    int merged_top = min(min_element(transformed_corners_int.begin(), transformed_corners_int.end(), [](const Point2f &p1, const Point2f &p2)
    {
        return p1.y < p2.y;
    })->y, 0);
    // 计算最下
    int merged_bottom = max(max_element(transformed_corners_int.begin(), transformed_corners_int.end(), [](const Point2f &p1, const Point2f &p2)
    {
        return p1.y < p2.y;
    })->y, img2.rows);
    // 计算融合后图像的宽度和高度
    int merged_width = merged_right - merged_left;
    int merged_height = merged_bottom - merged_top;
    cout << "最左面：" << merged_left << endl;
    cout << "最右面：" << merged_right << endl;
    cout << "最上面：" << merged_top << endl;
    cout << "最下面：" << merged_bottom << endl;
    cout << "宽度：" << merged_width << endl;
    cout << "高度：" << merged_height << endl;
    // 定义平移矩阵
    Mat m = (Mat_<double>(2, 3) << 1, 0, -merged_left, 0, 1, -merged_top);
    return make_tuple(merged_height, merged_width, m);
}


// 定义点的类型
typedef bg::model::d2::point_xy<double> myPoint;

vector<myPoint> transform_corners(Mat img, Mat T)
{
    // 计算 img 的边界角点
    int height = img.rows;
    int width = img.cols;
    vector<Point2f> corners = {Point2f(0, 0), Point2f(width, 0), Point2f(width, height), Point2f(0, height)};
    cout << "原始角点：" << endl;
    for (const auto &corner : corners)
    {
        cout << corner << endl;
    }
    // 获取变换后的角点坐标，变换前需要对数据进行整理
    vector<Point2f> transformed_corners;
    perspectiveTransform(corners, transformed_corners, T);
    // 将角点坐标转换为整数类型
    vector<myPoint> transformed_corners_int;
    for (const auto &corner : transformed_corners)
    {
        transformed_corners_int.push_back(myPoint(int(corner.x), int(corner.y)));
    }

    cout << "变换后角点为：" << endl;
    for (const auto &p : transformed_corners_int)
    {
        cout <<  "(" << bg::get<0>(p) << ", " << bg::get<1>(p) << ")"<<endl;
    }

    return transformed_corners_int;
}

//定义重叠函数
double overlap(Mat img1_before, Mat img2_before, Mat img1_transform, Mat img2_transform, Mat T, Mat t)
{

    //定义多边形的类型
    typedef bg::model::polygon<myPoint, false> Polygon;
    typedef bg::model::box<myPoint> Box;
    cout<<"img1_transform_corners"<<endl;
    // 计算图像1和图像2的变换后角点
    vector<myPoint> img1_transform_corners = transform_corners(img1_before, T);
    vector<myPoint> img2_transform_corners = transform_corners(img2_before, t);
    for (const auto &p : img1_transform_corners)
    {
        cout <<  "(" << bg::get<0>(p) << ", " << bg::get<1>(p) << ")"<<endl;
    }
    cout<<"img2_transform_corners"<<endl;
    for (const auto &p : img2_transform_corners)
    {
        cout <<  "(" << bg::get<0>(p) << ", " << bg::get<1>(p) << ")"<<endl;
    }



// 创建矩形1的Polygon对象
    Polygon poly1, poly2;

    poly1.outer() = boost::assign::list_of<myPoint>(img1_transform_corners[0])(img1_transform_corners[1])(img1_transform_corners[2])(img1_transform_corners[3])(img1_transform_corners[0]);
    poly2.outer() = boost::assign::list_of<myPoint>(img2_transform_corners[0])(img2_transform_corners[1])(img2_transform_corners[2])(img2_transform_corners[3])(img2_transform_corners[0]);

    // 计算重叠部分的得分
    std::list<Polygon> intersectionResult;
    bg::intersection(poly1, poly2, intersectionResult);

    // 打印结果
    std::cout << "Intersection result:" << std::endl;
    for (const auto& poly : intersectionResult)
    {
        std::cout << "Polygon:" << std::endl;
        for (const auto& point : poly.outer())
        {
            std::cout << "Point(" << bg::get<0>(point) << ", " << bg::get<1>(point) << ")" << std::endl;
        }

    }
    // 创建一个新的多边形 polygon3
    Polygon polygon3;
    for (const auto& poly : intersectionResult)
    {
        // 添加每个多边形的外环到 polygon3 的外环中
        for (const auto& point : poly.outer())
        {
            bg::append(polygon3.outer(), point);
        }
    }


    // 计算 polygon3 的外接矩形
    Box envelope;
    bg::envelope(polygon3, envelope);

    // 获取外接矩形的最小点和最大点坐标
    myPoint rec_min = envelope.min_corner();
    myPoint rec_max = envelope.max_corner();


    // 打印结果
    std::cout << "Intersection result envelope: ";
    int Min_point_x = int( bg::get<0>(rec_min));
    int Min_point_y = int( bg::get<1>(rec_min));
    int Max_point_x = int(bg::get<0>(rec_max));
    int Max_point_y = int(bg::get<1>(rec_max));
    std::cout << "Min point: (" << Min_point_x << ", " << Min_point_y << ")";
    std::cout << ", Max point: (" << Max_point_x << ", " << Max_point_y << ")";
    std::cout << std::endl;

    // 统计重叠区域内像素的对齐程度
    int compute_sum = 0;
    int overlap_sum = 0;
    int effective_sum = 0;
    int alignment_sum = 0;
    int sample = 1;  // 抽样指标
    for (int x = Min_point_x; x < Max_point_x+1; x += sample)
    {
        for (int y = Min_point_y; y < Max_point_y +1; y += sample)
        {
            compute_sum++;
            if(bg::within(myPoint(x, y), polygon3))
            {
                overlap_sum++;
                // 计算对齐程度
                if ((img1_transform.at<uchar>(y, x) >= 250 || img1_transform.at<uchar>(y, x) <= 5) &&
                        (img2_transform.at<uchar>(y, x) >= 250 || img2_transform.at<uchar>(y, x) <= 5))
                {
                    effective_sum++;
                    if (abs(img1_transform.at<uchar>(y, x) - img2_transform.at<uchar>(y, x)) <= 5)
                    {
                        alignment_sum++;
                    }
//                    else
//                    {
//                        std::cout << "未对齐的部分坐标：(" << x << ", " << y << ")" << std::endl;
//                        std::cout << "对应位置上图像1的像素值：" << (int)img1_transform.at<uchar>(y, x) << std::endl;
//                        std::cout << "对应位置上图像2的像素值：" << (int)img2_transform.at<uchar>(y, x) << std::endl;
//                    }

                }
            }

        }
    }
    cout << "重叠区域计算（步长为：" << sample << "）的像素数量：" << compute_sum << endl;
    cout << "实际重叠区域（步长为：" << sample << "）的像素数量：" << overlap_sum << endl;
    cout << "有效重叠区域（步长为：" << sample << "）的像素数量：" << effective_sum << endl;
    cout << "对齐重叠区域（步长为：" << sample << "）的像素数量：" << alignment_sum << endl;

//      计算对齐程度得分
    double score = (effective_sum > 0) ? (double) alignment_sum / effective_sum : 0.0f;
    cout << "对齐/有效（步长为：" << sample << "）: " << score << endl;
    return score;
}


// merge 函数
void merge(cv::Mat img1, cv::Mat img2, cv::Mat T)
{
    // 计算融合后的图像 ROI
    int height, width;
    cv::Mat t;
    std::tie(height, width, t) = ROI(img1, img2, T);
    std::cout << "平移矩阵为：\n" << t << std::endl;
    std::cout << "变换矩阵修正前为：\n" << T << std::endl;
    // 调整变换矩阵
    T.at<double>(0, 2) += t.at<double>(0, 2);  // 设置水平平移分量
    T.at<double>(1, 2) += t.at<double>(1, 2);  // 设置垂直平移分量

    std::cout << "变换矩阵修正后为：\n" << T << std::endl;
    // 对第一张图片进行透视变换
    Mat transform_img1;
    warpPerspective(img1, transform_img1, T, Size(width, height), INTER_LINEAR, BORDER_CONSTANT, Scalar(125, 125, 125));
    show("Transform_img1", transform_img1);
    // 对齐 img2 左上角和融合后的左上角
    // 用仿射变换实现平移
    Mat img2_transform;
    warpAffine(img2, img2_transform, t, Size(width, height), INTER_LINEAR, BORDER_CONSTANT, Scalar(125, 125, 125));
    show("img2_transform", img2_transform);
    // 保存 transform_img1 为 PGM 格式
//    string transform_img1_filename = "transform_img1.pgm";
//    imwrite(transform_img1_filename, transform_img1);
//
//// 保存 img2_transform 为 PGM 格式
//    string img2_transform_filename = "img2_transform.pgm";
//    imwrite(img2_transform_filename, img2_transform);

    // 将两张图片并排在一起显示
    Mat connect_image;
    hconcat(transform_img1, img2_transform, connect_image);
    show("connect_image", connect_image);
    cv::waitKey(0);

//    计算得分
    Mat t_3x3 = cv::Mat::zeros(3, 3, CV_64FC1);
    t.rowRange(0, 2).copyTo(t_3x3.rowRange(0, 2));
    // 将 t_3x3 的最后一行设置为 [0, 0, 1]
    t_3x3.at<double>(2, 2) = 1;
    double scores = overlap(img1, img2, transform_img1, img2_transform, T, t_3x3);
    cout << "最优得分：" <<setprecision(8)<< scores << endl;


    //自定义融合
    Mat merged_image = Mat::zeros(transform_img1.size(), transform_img1.type());
    for (int y = 0; y < merged_image.rows; ++y)
    {
        for (int x = 0; x < merged_image.cols; ++x)
        {
            // 获取像素值

            uchar pixel_img1 = transform_img1.at<uchar>(y, x);
            uchar pixel_img2 = img2_transform.at<uchar>(y, x);

            if(static_cast<int>(pixel_img1)<120||static_cast<int>(pixel_img2)<120)
            {
                continue;
            }
            if(static_cast<int>(pixel_img1)>static_cast<int>(pixel_img2))
            {
                merged_image.at<uchar>(y, x) = pixel_img1;
            }
            else
            {
                merged_image.at<uchar>(y, x) = pixel_img2;
            }

        }
    }
    // 直接叠加融合
//    addWeighted(transform_img1, 1, img2_transform, 1, 0, merged_image);

    show("merged_image", merged_image);
    cv::imwrite("global_map.pgm", merged_image);
    waitKey(0);
}


int main()
{

    // 读取输入图像
//    aces
    Mat image1 = imread("..\\maps\\aces\\aces_4.pgm", IMREAD_GRAYSCALE);
    Mat image2 = imread("..\\maps\\aces\\aces_7.pgm", IMREAD_GRAYSCALE);
//    Mat image1 = imread("..\\maps\\aces\\aces_5.pgm", IMREAD_GRAYSCALE);
//    Mat image2 = imread("..\\maps\\aces\\aces_8.pgm", IMREAD_GRAYSCALE);
    // intel
//    Mat image1 = imread("..\\maps\\intel\\intel_5.pgm", IMREAD_GRAYSCALE);
//    Mat image2 = imread("..\\maps\\intel\\intel_10.pgm", IMREAD_GRAYSCALE);
//    Mat image1 = imread("..\\maps\\intel\\intel_3.pgm", IMREAD_GRAYSCALE);
//    Mat image2 = imread("..\\maps\\intel\\intel_51.pgm", IMREAD_GRAYSCALE);

//     Edmonton
//    Mat image1 = imread("..\\maps\\Edmonton\\edmonton_6.pgm", IMREAD_GRAYSCALE);
//    Mat image2 = imread("..\\maps\\Edmonton\\edmonton_10.pgm", IMREAD_GRAYSCALE);
    //Fr079
//    Mat image1 = imread("..\\maps\\fr079\\fr079_4.pgm", IMREAD_GRAYSCALE);
//    Mat image2 = imread("..\\maps\\fr079\\fr079_9.pgm", IMREAD_GRAYSCALE);

////     DM
//    Mat image1 = imread("..\\maps\\DM1\\DM_5.pgm", IMREAD_GRAYSCALE);
//    Mat image2 = imread("..\\maps\\DM1\\DM_9.pgm", IMREAD_GRAYSCALE);

//    Mat image1 = imread("..\\maps\\DM1\\DM_3.pgm", IMREAD_GRAYSCALE);
//    Mat image2 = imread("..\\maps\\DM1\\DM_7.pgm", IMREAD_GRAYSCALE);


    // 检测关键点和计算描述子
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detectKeypointsAndDescriptors(image1, keypoints1, descriptors1,0);
    detectKeypointsAndDescriptors(image2, keypoints2, descriptors2,0);

    //特征匹配
    // 使用 feature_match_ByBF 函数进行特征匹配
    //SIFT/SURF+BF flag=1
    //ORB/AKAZE+BF flag=2
//    vector<DMatch> matches = feature_match_ByBF(descriptors1, descriptors2,0.80,1);

    //SIFT/SURF+FLANN flag=1
    vector<DMatch> matches = feature_match_ByFLANN(descriptors1, descriptors2,0.75,2,1);
    //ORB/AKAZE+BF flag=2
//    vector<DMatch> matches = feature_match_ByFLANN(descriptors1, descriptors2,0.80,1,2);

    cout << "Number of matches: " << matches.size() << endl;
    //细匹配：RANSAC并得到变换矩阵T
    Mat T = feature_match_ByRANSAC(image1, keypoints1, image2, keypoints2, matches);

    //地图融合
    merge(image1, image2,  T);

    system("pause");
    return 0;
}
