#include <vector>
#include <tuple>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "ClusterAnalysis.h"
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

void detectAKAZEKeypointsAndDescriptors(const Mat& image, vector<KeyPoint>& keypoints_akaze, Mat& descriptors_akaze)
{
    // ��ʼ��AKAZE�����
    Ptr<AKAZE> akaze = AKAZE::create();

    // ���ؼ���ͼ���������
    akaze->detectAndCompute(image, noArray(), keypoints_akaze, descriptors_akaze);

    // ��ӡ�ؼ�������
//    cout << "AKAZE�ؼ�������: " << keypoints_akaze.size() << endl;
//
//    cout << "AKAZE����������: " << descriptors_akaze.rows << endl;

    // �����������λ�ò���ʾ������
//    Mat image_with_keypoints_akaze;
//    drawKeypoints(image, keypoints_akaze, image_with_keypoints_akaze, Scalar(0, 0, 255), DrawMatchesFlags::DEFAULT);


//    // ��ʾͼ��ͼ�⵽�Ĺؼ���
//    imshow("AKAZE�������", image_with_keypoints_akaze);
    waitKey(0);
}

// �� std::vector<KeyPoint> ת��Ϊ float* raw_data
float* keypointsToRawData(vector<KeyPoint>& keypoints)
{
    int numKeypoints = keypoints.size();
    float* raw_data = new float[numKeypoints * 3]; // ÿ���ؼ������ x��y ���������
    int index= 0;
    for (int i = 0; i < numKeypoints; ++i)
    {
        raw_data[i * 3] = keypoints[i].pt.x;
        raw_data[i * 3 + 1] = keypoints[i].pt.y;
        raw_data[i * 3 + 2] = index++;
    }

    return raw_data;
}
void printRawData(const float* raw_data, int numElements)
{
    for (int i = 0; i < numElements; ++i)
    {
        cout << raw_data[i] << " ";
    }
    cout << endl;
}

void printResults(Mat image, vector<KeyPoint>& keypoints,  vector<DataPoint>& results,vector<DataPoint>& filteredResults,int clusterNum)
{
    // ����һ����ɫͼ�񸱱�
    Mat resultImage;
    cvtColor(image, resultImage, COLOR_GRAY2BGR);

    Mat resultImage1;
    cvtColor(image, resultImage1, COLOR_GRAY2BGR);

    Mat resultImage2;
    cvtColor(image, resultImage2, COLOR_GRAY2BGR);

    int num_points = results.size();
    int num_points1 = filteredResults.size();
//    cout << "Number of clusters: " << clusterNum << std::endl;
//    cout << "Number of num_points: " << num_points << std::endl;
//    cout << "Number of num_points1: " << num_points1 << std::endl;

    // ��ɫӳ�䣨ÿ���ض�Ӧһ����ɫ��
    vector<Scalar> colors(clusterNum);
    for (int i = 0; i < clusterNum; ++i)
    {
        colors[i] = Scalar(rand() % 256, rand() % 256, rand() % 256); // ���������ɫ
    }

    // ��ͼ���ϻ���ÿ���ؼ��㣬��ʹ�ôص���ɫ���б��
    for (size_t i = 0; i < num_points; ++i)
    {
        int oldIndex = results[i].GetOldIndex();
        int clusterId = results[i].GetClusterId();

        // ��ȡ�ؼ��������
        Point2f pt = keypoints[oldIndex].pt;

        // ��ȡ�ؼ����Ӧ����ɫ
        Scalar color = colors[clusterId];

        // ���ƹؼ���
        circle(resultImage, pt, 3, color, -1); // ��ͼ���ϻ�һ������Բ
    }

    // ��ͼ���ϻ���ÿ���ؼ��㣬��ʹ�ôص���ɫ���б��
    for (size_t i = 0; i < num_points; ++i)
    {
        int oldIndex = results[i].GetOldIndex();
        int clusterId = results[i].GetClusterId();
        if (clusterId == -1)
        {
            continue;
        }

        // ��ȡ�ؼ��������
        Point2f pt = keypoints[oldIndex].pt;

        // ��ȡ�ؼ����Ӧ����ɫ
        Scalar color = colors[clusterId];

        // ���ƹؼ���
        circle(resultImage1, pt, 3, color, -1); // ��ͼ���ϻ�һ������Բ
    }

    // ��ͼ���ϻ���ÿ���ؼ��㣬��ʹ�ôص���ɫ���б��
    for (size_t i = 0; i < num_points1; ++i)
    {
        int oldIndex = filteredResults[i].GetOldIndex();
        int clusterId = filteredResults[i].GetClusterId();
        if (clusterId == -1)
        {
            continue;
        }

        // ��ȡ�ؼ��������
        Point2f pt = keypoints[oldIndex].pt;

        // ��ȡ�ؼ����Ӧ����ɫ
        Scalar color = colors[clusterId];

        // ���ƹؼ���
        circle(resultImage2, pt, 3, color, -1); // ��ͼ���ϻ�һ������Բ
    }


    // ��ʾͼ��
//    imshow("Clustered Image", resultImage);
//    imshow("Clustered Image1��delete-1��", resultImage1);
//    imshow("Clustered Image1(after filtered)", resultImage2);
    waitKey(0);
}

// ȥ����idΪ-1��������
vector<DataPoint> removeNoiseClusters(vector<DataPoint>& clusteringResults)
{
    vector<DataPoint> filtered_clusters;
    for ( auto& point : clusteringResults)
    {
        if (point.GetClusterId() != -1)
        {
            filtered_clusters.push_back(point);
        }
    }
    return filtered_clusters;
}

// ����ص�ƽ����С
double computeAverageClusterSize(vector<DataPoint>& clusteringResults, int clusterNum)
{
    double total_size = 0.0;
    for  (auto& point : clusteringResults)
    {
        if (point.GetClusterId() != -1)
        {
            total_size++;
        }
    }
    return total_size / clusterNum ;

}

// ȥ��С��ƽ����С�Ĵ�
vector<DataPoint> filterSmallClusters( vector<DataPoint>& clusteringResults, double avg_size,int clusterNum)
{
    vector<DataPoint> filtered_clusters;
    //��Ϊ�Ѿ�ȥ�����������ˣ�����-1
    int clusterNum1 = clusterNum-1;
    int* temp = (int*)calloc(clusterNum1, sizeof(int));
    int num_points = clusteringResults.size();
    cout << "Number of clusters(delete -1): " << clusterNum1 << std::endl;
    cout << "Number of num_points(delete -1): " << num_points << std::endl;

    for (int i = 0; i < num_points; i++)
    {
        if (clusteringResults[i].GetClusterId() != -1)
        {
            temp[clusteringResults[i].GetClusterId()] += 1;
        }
    }

    printf("Cluster filtering ......\n");
    for (int i = 0; i < clusterNum; i++)
    {
        if (temp[i] >= avg_size)
        {
            printf("Cluster %d: %d pts\n", i + 1, temp[i]);

        }
    }



    for  (auto& point : clusteringResults)
    {
        if (point.GetClusterId() != -1 &&temp[point.GetClusterId()] >= avg_size)
        {
            filtered_clusters.push_back(point);
        }
    }
    return filtered_clusters;
}
//����ص���������
int computeClusterNum(vector<DataPoint>& clusteringResults)
{
    set<int> unique_cluster_ids;
    // �����������ݵ㣬�����ID���뵽set�У�����set�����ԣ����Զ�ȥ��
    for ( auto& point : clusteringResults)
    {
        unique_cluster_ids.insert(point.GetClusterId());
    }
    // ���ز�ͬ�ص�����
    return unique_cluster_ids.size();
}
//// �صĹ���
vector<DataPoint> filter_clusters( vector<DataPoint>& clusteringResults,int clusterNum)
{
    vector<DataPoint> filtered_clusters = removeNoiseClusters(clusteringResults);
    double avg_cluster_size = computeAverageClusterSize(filtered_clusters,clusterNum);
    cout << "Average cluster size: " << avg_cluster_size << endl;
    filtered_clusters = filterSmallClusters(filtered_clusters, avg_cluster_size,clusterNum);

    return filtered_clusters;
}
//��������ӳ��
void mapDataPointsToKeyPoints( vector<DataPoint>& clusteringResults,  vector<KeyPoint>& keypoints,  Mat& descriptors, vector<KeyPoint>& filtered_kp, Mat& filtered_des)
{
    filtered_kp.clear(); // ��� filtered_kp
    filtered_des.release(); // �ͷ� filtered_des ���ڴ�
    int num_points = clusteringResults.size();

    // ����ÿ�����ݵ㣬������ӳ��عؼ����������
    for (size_t i = 0; i < num_points; ++i)
    {
        int oldIndex = clusteringResults[i].GetOldIndex();
        // ��ȡ�ؼ��������
        KeyPoint kp = keypoints[oldIndex];
        // ���ؼ�����ӵ� filtered_kp ��
        filtered_kp.push_back(kp);
        // ����Ӧ����������ӵ� filtered_des ��
        filtered_des.push_back(descriptors.row(oldIndex));
    }
}
//����ƥ��
//1 ��ƥ�䣺����ƥ��
//1 ��ƥ�䣺����ƥ��
vector<DMatch> feature_match_ByBF(Mat& des1,Mat& des2,double threshold)
{
    // ʹ�ñ���ƥ������������ƥ��
//    BFMatcher bf(cv::NORM_L2); // ʹ��ŷ�Ͼ������ƥ��
    BFMatcher bf(NORM_HAMMING);//ʹ�ú�������
    vector<vector<DMatch>> matches;
    bf.knnMatch(des1, des2, matches, 2);

    // ��һ��ɸѡ��
    // Ӧ�ñ��ʲ��ԣ������õ�ƥ��
    vector<DMatch> good_matches;
    for (size_t i = 0; i < matches.size(); ++i)
    {
        if (matches[i][0].distance < threshold * matches[i][1].distance)
        {
            good_matches.push_back(matches[i][0]);
        }
    }

    // �Ȱ��վ����matches�е�ƥ��Խ������򡣰��վ������������ǽ���������ʹ���ƥ�䣨�;��룩
    // ����������ָ����ʲô������������
    sort(good_matches.begin(), good_matches.end(), [](DMatch& a, DMatch& b)
    {
        return a.distance < b.distance;
    });

    return good_matches;
}
//ϸƥ��
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
        cv::Mat mask; // RANSAC��������룬ָʾ��Щ�㱻��Ϊ���ڵ�
        T = cv::estimateAffinePartial2D(src_pts, dst_pts, mask, cv::RANSAC);

//        cv::Mat warped_img1;
//        cv::warpAffine(img1, warped_img1, T, img2.size());
//
//        // Merge img1 and img2
//        cv::Mat merged_img;
//        cv::addWeighted(warped_img1, 0.5, img2, 0.5, 0, merged_img);
//        cv::imshow("merged_img", merged_img);
        std::cout << "T: " <<T << std::endl;

        // �� 2x3 ���� T ת��Ϊ 3x3 ����
        T_3x3 = cv::Mat::zeros(3, 3, CV_64FC1);
        T.rowRange(0, 2).copyTo(T_3x3.rowRange(0, 2));

        // �� T_3x3 �����һ������Ϊ [0, 0, 1]
        T_3x3.at<double>(2, 2) = 1;

        std::cout << "ת����ķ���任���� T��\n" << T_3x3 << std::endl;
        // �����ڵ���
        std::cout << "Total Points: " << mask.total() << std::endl;
        std::cout << "Inliers Number: " << countNonZero(mask) << std::endl;
        std::cout << "Inliers Ratio: " << static_cast<double>(cv::countNonZero(mask)) / mask.total() << std::endl;


        //ȫ��ƥ��
        cv::Mat img_matches;
        cv::drawMatches( img1, kp1, img2, kp2, good_matches, img_matches,cv::Scalar(0,0,255),cv::Scalar(0,0,255),vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


        // �����ڵ�
        cv::Mat img_matches_ransac;
        cv::drawMatches(img1, kp1, img2, kp2, good_matches, img_matches_ransac, cv::Scalar(0,0,255),cv::Scalar(0,0,255), mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//        imshow("Matches", img_matches );
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
    // ���� img1 �ı߽�ǵ�
    int height = img1.rows;
    int width = img1.cols;
    cout << height << endl;
    cout << width << endl;
    vector<Point2f> corners = {Point2f(0, 0), Point2f(width, 0), Point2f(width, height), Point2f(0, height)};
    cout << "ԭʼ�ǵ㣺" << endl;
    for (const auto &corner : corners)
    {
        cout << corner << endl;
    }
    // ��ȡ�任��Ľǵ����꣬�任ǰ��Ҫ�����ݽ�������
    vector<Point2f> transformed_corners;
    perspectiveTransform(corners, transformed_corners, T);
    // ���ǵ�����ת��Ϊ��������
    vector<Point> transformed_corners_int;
    for (const auto &corner : transformed_corners)
    {
        transformed_corners_int.push_back(Point(corner.x, corner.y));
    }

    cout << "�任��ǵ�Ϊ��" << endl;
    for (const auto &corner : transformed_corners_int)
    {
        cout << corner << endl;
    }
    // ��������
    int merged_left = min(min_element(transformed_corners_int.begin(), transformed_corners_int.end(), [](const Point2f &p1, const Point2f &p2)
    {
        return p1.x < p2.x;
    })->x, 0);
    // ��������
    int merged_right = max(max_element(transformed_corners_int.begin(), transformed_corners_int.end(), [](const Point2f &p1, const Point2f &p2)
    {
        return p1.x < p2.x;
    })->x, img2.cols);
    // ��������
    int merged_top = min(min_element(transformed_corners_int.begin(), transformed_corners_int.end(), [](const Point2f &p1, const Point2f &p2)
    {
        return p1.y < p2.y;
    })->y, 0);
    // ��������
    int merged_bottom = max(max_element(transformed_corners_int.begin(), transformed_corners_int.end(), [](const Point2f &p1, const Point2f &p2)
    {
        return p1.y < p2.y;
    })->y, img2.rows);
    // �����ںϺ�ͼ��Ŀ�Ⱥ͸߶�
    int merged_width = merged_right - merged_left;
    int merged_height = merged_bottom - merged_top;
    cout << "�����棺" << merged_left << endl;
    cout << "�����棺" << merged_right << endl;
    cout << "�����棺" << merged_top << endl;
    cout << "�����棺" << merged_bottom << endl;
    cout << "��ȣ�" << merged_width << endl;
    cout << "�߶ȣ�" << merged_height << endl;
    // ����ƽ�ƾ���
    Mat m = (Mat_<double>(2, 3) << 1, 0, -merged_left, 0, 1, -merged_top);
    return make_tuple(merged_height, merged_width, m);
}

// ����������
typedef bg::model::d2::point_xy<double> myPoint;

vector<myPoint> transform_corners(Mat img, Mat T)
{
    // ���� img �ı߽�ǵ�
    int height = img.rows;
    int width = img.cols;
    vector<Point2f> corners = {Point2f(0, 0), Point2f(width, 0), Point2f(width, height), Point2f(0, height)};
    cout << "ԭʼ�ǵ㣺" << endl;
    for (const auto &corner : corners)
    {
        cout << corner << endl;
    }
    // ��ȡ�任��Ľǵ����꣬�任ǰ��Ҫ�����ݽ�������
    vector<Point2f> transformed_corners;
    perspectiveTransform(corners, transformed_corners, T);
    // ���ǵ�����ת��Ϊ��������
    vector<myPoint> transformed_corners_int;
    for (const auto &corner : transformed_corners)
    {
        transformed_corners_int.push_back(myPoint(int(corner.x), int(corner.y)));
    }

    cout << "�任��ǵ�Ϊ��" << endl;
    for (const auto &p : transformed_corners_int)
    {
        cout <<  "(" << bg::get<0>(p) << ", " << bg::get<1>(p) << ")"<<endl;
    }

    return transformed_corners_int;
}

//�����ص�����
double overlap(Mat img1_before, Mat img2_before, Mat img1_transform, Mat img2_transform, Mat T, Mat t)
{

    //�������ε�����
    typedef bg::model::polygon<myPoint, false> Polygon;
    typedef bg::model::box<myPoint> Box;
    cout<<"img1_transform_corners"<<endl;
    // ����ͼ��1��ͼ��2�ı任��ǵ�
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



// ��������1��Polygon����
    Polygon poly1, poly2;

    poly1.outer() = boost::assign::list_of<myPoint>(img1_transform_corners[0])(img1_transform_corners[1])(img1_transform_corners[2])(img1_transform_corners[3])(img1_transform_corners[0]);
    poly2.outer() = boost::assign::list_of<myPoint>(img2_transform_corners[0])(img2_transform_corners[1])(img2_transform_corners[2])(img2_transform_corners[3])(img2_transform_corners[0]);

    // �����ص����ֵĵ÷�
    std::list<Polygon> intersectionResult;
    bg::intersection(poly1, poly2, intersectionResult);

    // ��ӡ���
    std::cout << "Intersection result:" << std::endl;
    for (const auto& poly : intersectionResult)
    {
        std::cout << "Polygon:" << std::endl;
        for (const auto& point : poly.outer())
        {
            std::cout << "Point(" << bg::get<0>(point) << ", " << bg::get<1>(point) << ")" << std::endl;
        }

    }
    // ����һ���µĶ���� polygon3
    Polygon polygon3;
    for (const auto& poly : intersectionResult)
    {
        // ���ÿ������ε��⻷�� polygon3 ���⻷��
        for (const auto& point : poly.outer())
        {
            bg::append(polygon3.outer(), point);
        }
    }


    // ���� polygon3 ����Ӿ���
    Box envelope;
    bg::envelope(polygon3, envelope);

    // ��ȡ��Ӿ��ε���С�����������
    myPoint rec_min = envelope.min_corner();
    myPoint rec_max = envelope.max_corner();


    // ��ӡ���
    std::cout << "Intersection result envelope: ";
    int Min_point_x = int( bg::get<0>(rec_min));
    int Min_point_y = int( bg::get<1>(rec_min));
    int Max_point_x = int(bg::get<0>(rec_max));
    int Max_point_y = int(bg::get<1>(rec_max));
    std::cout << "Min point: (" << Min_point_x << ", " << Min_point_y << ")";
    std::cout << ", Max point: (" << Max_point_x << ", " << Max_point_y << ")";
    std::cout << std::endl;

    // ͳ���ص����������صĶ���̶�
    int compute_sum = 0;
    int overlap_sum = 0;
    int effective_sum = 0;
    int alignment_sum = 0;
    int sample = 1;  // ����ָ��
    for (int x = Min_point_x; x < Max_point_x+1; x += sample)
    {
        for (int y = Min_point_y; y < Max_point_y +1; y += sample)
        {
            compute_sum++;
            if(bg::within(myPoint(x, y), polygon3))
            {
                overlap_sum++;
                // �������̶�
                if ((img1_transform.at<uchar>(y, x) >= 250 || img1_transform.at<uchar>(y, x) <= 5) &&
                        (img2_transform.at<uchar>(y, x) >= 250 || img2_transform.at<uchar>(y, x) <= 5))
                {
                    effective_sum++;
                    if (abs(img1_transform.at<uchar>(y, x) - img2_transform.at<uchar>(y, x)) <= 5)
                    {
                        alignment_sum++;
                    }

                }
            }

        }
    }
    cout << "�ص�������㣨����Ϊ��" << sample << "��������������" << compute_sum << endl;
    cout << "ʵ���ص����򣨲���Ϊ��" << sample << "��������������" << overlap_sum << endl;
    cout << "��Ч�ص����򣨲���Ϊ��" << sample << "��������������" << effective_sum << endl;
    cout << "�����ص����򣨲���Ϊ��" << sample << "��������������" << alignment_sum << endl;

//      �������̶ȵ÷�
    double score = (effective_sum > 0) ? (double) alignment_sum / effective_sum : 0.0f;
    cout << "����/��Ч������Ϊ��" << sample << "��: " << score << endl;
    return score;
}


// merge ����
void merge(cv::Mat img1, cv::Mat img2, cv::Mat T)
{
    // �����ںϺ��ͼ�� ROI
    int height, width;
    cv::Mat t;
    std::tie(height, width, t) = ROI(img1, img2, T);
    std::cout << "ƽ�ƾ���Ϊ��\n" << t << std::endl;
    std::cout << "�任��������ǰΪ��\n" << T << std::endl;
    cout<<"t.at<double>(1, 2);"<<t.at<double>(1, 2)<<endl;
    // �����任����
    T.at<double>(0, 2) += t.at<double>(0, 2);  // ����ˮƽƽ�Ʒ���
    T.at<double>(1, 2) += t.at<double>(1, 2);  // ���ô�ֱƽ�Ʒ���

    std::cout << "�任����������Ϊ��\n" << T << std::endl;
    // �Ե�һ��ͼƬ����͸�ӱ任
    Mat transform_img1;
    warpPerspective(img1, transform_img1, T, Size(width, height), INTER_LINEAR, BORDER_CONSTANT, Scalar(205, 205, 205));
//    imshow("Transform_img1", transform_img1);
    // ���� img2 ���ϽǺ��ںϺ�����Ͻ�
    // �÷���任ʵ��ƽ��
    Mat img2_transform;
    warpAffine(img2, img2_transform, t, Size(width, height), INTER_LINEAR, BORDER_CONSTANT, Scalar(205, 205, 205));

    // ������ͼƬ������һ����ʾ
    Mat connect_image;
    hconcat(transform_img1, img2_transform, connect_image);
//    imshow("connect_image", connect_image);
    cv::waitKey(0);

//    ����÷�
    Mat t_3x3 = cv::Mat::zeros(3, 3, CV_64FC1);
    t.rowRange(0, 2).copyTo(t_3x3.rowRange(0, 2));
    // �� t_3x3 �����һ������Ϊ [0, 0, 1]
    t_3x3.at<double>(2, 2) = 1;
    double scores = overlap(img1, img2, transform_img1, img2_transform, T, t_3x3);
    cout << "���ŵ÷֣�" <<setprecision(8)<< scores << endl;
//�Զ����ں�
    Mat merged_image = Mat::zeros(transform_img1.size(), transform_img1.type());
    for (int y = 0; y < merged_image.rows; ++y)
    {
        for (int x = 0; x < merged_image.cols; ++x)
        {
            // ��ȡ����ֵ

            uchar pixel_img1 = transform_img1.at<uchar>(y, x);
            uchar pixel_img2 = img2_transform.at<uchar>(y, x);
            //ԭ��������

//            if(static_cast<int>(pixel_img1)==0||static_cast<int>(pixel_img2)==0)
//            {
//                continue;
//            }
//            if(static_cast<int>(pixel_img1)>static_cast<int>(pixel_img2))
//            {
//                merged_image.at<uchar>(y, x) = pixel_img1;
//            }
//            else
//            {
//                merged_image.at<uchar>(y, x) = pixel_img2;
//            }

            //ͼ������
//            ����һ����ֵ:��205+-5������Ϊ��ɫ��С��200������Ϊ0������Ϊ254
            if(pixel_img1>210)
                pixel_img1=254;
            else if(pixel_img1<200)
            {
                if(pixel_img1==125)
                    pixel_img1=125;
                else
                    pixel_img1=0;
            }

            else
                pixel_img1=205;

            if(pixel_img2>210)
                pixel_img2=254;
            else if(pixel_img2<200)
            {
                if(pixel_img2==125)
                    pixel_img2=125;
                else
                    pixel_img2=0;
            }

            else
                pixel_img2=205;
            if(static_cast<int>(pixel_img1)==0||static_cast<int>(pixel_img2)==0)
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
    // ֱ�ӵ����ں�
//    addWeighted(transform_img1, 1, img2_transform, 1, 0, merged_image);

    show("merged_image", merged_image);
    cv::imwrite("global_map.pgm", merged_image);
    waitKey(0);
}
int main()
{
    // ��ȡ����ͼ��

//    //aces
//    Mat image1 = imread("..\\maps\\aces\\aces_4.pgm", IMREAD_GRAYSCALE);
//    Mat image2 = imread("..\\maps\\aces\\aces_7.pgm", IMREAD_GRAYSCALE);
//    Mat image1 = imread("..\\maps\\aces\\aces_5.pgm", IMREAD_GRAYSCALE);
//    Mat image2 = imread("..\\maps\\aces\\aces_8.pgm", IMREAD_GRAYSCALE);
    // intel
//    Mat image1 = imread("..\\maps\\intel\\intel_5.pgm", IMREAD_GRAYSCALE);
//    Mat image2 = imread("..\\maps\\intel\\intel_10.pgm", IMREAD_GRAYSCALE);
//    Mat image1 = imread("..\\maps\\intel\\intel_3.pgm", IMREAD_GRAYSCALE);
//    Mat image2 = imread("..\\maps\\intel\\intel_51.pgm", IMREAD_GRAYSCALE);

//    Mat image1 = imread("..\\maps\\fr079\\fr079_4.pgm", IMREAD_GRAYSCALE);
//    Mat image2 = imread("..\\maps\\fr079\\fr079_9.pgm", IMREAD_GRAYSCALE);
//     Edmonton
//    Mat image1 = imread("..\\maps\\Edmonton\\edmonton_42.pgm", IMREAD_GRAYSCALE);
//    Mat image2 = imread("..\\maps\\Edmonton\\edmonton_10.pgm", IMREAD_GRAYSCALE);
//     DM
    Mat image1 = imread("..\\maps\\DM1\\DM_5.pgm", IMREAD_GRAYSCALE);
    Mat image2 = imread("..\\maps\\DM1\\DM_9.pgm", IMREAD_GRAYSCALE);
//    Mat image1 = imread("..\\maps\\DM1\\DM_3.pgm", IMREAD_GRAYSCALE);
//    Mat image2 = imread("..\\maps\\DM1\\DM_7.pgm", IMREAD_GRAYSCALE);



    // ���ؼ���ͼ���������
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detectAKAZEKeypointsAndDescriptors(image1, keypoints1, descriptors1);
    detectAKAZEKeypointsAndDescriptors(image2, keypoints2, descriptors2);




    // ���ؼ���ת��Ϊ raw_data
    float* raw_data1 = keypointsToRawData(keypoints1);
    float* raw_data2 = keypointsToRawData(keypoints2);

    // ��ӡ��� raw_data
//    int data_size = keypoints_akaze.size();
//    printRawData(raw_data,data_size *3);

    ClusterAnalysis myClusterAnalysis1;       //Clustering algorithm object declaration.
    ClusterAnalysis myClusterAnalysis2;       //Clustering algorithm object declaration.

    myClusterAnalysis1.Init(raw_data1,DIME_NUM,keypoints1.size(), 13, 3);      //5cm Algorithm initialization.
    myClusterAnalysis2.Init(raw_data2,DIME_NUM,keypoints2.size(), 13, 3);      //10cm Algorithm initialization.
    printf("clusting the data...\n");




    double start = getCurrentTime();
    myClusterAnalysis1.DoDBSCANRecursive();                    //Perform GriT-DBSCAN.
    myClusterAnalysis2.DoDBSCANRecursive();                    //Perform GriT-DBSCAN.

    start = getCurrentTime() - start;
    myClusterAnalysis1.printMessage();
    myClusterAnalysis2.printMessage();
    printf("the GriT-DBSCAN running time is %.4f\n", start);

    vector< DataPoint > clusteringResults1 = myClusterAnalysis1.getDataSets();
    vector< DataPoint > clusteringResults2 = myClusterAnalysis2.getDataSets();
    //��ȡ�������Ĵ����������ﲻ����������+1
    int clusterNum1 = myClusterAnalysis1.getclusterId()+1;
    int clusterNum2 = myClusterAnalysis2.getclusterId()+1;

    // ʹ�� filter_clusters ����ɸѡ��
    vector<DataPoint> filtered_clusters1 = filter_clusters(clusteringResults1,clusterNum1);
    vector<DataPoint> filtered_clusters2 = filter_clusters(clusteringResults2,clusterNum2);

    int filtered_clusterNum1 = computeClusterNum(filtered_clusters1);
    int filtered_clusterNum2 = computeClusterNum(filtered_clusters2);

    cout << "Number of clusters1(after filtered): " << filtered_clusterNum1 << std::endl;
    cout << "Point_Num of clusters1(after filtered): " << filtered_clusters1.size() << std::endl;
    cout << "Number of clusters2(after filtered): " << filtered_clusterNum2 << std::endl;
    cout << "Point_Num of clusters2(after filtered): " << filtered_clusters2.size() << std::endl;



    // ��������ӳ�����
    vector<KeyPoint> filtered_kp1,filtered_kp2;
    Mat filtered_des1,filtered_des2;
    mapDataPointsToKeyPoints(filtered_clusters1,keypoints1,descriptors1,filtered_kp1,filtered_des1);
    mapDataPointsToKeyPoints(filtered_clusters2,keypoints2,descriptors2,filtered_kp2,filtered_des2);
    //��ӡ������
    printResults(image1, keypoints1, clusteringResults1,filtered_clusters1,clusterNum1);
    printResults(image2, keypoints2, clusteringResults2,filtered_clusters2,clusterNum2);

    //����ƥ��
//     ʹ�� feature_match_ByBF ������������ƥ��
    vector<DMatch> matches = feature_match_ByBF(filtered_des1, filtered_des2,0.85);
    cout << "Number of matches: " << matches.size() << endl;
    //ϸƥ�䣺RANSAC���õ��任����T
    Mat T = feature_match_ByRANSAC(image1, filtered_kp1, image2, filtered_kp2, matches);
    //��ͼ�ں�
    merge(image1, image2,  T);
    system("pause");
    return 0;
}
