import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.security.SecureClassLoader;
import java.util.ArrayList;

public class DBScan {
    private Mat img;
    private Mat hsvImg;
    private Mat threshImg;

    private double maxDistance;
    private int minPoints;

    public DBScan(double maxDistance, int minPoints) {
        hsvImg = new Mat();
        threshImg = new Mat();
        this.maxDistance = maxDistance;
        this.minPoints = minPoints;
    }

    public void loadImg(String fileName) {
        img = Imgcodecs.imread(fileName);
        Imgproc.resize(img, img, new Size(), 0.2, 0.2);
    }

    public void showClusters(ArrayList<Cluster> clusters) {
        for(Cluster cluster : clusters) {
            Imgproc.rectangle(img, cluster.getClusterRect(), new Scalar(255, 0, 0), 2);
        }
        HighGui.imshow("", img);
        HighGui.waitKey(0);
    }

    public ArrayList<Cluster> findClusters() {
        ArrayList<Cluster> clusterList = new ArrayList<>();
        ArrayList<Point> pointList = findRangeHSV(img);
        while(!pointList.isEmpty()) {
            ArrayList<Point> clusterPoints = new ArrayList<>();
            Point center = pointList.remove(0);
            findNeighbours(center, pointList, clusterPoints);
            if(clusterPoints.size() > minPoints) {
                Cluster cluster = new Cluster(clusterPoints);
                clusterList.add(cluster);
            }
        }
        System.out.println(clusterList.size());
        return clusterList;
    }

    private void findNeighbours(Point center, ArrayList<Point> pointList, ArrayList<Point> clusterPoints) {
        for(int i = 0; i < pointList.size(); i++) {
            Point p = pointList.get(i);
            double distance = Math.abs(getNorm(center, p));
            if(distance < maxDistance) {
                clusterPoints.add(p);
                pointList.remove(i--);
                findNeighbours(p, pointList, clusterPoints);
            }
        }
    }

    private double getNorm(Point point1, Point point2) {
        return getNorm((int) point1.x, (int) point1.y, (int) point2.x, (int) point2.y);
    }

    private double getNorm(int x1, int y1, int x2, int y2) {
        int x = x1 - x2;
        int y = y1 - y2;
        double distance = Math.sqrt(Math.pow(x, 2) + Math.pow(y, 2));
        return distance;
    }

    private ArrayList<Point> findRangeHSV(Mat img) {
        Imgproc.cvtColor(img, hsvImg, Imgproc.COLOR_BGR2HSV);
        Core.inRange(hsvImg, new Scalar(18, 125, 0), new Scalar(20, 239, 255), threshImg);
        Mat nonZero = new Mat();
        Core.findNonZero(threshImg, nonZero);
        int[] data = new int[(int) nonZero.total() * nonZero.channels()];
        nonZero.get(0, 0, data);
        int len = data.length/2;
        ArrayList<Point> pointList = new ArrayList<>();
        for (int i = 0; i < len; i++) {
            int trueI = i*2;
            int x = data[trueI];
            int y = data[trueI + 1];
            pointList.add(new Point(x, y));
        }
        return pointList;
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        DBScan dbscan = new DBScan(7, 150);
        dbscan.loadImg("C:\\Users\\Alex\\Desktop\\Bilder_Korner_original_20180411\\mais\\DSC_0922.JPG");
        ArrayList<Cluster> clusters = dbscan.findClusters();
        dbscan.showClusters(clusters);
    }

    public static class Cluster {
        private ArrayList<Point> clusterPoints;
        private Rect clusterRect;

        private Cluster(ArrayList<Point> clusterPoints) {
            this.clusterPoints = clusterPoints;
        }

        public Rect getClusterRect() {
            if(clusterRect == null) {
                clusterRect = calculateClusterRect();
            }
            return clusterRect;
        }

        private Rect calculateClusterRect() {
            double leftPoint = clusterPoints.get(0).x;
            double rightPoint = 0;
            double topPoint = clusterPoints.get(0).y;
            double bottomPoint = 0;
            for(Point p : clusterPoints) {
                if(p.x < leftPoint) {
                    leftPoint = p.x;
                } else if(p.x > rightPoint) {
                    rightPoint = p.x;
                }
                if(p.y < topPoint) {
                    topPoint = p.y;
                } else if(p.y > bottomPoint) {
                    bottomPoint = p.y;
                }
            }
            int x = (int) Math.round(leftPoint);
            int y = (int) Math.round(topPoint);
            int width = (int) (rightPoint - leftPoint);
            int height = (int) (bottomPoint - topPoint);
            return new Rect(x, y, width, height);
        }
    }
}
