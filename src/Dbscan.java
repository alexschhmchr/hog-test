import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Dbscan {
    private Mat img;
    private Mat hsvImg;
    private Mat threshImg;

    public Dbscan() {
        hsvImg = new Mat();
        threshImg = new Mat();
    }

    public void loadImg(String fileName) {
        img = Imgcodecs.imread(fileName);
        /*if(hsvImg == null) {
            initMats();
        } else if(hsvImg.size() != img.size()) {
            initMats();
        }*/
    }

    private void initMats() {
        hsvImg = new Mat(img.size(), img.type());
        threshImg = new Mat(img.size(), CvType.CV_8U);
    }

    public void findClusters() {
        findRangeHSV(img);
    }

    private double getNorm(int x1, int y1, int x2, int y2) {
        return Math.abs(Math.sqrt(Math.pow(x2, 2) + Math.pow(y2, 2)) - Math.sqrt(Math.pow(x2, 2) + Math.pow(y2, 2)));
    }

    private void findRangeHSV(Mat img) {
        Imgproc.resize(img, img, new Size(), 0.3, 0.3);
        Imgproc.cvtColor(img, hsvImg, Imgproc.COLOR_BGR2HSV);
        Core.inRange(hsvImg, new Scalar(18, 125, 0), new Scalar(20, 239, 255), threshImg);
        Mat nonZero = new Mat();
        Core.findNonZero(threshImg, nonZero);
        int[] data = new int[(int) nonZero.total() * nonZero.channels()];
        System.out.println(nonZero.channels());
        System.out.println(nonZero.type());
        nonZero.get(0, 0, data);
        int len = data.length/2;
        for (int i = 0; i < len; i++) {
            int trueI = i*2;
            int x = data[trueI];
            int y = data[trueI + 1];
            System.out.println(String.format("x: %d, y: %d", x, y));
        }
        HighGui.imshow(null, threshImg);
        HighGui.waitKey(0);
        System.out.println(String.format("width: %d, height: %d", (int) nonZero.size().width, (int) nonZero.size().height));
        System.out.println(nonZero.total());
    }

}
