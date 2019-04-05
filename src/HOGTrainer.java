import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
import org.opencv.ml.TrainData;
import org.opencv.objdetect.HOGDescriptor;

import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Stream;

public class HOGTrainer {
    private static final int BLOCK_PER_LINE = 4;
    private static final int CELLS_PER_BLOCK_LINE = 2;
    private static final int BLOCK_STRIDE_IN_CELLS = 1;
    private static final int N_BINS = 8;
    private static final String NEG_PATH_STRING = "C:\\Users\\Alex\\Desktop\\Bilder_Korner_original_20180411\\mehl";
    private static final Size NEG_IMG_MARGIN = new Size(1000, 500);
    private static final Size IMG_SIZE = new Size(6016, 4016);
    private static final int POS_LABEL = 1;
    private static final int NEG_LABEL = -1;

    private DataSet dataSet;
    private HOGDescriptor hog;


    private Size cellSize;
    private Size blockSize;
    private Size windowSize;
    private Size blockStride;
    private Size winStride;


    public HOGTrainer(DataSet dataSet) {
        this.dataSet = dataSet;
    }

    public void configure() {
        Size maxWindow = dataSet.getMaxSize();

        initHOGSizes(maxWindow);

        hog = new HOGDescriptor(windowSize, blockSize, blockStride, cellSize, N_BINS);
    }

    public void train() throws IOException {
        MatOfFloat posSamples = computeDescriptor();
        MatOfFloat negSamples = computeNegative(450);
        MatOfFloat samples = new MatOfFloat();
        Core.hconcat(Arrays.asList(posSamples, negSamples), samples);
        System.out.println(posSamples.rows());
        Mat labels = new Mat(posSamples.cols(), 1,CvType.CV_32S, new Scalar(POS_LABEL));
        Mat negLabels = new Mat(negSamples.cols(), 1, CvType.CV_32S, new Scalar(NEG_LABEL));
        labels.push_back(negLabels);
        Core.transpose(labels, labels);
        System.out.println(labels.cols());
        System.out.println(samples.cols());
        SVM svm = SVM.create();
        svm.setKernel(SVM.LINEAR);
        svm.setType(SVM.EPS_SVR);
        svm.setP(0.1);
        svm.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER + TermCriteria.EPS, 1000, 1e-3));
        System.out.println("training");
        svm.train(samples, Ml.COL_SAMPLE, labels);
        hog.setSVMDetector(svm.getSupportVectors());
        detectTest();
    }

    private void detectTest() {
        Mat testImg = Imgcodecs.imread("C:\\Users\\Alex\\Desktop\\Bilder_Korner_original_20180411\\mais\\DSC_0927.JPG");
        Mat gray = new Mat();
        Imgproc.cvtColor(testImg, gray, Imgproc.COLOR_BGR2GRAY);
        MatOfPoint points = new MatOfPoint();
        MatOfDouble weights = new MatOfDouble();
        hog.detect(gray, points, weights, 0, winStride);
        Point[] pointsArray = points.toArray();
        double[] weightsArray = weights.toArray();
        Rect window = new Rect(0, 0, (int) windowSize.width, (int) windowSize.height);
        for(int i = 0; i < pointsArray.length; i++) {
            window.x = (int) pointsArray[i].x;
            window.y = (int) pointsArray[i].y;
            if(weightsArray[i] > 2.7) {
                Imgproc.rectangle(testImg, window, new Scalar(0, 255, 0), 3);
                Imgproc.putText(testImg, Double.toString(Math.round(weightsArray[i] * 100) / 100d), pointsArray[i], Imgproc.FONT_HERSHEY_SIMPLEX, 3, new Scalar(0, 255, 0), 3);
            }
            System.out.println(weightsArray[i]);
        }
        Imgproc.resize(testImg, testImg, new Size(), 0.25, 0.25);
        HighGui.imshow("", testImg);
        HighGui.waitKey(0);
    }

    private void initHOGSizes(Size maxSize) {
        int cellsPerLine = BLOCK_PER_LINE  * CELLS_PER_BLOCK_LINE;

        int cellWidth = (int) Math.ceil(maxSize.width / cellsPerLine);
        int cellHeight = cellWidth;
        cellSize = new Size(cellWidth, cellHeight);

        int blockWidth = cellWidth * CELLS_PER_BLOCK_LINE;
        int blockHeight = blockWidth;
        blockSize = new Size(blockWidth, blockHeight);

        int blockStrideX = cellWidth * BLOCK_STRIDE_IN_CELLS;
        int blockStrideY = cellHeight * BLOCK_STRIDE_IN_CELLS;
        blockStride = new Size(blockStrideX, blockStrideY);

        int windowWidth = blockWidth * BLOCK_PER_LINE;
        int windowHeight = (int) (Math.ceil(maxSize.height / blockHeight) * blockHeight);
        windowSize = new Size(windowWidth, windowHeight);

        int winStrideX = blockWidth;
        int winStrideY = blockHeight;
        winStride = new Size(winStrideX, winStrideY);
    }

    private MatOfFloat computeDescriptor() {
        ArrayList<Mat> descList = new ArrayList<>();

        ArrayList<DataSet.Labels> data = dataSet.getDataList();
        double halfWX = windowSize.width/2;
        double halfWY = windowSize.height/2;

        Rect roiRect = new Rect();
        roiRect.width = (int) windowSize.width;
        roiRect.height = (int) windowSize.height;

        //Mat testImg = new Mat();
        for(DataSet.Labels labels : data) {
            Mat img = Imgcodecs.imread(labels.getImagePath().toString(), Imgcodecs.IMREAD_GRAYSCALE);
            //Imgproc.cvtColor(img, testImg, Imgproc.COLOR_BGR2GRAY);
            System.out.println(labels.getImagePath().toString());
            Size imgSize = img.size();
            ArrayList<Point> centerList = labels.getCenterList();
            for(Point center : centerList) {
                roiRect.x = (int) (center.x - halfWX);
                roiRect.y = (int) (center.y - halfWY);
                System.out.println(roiRect);
                if(isRectInSize(imgSize, roiRect)) {
                    MatOfFloat desc = new MatOfFloat();
                    Mat roiImg = new Mat(img, roiRect);
                    Imgproc.rectangle(img, roiRect, new Scalar(255), 2);
                    hog.compute(roiImg, desc, blockStride);
                    descList.add(desc);
                    roiImg.release();
                }
            }
            img.release();
            /**Imgproc.resize(img, img, new Size(), 0.25, 0.25);
            HighGui.imshow("", img);
            HighGui.waitKey(0);**/
        }
        MatOfFloat posSamples = new MatOfFloat();
        Core.hconcat(descList, posSamples);
        return posSamples;
    }

    private MatOfFloat computeNegative(int posDescSize) throws IOException{
        ArrayList<Mat> descList = new ArrayList<>();
        Path negPath = Paths.get(NEG_PATH_STRING);
        Stream<Path> pathStream = Files.list(negPath);
        int imgCount = (int) Files.list(negPath).count();
        int imgNeeded = getImgsForNeg(imgCount);
        long start = System.currentTimeMillis();
        pathStream.limit(imgNeeded).forEach(path -> {
            Mat negImg = Imgcodecs.imread(path.toString(), Imgcodecs.IMREAD_GRAYSCALE);
            Rect winRect = new Rect((int) NEG_IMG_MARGIN.width, (int) NEG_IMG_MARGIN.height, (int) windowSize.width, (int) windowSize.height);
            int rightBorder = negImg.cols() - (int) NEG_IMG_MARGIN.width;
            int buttomBorder = negImg.rows() - (int) NEG_IMG_MARGIN.height;
            for (; winRect.y + winRect.height < buttomBorder && descList.size() < posDescSize; winRect.y += windowSize.height) {
                for (; winRect.x + winRect.width < rightBorder && descList.size() < posDescSize; winRect.x += windowSize.width) {
                    Mat roiImg = new Mat(negImg, winRect);
                    MatOfFloat desc = new MatOfFloat();
                    hog.compute(roiImg, desc);
                    descList.add(desc);
                    roiImg.release();
                }
            }
            negImg.release();
        });
        long benchmark = System.currentTimeMillis()- start;
        System.out.println(benchmark);
        MatOfFloat negDescs = new MatOfFloat();
        Core.hconcat(descList, negDescs);
        System.out.println(negDescs);
        return negDescs;
    }

    private int getImgsForNeg(int imgCount) {
        double roiWidth = IMG_SIZE.width - NEG_IMG_MARGIN.width * 2;
        double roiHeight = IMG_SIZE.height - NEG_IMG_MARGIN.height * 2;
        int colsInROI = (int) Math.floor(roiWidth/windowSize.width);
        int rowsInROI = (int) Math.floor(roiHeight/windowSize.height);
        return colsInROI * rowsInROI * imgCount;
    }

    private boolean isRectInSize(Size size, Rect rect) {
        if(rect.x >= 0 && rect.y >= 0 && rect.x + rect.width <= size.width && rect.y + rect.height <= size.height) {
            return true;
        } else {
            return false;
        }
    }
}
