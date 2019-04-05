import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;

import java.io.IOException;

public class App {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        DataSetLoader loader = new DataSetLoader();
        try {
            DataSet dataSet = loader.load("C:\\Users\\Alex\\IdeaProjects\\grain-swpt\\dataset\\corn");
            HOGTrainer trainer = new HOGTrainer(dataSet);
            trainer.configure();
            trainer.train();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
