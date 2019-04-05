import org.opencv.core.Size;
import org.opencv.objdetect.HOGDescriptor;

public class HogGrainDetector {
    private HOGDescriptor descriptor;
    private DataSet dataSet;


    public HogGrainDetector(DataSet dataSet) {
        this.dataSet = dataSet;
        descriptor = new HOGDescriptor();
    }

    public void computeDescriptor() {
        Size windowSize = dataSet.getMaxSize();
    }


}
