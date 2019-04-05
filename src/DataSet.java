import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Optional;

public class DataSet {
    private Path dataPath;
    private ArrayList<Labels> dataList;
    private Optional<Size> maxSize = Optional.empty();

    public DataSet(ArrayList<Labels> dataList, Path dataPath) {
        this.dataList = dataList;
        this.dataPath = dataPath;
    }

    public Size getMaxSize() {
        if(!maxSize.isPresent()) {
            Size maxSize = new Size();
            for (Labels labels : dataList) {
                Size size = labels.getMaxSize();
                if (size.width > maxSize.width) {
                    maxSize.width = size.width;
                }
                if (size.height > maxSize.height) {
                    maxSize.height = size.height;
                }
            }
            this.maxSize = Optional.of(maxSize);
        }
        return maxSize.get();
    }

    public ArrayList<Labels> getDataList() {
        return dataList;
    }

    public static class Labels {
        private ArrayList<Rect> labelList;
        private ArrayList<Point> centerList = new ArrayList<>();
        private Path imagePath;

        public Labels(ArrayList<Rect> labelList, Path imagePath) {
            this.labelList = labelList;
            this.imagePath = imagePath;
        }

        private Size getMaxSize() {
            Size maxSize = new Size();
            for(Rect rect : labelList) {
                if(rect.width > maxSize.width) {
                    maxSize.width = rect.width;
                }
                if(rect.height > maxSize.height) {
                    maxSize.height = rect.height;
                }
            }
            return maxSize;
        }

        public ArrayList<Point> getCenterList() {
            if(centerList.size() != labelList.size()) {
                calcuteCenters();
            }
            return centerList;
        }

        private void calcuteCenters() {
            for(Rect rect: labelList) {
                double centerX = rect.x + rect.width/2d;
                double centerY = rect.y + rect.height/2d;
                Point center = new Point(centerX, centerY);
                centerList.add(center);
            }
        }

        public ArrayList<Rect> getLabelList() {
            return labelList;
        }

        public Path getImagePath() {
            return imagePath;
        }
    }

}
