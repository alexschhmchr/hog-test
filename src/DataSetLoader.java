import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONTokener;
import org.opencv.core.Point;
import org.opencv.core.Rect;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

public class DataSetLoader {
    private static final String CORN_NAME = "corn";

    public DataSetLoader() {

    }

    public DataSet load(String datasetPath) throws IOException{
        ArrayList<DataSet.Labels> dataList = new ArrayList<>();
        Path cornPath = Paths.get(datasetPath);
        DirectoryStream<Path> pathStream = Files.newDirectoryStream(cornPath, "*.json");
        pathStream.forEach(path -> {
            try {
                String fileName = path.getFileName().toString();
                String imgName = fileName.substring(0, fileName.indexOf('.'));
                Path imgPath = path.resolveSibling(imgName + ".jpg");
                ArrayList<Rect> labelList = new ArrayList<>();
                BufferedReader reader = Files.newBufferedReader(path);
                JSONTokener tokener = new JSONTokener(reader);
                JSONObject object = (JSONObject) tokener.nextValue();
                JSONArray shapes = object.getJSONArray("shapes");
                for(int i = 0; i < shapes.length(); i++) {
                    JSONObject shape = shapes.getJSONObject(i);
                    JSONArray points = shape.getJSONArray("points");
                    JSONArray point1 = points.getJSONArray(0);
                    JSONArray point2 = points.getJSONArray(1);
                    Rect window = new Rect(new Point(point1.getInt(0), point1.getInt(1)), new Point(point2.getInt(0), point2.getInt(1)));
                    labelList.add(window);
                }
                DataSet.Labels labels = new DataSet.Labels(labelList, imgPath);
                dataList.add(labels);
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
        DataSet dataSet = new DataSet(dataList, cornPath);
        return dataSet;
    }


}
