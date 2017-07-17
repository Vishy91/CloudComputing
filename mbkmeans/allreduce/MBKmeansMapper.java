package edu.iu.mbkmeans.allreduce;

import edu.iu.harp.example.DoubleArrPlus;
import edu.iu.harp.partition.Partition;
import edu.iu.harp.partition.Table;
import edu.iu.harp.resource.DoubleArray;
import edu.iu.mbkmeans.common.MBKmeansConstants;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.CollectiveMapper;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;
import java.util.Random;

/**
 * Created by cc on 4/20/17.
 */
public class MBKmeansMapper extends CollectiveMapper<String, String, Object, Object> {

    private int numMappers;
    private int vectorSize;
    private int iteration;
    private int sizeBatch;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        LOG.info("start setup" + new SimpleDateFormat("yyyyMMdd_HHmmss").format(Calendar.getInstance().getTime()));
        long startTime = System.currentTimeMillis();
        Configuration configuration = context.getConfiguration();
        numMappers = configuration.getInt(MBKmeansConstants.NUM_MAPPERS, 10);
        vectorSize = configuration.getInt(MBKmeansConstants.VECTOR_SIZE, 20);
        iteration = configuration.getInt(MBKmeansConstants.NUM_ITERATONS, 1);
        sizeBatch = configuration.getInt(MBKmeansConstants.BATCH_SIZE, 10);
        long endTime = System.currentTimeMillis();
        LOG.info("config (ms) :" + (endTime - startTime));
    }

    protected void mapCollective( KeyValReader reader, Context context) throws IOException, InterruptedException {
        LOG.info("Start collective mapper.");
        long startTime = System.currentTimeMillis();
        List<String> pointFiles = new ArrayList<String>();
        while (reader.nextKeyValue()) {
            String key = reader.getCurrentKey();
            String value = reader.getCurrentValue();
            LOG.info("Key: " + key + ", Value: " + value);
            pointFiles.add(value);
        }
        Configuration conf = context.getConfiguration();
        runMBKmeans(pointFiles, conf, context);
        LOG.info("Total iterations in master view: " + (System.currentTimeMillis() - startTime));
    }

    private void broadcastCentroids( Table<DoubleArray> cenTable) throws IOException{
        //broadcast centroids
        boolean isSuccess = false;
        try {
            isSuccess = broadcast("main", "broadcast-centroids", cenTable, this.getMasterID(),false);
        } catch (Exception e) {
            LOG.error("Fail to bcast.", e);
        }
        if (!isSuccess) {
            throw new IOException("Fail to bcast");
        }
    }
        /**

        First we load the centroids
        And every partition in the centoid table, we will use the last element to store the number of points
        which are clustered to the particular partitionID

        **/

    private void runMBKmeans(List<String> fileNames, Configuration conf, Context context) throws IOException {

        Table<DoubleArray> centeroidT = new Table<>(0, new DoubleArrPlus());
        if (this.isMaster()) {
            loadCentroids(centeroidT, vectorSize, conf.get(MBKmeansConstants.CFILE), conf);
        }

        System.out.println("Centroids are loaded");
        printTable(centeroidT);

        //broadcast centroids
        broadcastCentroids(centeroidT);


        //after broadcasting
        System.out.println("After brodcasting centroids");
        printTable(centeroidT);

        //load data
        ArrayList<DoubleArray> dataPoints = loadData(fileNames, vectorSize, conf);

        Table<DoubleArray> previousCenTable =  null;
        ArrayList<DoubleArray> batchPoints = null;

        //iterations
        for(int iter=0; iter < iteration; iter++){
            previousCenTable =  centeroidT;
            centeroidT = new Table<>(0, new DoubleArrPlus());
            batchPoints = generateBatch(dataPoints);

            System.out.println("Iteraton No."+iter);

            //compute new partial centroid table using previousCenTable and data points
            //computation(centeroidT, previousCenTable, dataPoints);

            //compute new partial centroid table using previousCenTable and batch points
            computation(centeroidT, previousCenTable, batchPoints);

            //AllReduce;
            /****************************************/
            allreduce("main", "allreduce_"+iter, centeroidT);
            //we can calculate new centroids
            //calculateCentroids(centeroidT);

            //calculate new centroids
            calculateCentroids(centeroidT, previousCenTable);
            /****************************************/

            printTable(centeroidT);

        }
        //output results
        if(this.isMaster()){
            outputCentroids(centeroidT,  conf,   context);
        }

    }


    private void computation(Table<DoubleArray> cenTable, Table<DoubleArray> previousCenTable, ArrayList<DoubleArray> dataPoints){
        double err=0;
        for(DoubleArray aPoint: dataPoints){
            //for each data point, find the nearest centroid
            double minDist = -1;
            double tempDist = 0;
            int nearestPartitionID = -1;
            for(Partition ap: previousCenTable.getPartitions()) {
                DoubleArray aCentroid = (DoubleArray) ap.get();
                tempDist = calcEucDistSquare(aPoint, aCentroid, vectorSize);
                if(minDist == -1 || tempDist < minDist) {
                    minDist = tempDist;
                    nearestPartitionID = ap.id();
                }
            }
            err+=minDist;

            //for the certain data point, found the nearest centroid.
            // add the data to a new cenTable.
            double[] partial = new double[vectorSize+1];
            for(int j=0; j < vectorSize; j++){
                partial[j] = aPoint.get()[j];
            }
            partial[vectorSize]=1;

            if(cenTable.getPartition(nearestPartitionID) == null){
                Partition<DoubleArray> tmpAp = new Partition<DoubleArray>(nearestPartitionID, new DoubleArray(partial, 0, vectorSize+1));
                cenTable.addPartition(tmpAp);

            }else{
                Partition<DoubleArray> apInCenTable = cenTable.getPartition(nearestPartitionID);
                for(int i=0; i < vectorSize +1; i++){
                    apInCenTable.get().get()[i] += partial[i];
                }
            }
        }
        System.out.println("Error: "+err);
    }


    //output centroids
    private void outputCentroids(Table<DoubleArray>  cenTable,Configuration conf, Context context){
        String output="";
        for( Partition<DoubleArray> ap: cenTable.getPartitions()){
            double res[] = ap.get().get();
            for(int i=0; i<vectorSize;i++)
                output+= res[i]+"\t";
            output+="\n";
        }
        try {
            context.write(null, new Text(output));
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (InterruptedException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }


    /**
     * This function updates the centroids ad prints the centroid table
     * @param cenTable
     */
    private void calculateCentroids( Table<DoubleArray> cenTable)
    {
        for( Partition<DoubleArray> partialCenTable: cenTable.getPartitions()){
            double[] cen = partialCenTable.get().get();
            for(int i = 0; i < vectorSize; i++){
                cen[i] /= cen[vectorSize];
            }
            cen[vectorSize] = 0;
        }
        System.out.println("New centroids are calculated. ");
        printTable(cenTable);
    }


    /**
     * This function calculates the new centroids and prints the table.
     *
     * @param centroidTable
     *
     * @param previouscentable
     */
    private void calculateCentroids(Table<DoubleArray> centroidTable, Table<DoubleArray> previouscentable){
        for( Partition<DoubleArray> unit: centroidTable.getPartitions())
        {

            Partition<DoubleArray> unit1 = previouscentable.getPartition(unit.id());
            double[] doubles,doubles1;
            doubles = unit.get().get();
            doubles1 = unit1.get().get();
            doubles[vectorSize] += doubles1[vectorSize];

            for (int i = 0; i < vectorSize; i++)
            {
                doubles[i] = (doubles1[i]*doubles1[vectorSize] + doubles[i]) / doubles[vectorSize];
            }
            System.out.println("data after calculating the new centroids");
            printTable(centroidTable);
        }
    }

    //generate mini batch randomly from dataPoints
    private ArrayList<DoubleArray> generateBatch(ArrayList<DoubleArray> dataPoints) {

        ArrayList<DoubleArray> data = new ArrayList<>();
        Random random = new Random();
        int points = -1;
        // we add "sizebatch"random points from the data points are added to the arraylist "data"
        // and we return the arraylist "data"
        for(int j = 0; j <= sizeBatch; j++)
        {
            points = random.nextInt(dataPoints.size());
            data.add(dataPoints.get(points));
        }
        System.out.println("Generated the mini-batch");
        return data;
    }


    //calculate Euclidean distance.
    private double calcEucDistSquare(DoubleArray aPoint, DoubleArray otherPoint, int vectorSize){
        double dist=0;
        for(int i=0; i < vectorSize; i++){
            dist += Math.pow(aPoint.get()[i]-otherPoint.get()[i],2);
        }
        return Math.sqrt(dist);
    }

    //load centroids from HDFS
    private void loadCentroids( Table<DoubleArray> cenTable, int vectorSize,  String cFileName, Configuration configuration) throws IOException{
        Path cPath = new Path(cFileName);
        FileSystem fs = FileSystem.get(configuration);
        FSDataInputStream in = fs.open(cPath);
        BufferedReader br = new BufferedReader( new InputStreamReader(in));
        String line="";
        String[] vector=null;
        int partitionId=0;
        while((line = br.readLine()) != null){
            vector = line.split("\\s+");
            if(vector.length != vectorSize){
                System.out.println("Errors while loading centroids .");
                System.exit(-1);
            }else{
                double[] aCen = new double[vectorSize+1];

                for(int i=0; i<vectorSize; i++){
                    aCen[i] = Double.parseDouble(vector[i]);
                }
                aCen[vectorSize]=0;
                Partition<DoubleArray> ap = new Partition<DoubleArray>(partitionId, new DoubleArray(aCen, 0, vectorSize+1));
                cenTable.addPartition(ap);
                partitionId++;
            }
        }
    }
    //load data form HDFS
    private ArrayList<DoubleArray>  loadData(List<String> fileNames,  int vectorSize, Configuration conf) throws IOException{
        ArrayList<DoubleArray> data = new  ArrayList<DoubleArray> ();
        for(String filename: fileNames){
            FileSystem fs = FileSystem.get(conf);
            Path dPath = new Path(filename);
            FSDataInputStream in = fs.open(dPath);
            BufferedReader br = new BufferedReader( new InputStreamReader(in));
            String line="";
            String[] vector=null;
            while((line = br.readLine()) != null){
                vector = line.split("\\s+");

                if(vector.length != vectorSize){
                    System.out.println("Errors while loading data.");
                    System.exit(-1);
                }else{
                    double[] aDataPoint = new double[vectorSize];

                    for(int i=0; i<vectorSize; i++){
                        aDataPoint[i] = Double.parseDouble(vector[i]);
                    }
                    DoubleArray da = new DoubleArray(aDataPoint, 0, vectorSize);
                    data.add(da);
                }
            }
        }
        return data;
    }

    //for testing
    private void printTable(Table<DoubleArray> dataTable){
        for( Partition<DoubleArray> ap: dataTable.getPartitions()){

            double res[] = ap.get().get();
            System.out.print("ID: "+ap.id() + ":");
            for(int i=0; i<res.length;i++)
                System.out.print(res[i]+"\t");
            System.out.println();
        }
    }
}