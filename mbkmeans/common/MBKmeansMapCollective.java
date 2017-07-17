package edu.iu.mbkmeans.common;

import edu.iu.fileformat.MultiFileInputFormat;
import edu.iu.mbkmeans.common.MBKmeansConstants;
import edu.iu.mbkmeans.common.MBKmeansMapCollective;
import edu.iu.mbkmeans.common.Utils;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;
import java.net.URISyntaxException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.concurrent.ExecutionException;

/**
 * Created by cc on 4/20/17.
 */
public class MBKmeansMapCollective extends Configured implements Tool {

    public static void main(String [] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new MBKmeansMapCollective(), args);
        System.exit(res);
    }

    @Override
    public int run(String[] args) throws Exception {
        if (args.length < 8) {
            System.err.println("Usage: Mini-BatchKmeansMapCollective <numOfDataPoints> <num of Centroids> "
                    + "<size of vector> <size of batch> <number of map tasks> <number of iteration> <workDir> <localDir> <communication operation>\n"
                    + "<communication operation> includes:\n  "
                    +  "[allreduce]: use allreduce operation to synchronize centroids \n");
            ToolRunner.printGenericCommandUsage(System.err);
            return -1;
        }

        int numOfDataPoints = Integer.parseInt(args[0]);
        int numCentroids = Integer.parseInt(args[1]);
        int sizeOfVector = Integer.parseInt(args[2]);
        int sizeOfBatch = Integer.parseInt(args[3]);
        int numMapTasks = Integer.parseInt(args[4]);
        int numIteration = Integer.parseInt(args[5]);
        String workDir = args[6];
        String localDir = args[7];
        String operation = args[8];

        System.out.println( "Number of Map Tasks = " + numMapTasks);
        System.out.println("Len : " + args.length);
        System.out.println("Args : ");
        for(String arg: args){
            System.out.print(arg+";");
        }
        System.out.println();

        launch(numOfDataPoints, numCentroids, sizeOfVector, sizeOfBatch, numMapTasks, numIteration, workDir, localDir, operation);
        System.out.println("Harp Mini-Batch Kmeans Completed");
        return 0;
    }

    void launch(int numOfDataPoints, int numCentroids, int sizeOfVector, int sizeOfBatch, int numMapTasks, int numIteration, String workDir, String localDir, String operation)
            throws IOException, URISyntaxException, InterruptedException, ExecutionException, ClassNotFoundException {

        Configuration configuration = getConf();
        Path workDirPath = new Path(workDir);
        FileSystem fs = FileSystem.get(configuration);
        Path dataDir = new Path(workDirPath, "data");
        Path cenDir = new Path(workDirPath, "centroids");
        Path outDir = new Path(workDirPath, "out");
        if (fs.exists(outDir)) {
            fs.delete(outDir, true);
        }
        fs.mkdirs(outDir);

        System.out.println("Generate data.");
        edu.iu.mbkmeans.common.Utils.generateData(numOfDataPoints, sizeOfVector, numMapTasks, fs, localDir, dataDir);

        int JobID = 0;
        Utils.generateInitialCentroids(numCentroids, sizeOfVector, configuration, cenDir, fs, JobID);

        long startTime = System.currentTimeMillis();

        runMBKMeans(numOfDataPoints,numCentroids, sizeOfVector, sizeOfBatch,
                numIteration, JobID,  numMapTasks, configuration,
                workDirPath, dataDir, cenDir, outDir, operation);
        long endTime = System.currentTimeMillis();
        System.out.println("Total Mini-Batch K-means Execution Time: "+ (endTime - startTime));
    }

    private void runMBKMeans(int numOfDataPoints, int numCentroids, int vectorSize, int sizeBatch,
                           int numIterations, int JobID, int numMapTasks, Configuration configuration,
                           Path workDirPath, Path dataDir, Path cDir, Path outDir, String operation)
            throws IOException,URISyntaxException, InterruptedException,ClassNotFoundException {

        System.out.println("Starting Job");
        long jobSubmitTime;
        boolean jobSuccess = true;
        int jobRetryCount = 0;

        do {
            // ----------------------------------------------------------------------
            jobSubmitTime = System.currentTimeMillis();
            System.out.println("Start Job#" + JobID + " "+ new SimpleDateFormat("HH:mm:ss.SSS").format(Calendar.getInstance().getTime()));

            Job kmeansJob = configureMBKMeansJob(numOfDataPoints,	numCentroids, vectorSize, sizeBatch, numMapTasks,
                    configuration, workDirPath, dataDir,cDir, outDir, JobID, numIterations, operation);

            System.out.println("| Job#"+ JobID+ " configure in "+ (System.currentTimeMillis() - jobSubmitTime)+ " miliseconds |");

            // ----------------------------------------------------------
            jobSuccess =kmeansJob.waitForCompletion(true);

            System.out.println("end Jod#" + JobID + " "
                    + new SimpleDateFormat("HH:mm:ss.SSS")
                    .format(Calendar.getInstance().getTime()));
            System.out.println("| Job#"+ JobID + " Finished in "
                    + (System.currentTimeMillis() - jobSubmitTime)
                    + " miliseconds |");

            // ---------------------------------------------------------
            if (!jobSuccess) {
                System.out.println("Mini-Batch KMeans Job failed. Job ID:"+ JobID);
                jobRetryCount++;
                if (jobRetryCount == 3) {
                    break;
                }
            }else{
                break;
            }
        } while (true);
    }

    private Job configureMBKMeansJob(int numOfDataPoints, int numCentroids, int vectorSize, int sizeBatch,
                                   int numMapTasks, Configuration configuration,Path workDirPath, Path dataDir, Path cDir,
                                   Path outDir, int jobID, int numIterations, String operation) throws IOException, URISyntaxException {

        Job job = Job.getInstance(configuration, "mbkmeans_job_"+ jobID);
        Configuration jobConfig = job.getConfiguration();
        Path jobOutDir = new Path(outDir, "mbkmeans_out_" + jobID);
        FileSystem fs = FileSystem.get(configuration);
        if (fs.exists(jobOutDir)) {
            fs.delete(jobOutDir, true);
        }
        FileInputFormat.setInputPaths(job, dataDir);
        FileOutputFormat.setOutputPath(job, jobOutDir);

        Path cFile = new Path(cDir, MBKmeansConstants.CENTROID_FILE_PREFIX + jobID);
        System.out.println("Centroid File Path: "+ cFile.toString());
        jobConfig.set(MBKmeansConstants.CFILE,cFile.toString());
        jobConfig.setInt(MBKmeansConstants.JOB_ID, jobID);
        jobConfig.setInt(MBKmeansConstants.NUM_ITERATONS, numIterations);
        job.setInputFormatClass(MultiFileInputFormat.class);
        job.setJarByClass(MBKmeansConstants.class);


        //use different kinds of mappers
        if(operation.equalsIgnoreCase("allreduce")){
            job.setMapperClass(edu.iu.mbkmeans.allreduce.MBKmeansMapper.class);
        //}else if (operation.equalsIgnoreCase("regroup-allgather")){
        //    job.setMapperClass(edu.iu.mbkmeans.regroupallgather.MBKmeansMapper.class);
        //}else if(operation.equalsIgnoreCase("broadcast-reduce")){
        //    job.setMapperClass(edu.iu.mbkmeans.bcastreduce.MBKmeansMapper.class);
        //}else if(operation.equalsIgnoreCase("push-pull")){
        //    job.setMapperClass(edu.iu.mbkmeans.pushpull.MBKmeansMapper.class);
        }else{//by default, allreduce
            job.setMapperClass(edu.iu.mbkmeans.allreduce.MBKmeansMapper.class);
        }


        JobConf jobConf = (JobConf) job.getConfiguration();
        jobConf.set("mapreduce.framework.name", "map-collective");
        jobConf.setNumMapTasks(numMapTasks);
        jobConf.setInt("mapreduce.job.max.split.locations", 10000);
        job.setNumReduceTasks(0);
        jobConf.setInt(MBKmeansConstants.BATCH_SIZE,sizeBatch);//vaishnavi--batch_size
        jobConfig.setInt(MBKmeansConstants.VECTOR_SIZE,vectorSize);
        jobConfig.setInt(MBKmeansConstants.NUM_CENTROIDS, numCentroids);
        jobConfig.set(MBKmeansConstants.WORK_DIR,workDirPath.toString());
        jobConfig.setInt(MBKmeansConstants.NUM_MAPPERS, numMapTasks);
        return job;
    }

}