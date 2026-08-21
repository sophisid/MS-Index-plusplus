package io.github.algorithms.msindexpp;

import io.github.algorithms.Algorithm;
import io.github.algorithms.msindexpp.ml.PortableGBDTModel;
import io.github.algorithms.msindexpp.ml.SecondPassTraversal;
import io.github.algorithms.msindexpp.ml.TrainingDataWriter;
import io.github.algorithms.msindexshared.BoundedPriorityQueue;
import io.github.algorithms.msindexshared.segmentation.SegmentMethods;
import io.github.algorithms.msindexpp.segmentation.AdhocSegment;
import io.github.utils.CandidateMVSubsequence;
import io.github.utils.CandidateSegment;
import io.github.utils.DFTUtils;
import io.github.utils.KMeans;
import io.github.utils.LandmarkMBR;
import io.github.utils.LandmarkPortfolio;
import io.github.utils.RunningThreshold;
import io.github.utils.Tuple2;
import io.github.utils.rtreemulti.Entry;
import io.github.utils.rtreemulti.Node;
import io.github.utils.rtreemulti.RTree;
import io.github.utils.rtreemulti.geometry.Geometry;
import io.github.utils.rtreemulti.internal.LeafDefault;
import io.github.utils.rtreemulti.internal.NonLeafDefault;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

import static io.github.utils.Parameters.*;

public class MSIndexPP extends Algorithm {
    final KMeans[] kMeans = new KMeans[channels];
    RTree<CandidateSegment, Geometry> tree;
    private final int treeDimensions = fourierLength();

    private boolean buildTree = true;
    private PortableGBDTModel earlyTerminationModel;
    private boolean attemptedModelLoad = false;

    public MSIndexPP(boolean buildTree) {
        this.buildTree = buildTree;

        DFTUtils.computeKMeansClusters(kMeans);
    }

    @Override
    protected void prepareQuerying() {
        if (!msIndexPPTrainingEnabled && msIndexPPEarlyTerminationEnabled) {
            getEarlyTerminationModel();
        }
    }

    @Override
    public String getIndexPath() {
        File folder = new File("index_caches");
        if (!folder.exists()) {
            folder.mkdir();
        }

        return "index_caches/" + String.join("_",
                "MSINDEX",
                datasetName.toString(),
                String.valueOf(N),
                String.valueOf(maxM),
                String.valueOf(seed),
                String.valueOf(qLen),
                String.valueOf(channels),
                normalize ? "1" : "0",
                String.valueOf(kMeansClusters),
                String.valueOf(seed),
                String.valueOf(segmentMethod),
                String.valueOf(indexLeafSize)
        ) + ".ser";
    }

    private void createTree(List<Entry<CandidateSegment, Geometry>> items) {
        Logger.getGlobal().info("[MS-Index] Building the R-tree");
        tree = RTree.star().dimensions(treeDimensions).leafSize(buildTree ? indexLeafSize : Integer.MAX_VALUE).create(items);

        if (segmentMethod.equals(SegmentMethods.ADHOC)) {
            Logger.getGlobal().info("[MS-Index] Post-index segmentation");
            AdhocSegment.postIndexSegment(tree);
        }
    }

    public void buildIndex() {
        Logger.getGlobal().info("[MS-Index] Segmenting the fourier trails");
        final List<Entry<CandidateSegment, Geometry>> items = DFTUtils.getSegmentedFourierTrail(kMeans);

        createTree(items);
    }

    @Override
    public void saveIndex() {
        String filename = getIndexPath();

        try {
            Logger.getGlobal().info("[MS-Index] Saving the index to file");
            FileOutputStream fileOut = new FileOutputStream(filename);
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(tree);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public Algorithm loadIndex() {
        String filename = getIndexPath();

        try {
            Logger.getGlobal().info("[MS-Index] Loading the index from file");
            FileInputStream fileIn = new FileInputStream(filename);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            tree = (RTree<CandidateSegment, Geometry>) in.readObject();
            Logger.getGlobal().info("[MS-Index] Index loaded");
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return this;
    }

    public ArrayList<CandidateMVSubsequence> kNN(int k, double[][] query) {
        final Tuple2<double[], LandmarkPortfolio> queryDFTs = DFTUtils.getQueryDFTs(query, kMeans);
        final double[] fftQFlat = queryDFTs._1();
        final LandmarkPortfolio queryPortfolio = queryDFTs._2();
        final MinimumBoundingRectangle queryMBR = new MinimumBoundingRectangle(fftQFlat, fftQFlat, kMeansClusters == 0 ? null : new LandmarkMBR(queryPortfolio));

        final ArrayList<CandidateMVSubsequence> initialTopK = firstPass(k, query, queryMBR);
        final double kThDistance = Math.max(0d, initialTopK.get(k - 1).totalDistance);
        Logger.getGlobal().info("[MS-Index] k-th distance after initial topK: " + kThDistance);

        final ArrayList<CandidateMVSubsequence> finalTopK = secondPass(k, query, queryMBR, kThDistance);

        if (finalTopK.isEmpty()) {
            Logger.getGlobal().warning("No results found for query");
        }
        return finalTopK;
    }

    private ArrayList<CandidateMVSubsequence> firstPass(int k, double[][] query, MinimumBoundingRectangle queryMBR) {
        long start = System.currentTimeMillis();
        final List<CandidateSegment> candidateSegments = closest(queryMBR, k);
        indexSearchTime += System.currentTimeMillis() - start;

        start = System.currentTimeMillis();
        ArrayList<CandidateMVSubsequence> initialTopK = postFilter(query, candidateSegments, k, Double.POSITIVE_INFINITY);
        exhaustiveTime += System.currentTimeMillis() - start;

        return initialTopK;
    }

    private ArrayList<CandidateMVSubsequence> secondPass(int k, double[][] query, MinimumBoundingRectangle queryMBR, double threshold) {
        if (msIndexPPEvaluationEnabled && msIndexPPEarlyTerminationEnabled && !msIndexPPTrainingEnabled) {
            return secondPassWithEvaluation(k, query, queryMBR, threshold);
        }

        long start = System.currentTimeMillis();
        final SecondPassTraversal.TraversalResult traversalResult = SecondPassTraversal.traverse(
                tree,
                queryMBR,
                threshold,
                k,
                msIndexPPCheckpointInterval,
                getCheckpointDecider(),
                msIndexPPTrainingEnabled
        );

        final List<CandidateSegment> finalCandidatesList = traversalResult.acceptedCandidates;

        segmentsUnderThreshold.getAndAdd(finalCandidatesList.size());

        Logger.getGlobal().info("[MS-Index] Under threshold candidates: " + finalCandidatesList.size());
        if (traversalResult.stoppedEarly) {
            msIndexPPEarlyStops.incrementAndGet();
            msIndexPPEarlyTerminationSteps.addAndGet(traversalResult.stepsExecuted);
            Logger.getGlobal().info(String.format(
                    "[MS-IndexPP] Early termination triggered at checkpoint %d after %d steps with score %.4f",
                    traversalResult.lastCheckpointId,
                    traversalResult.stepsExecuted,
                    traversalResult.stopScore
            ));
        }
        indexSearchTime += System.currentTimeMillis() - start;

        start = System.currentTimeMillis();
        final ArrayList<CandidateMVSubsequence> finalTopK = postFilter(query, finalCandidatesList, k, threshold);
        exhaustiveTime += System.currentTimeMillis() - start;

        if (msIndexPPTrainingEnabled && !traversalResult.checkpoints.isEmpty()) {
            persistTrainingRows(query, k, threshold, finalCandidatesList, finalTopK, traversalResult);
        }

        return finalTopK;
    }

    private ArrayList<CandidateMVSubsequence> secondPassWithEvaluation(int k, double[][] query, MinimumBoundingRectangle queryMBR, double threshold) {
        long start = System.currentTimeMillis();
        final SecondPassTraversal.TraversalResult exactTraversal = SecondPassTraversal.traverse(
                tree,
                queryMBR,
                threshold,
                k,
                msIndexPPCheckpointInterval,
                null,
                false
        );
        final SecondPassTraversal.TraversalResult earlyTraversal = SecondPassTraversal.traverse(
                tree,
                queryMBR,
                threshold,
                k,
                msIndexPPCheckpointInterval,
                getCheckpointDecider(),
                false
        );
        indexSearchTime += System.currentTimeMillis() - start;

        segmentsUnderThreshold.getAndAdd(earlyTraversal.acceptedCandidates.size());

        start = System.currentTimeMillis();
        final ArrayList<CandidateMVSubsequence> exactTopK = postFilter(query, exactTraversal.acceptedCandidates, k, threshold);
        final ArrayList<CandidateMVSubsequence> earlyTopK = postFilter(query, earlyTraversal.acceptedCandidates, k, threshold);
        exhaustiveTime += System.currentTimeMillis() - start;

        final double exactTopKDist = topKDistance(exactTopK);
        final double earlyTopKDist = topKDistance(earlyTopK);
        double achievedAccuracy = overlapAccuracy(earlyTopK, exactTopK, k);
        msIndexPPEarlyTerminationAccuracySum.add(achievedAccuracy);
        if (earlyTraversal.stoppedEarly) {
            msIndexPPEarlyStops.incrementAndGet();
            msIndexPPEarlyTerminationSteps.addAndGet(earlyTraversal.stepsExecuted);
        }

        Logger.getGlobal().info(String.format(
                "[MS-IndexPP] Eval query=%d earlyStop=%s accuracy=%.4f earlyCandidates=%d exactCandidates=%d earlySteps=%d exactSteps=%d earlyTopKDist=%.12f exactTopKDist=%.12f",
                currQueryId,
                earlyTraversal.stoppedEarly,
                achievedAccuracy,
                earlyTraversal.acceptedCandidates.size(),
                exactTraversal.acceptedCandidates.size(),
                earlyTraversal.stepsExecuted,
                exactTraversal.stepsExecuted,
                earlyTopKDist,
                exactTopKDist
        ));

        return earlyTopK;
    }

    private double topKDistance(ArrayList<CandidateMVSubsequence> topK) {
        if (topK == null || topK.isEmpty()) {
            return Double.NaN;
        }
        return topK.get(topK.size() - 1).totalDistance;
    }

    private List<CandidateSegment> closest(MinimumBoundingRectangle queryMBR, int k) {
        final RunningThreshold runningThreshold = new RunningThreshold(Double.POSITIVE_INFINITY);
        final BoundedPriorityQueue<CandidateSegment> q = new BoundedPriorityQueue<>(k, Comparator.comparingDouble(CandidateSegment::getCurrLB));

        for (Entry<CandidateSegment, Geometry> entry : tree.search(queryMBR, runningThreshold)) {
            final CandidateSegment candidateSegment = entry.value();
            q.add(candidateSegment);

            if (q.size() == k) {
                runningThreshold.set(q.peek().getCurrLB());
            }
        }
        return q.asOrderedList();
    }

    private SecondPassTraversal.CheckpointDecider getCheckpointDecider() {
        if (msIndexPPTrainingEnabled || !msIndexPPEarlyTerminationEnabled) {
            return null;
        }

        PortableGBDTModel model = getEarlyTerminationModel();
        if (model == null) {
            return null;
        }

        final double scoreThreshold = model.getConformalThreshold() != null
                ? model.getConformalThreshold()
                : msIndexPPModelScoreThreshold;

        return new SecondPassTraversal.CheckpointDecider() {
            @Override
            public double score(io.github.algorithms.msindexpp.ml.TraversalFeatureSnapshot snapshot) {
                return model.predictProbability(snapshot);
            }

            @Override
            public boolean shouldStop(io.github.algorithms.msindexpp.ml.TraversalFeatureSnapshot snapshot, double score) {
                return score >= scoreThreshold;
            }
        };
    }

    private PortableGBDTModel getEarlyTerminationModel() {
        if (attemptedModelLoad) {
            return earlyTerminationModel;
        }
        attemptedModelLoad = true;

        File modelFile = new File(msIndexPPModelPath);
        if (!modelFile.exists()) {
            Logger.getGlobal().warning("[MS-IndexPP] No early termination model found at " + modelFile.getPath());
            return null;
        }

        try {
            earlyTerminationModel = PortableGBDTModel.load(modelFile);
            Logger.getGlobal().info("[MS-IndexPP] Loaded early termination model from " + modelFile.getPath());
        } catch (IOException e) {
            Logger.getGlobal().warning("[MS-IndexPP] Could not load early termination model: " + e.getMessage());
            earlyTerminationModel = null;
        }
        return earlyTerminationModel;
    }

    private void persistTrainingRows(double[][] query, int k, double threshold,
                                     List<CandidateSegment> finalCandidatesList,
                                     ArrayList<CandidateMVSubsequence> finalTopK,
                                     SecondPassTraversal.TraversalResult traversalResult) {
        List<TrainingDataWriter.TrainingRow> rows = new ArrayList<>();
        for (SecondPassTraversal.CheckpointRecord checkpoint : traversalResult.checkpoints) {
            if (checkpoint.candidateCountAtCheckpoint < k) {
                continue;
            }

            List<CandidateSegment> partialCandidates = finalCandidatesList.subList(0, checkpoint.candidateCountAtCheckpoint);
            ArrayList<CandidateMVSubsequence> partialTopK = postFilter(query, partialCandidates, k, threshold);
            double achievedAccuracy = overlapAccuracy(partialTopK, finalTopK, k);
            boolean label = achievedAccuracy >= msIndexPPAccuracyTarget;

            rows.add(new TrainingDataWriter.TrainingRow(
                    checkpoint.snapshot,
                    label,
                    achievedAccuracy,
                    checkpoint.candidateCountAtCheckpoint,
                    finalCandidatesList.size()
            ));
        }

        try {
            TrainingDataWriter.appendRows(msIndexPPTrainingDataPath, currQueryId, rows);
        } catch (IOException e) {
            Logger.getGlobal().warning("[MS-IndexPP] Could not persist training rows: " + e.getMessage());
        }
    }

    private double overlapAccuracy(List<CandidateMVSubsequence> partialTopK, List<CandidateMVSubsequence> finalTopK, int k) {
        Set<CandidateMVSubsequence> partialSet = new HashSet<>(partialTopK);
        int overlap = 0;
        for (CandidateMVSubsequence candidate : finalTopK) {
            if (partialSet.contains(candidate)) {
                overlap++;
            }
        }
        return overlap / (double) k;
    }

    private long recursiveTreeMemoryUsage(Node<CandidateSegment, Geometry> node) {
        if (node instanceof NonLeafDefault) {
            NonLeafDefault<CandidateSegment, Geometry> nonLeaf = (NonLeafDefault<CandidateSegment, Geometry>) node;
            long memoryUsage = tree.dimensions() * 8L * 2L + 4L * 2L;
            for (Node<CandidateSegment, Geometry> child : nonLeaf.children()) {
                memoryUsage += recursiveTreeMemoryUsage(child);
            }
            return memoryUsage;
        } else if (node instanceof LeafDefault) {
            LeafDefault<CandidateSegment, Geometry> leaf = (LeafDefault<CandidateSegment, Geometry>) node;
            long memoryUsage = tree.dimensions() * 8L * 2L;
            memoryUsage += leaf.count() * tree.dimensions() * 8L * 2L + leaf.count() * (4L * 3L + (kMeansClusters == 0 ? 0 : channels * 2L * 8L + channels * 4L));
            return memoryUsage;
        }
        throw new IllegalStateException("Unknown node type " + node.getClass());
    }

    public double memoryUsage() {
        return recursiveTreeMemoryUsage(tree.root().get()) / 1000000d;
    }
}
