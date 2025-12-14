package io.github.algorithms.msindexpp;

import io.github.algorithms.Algorithm;
import io.github.algorithms.mass.MVMASS;
import io.github.algorithms.msindexpp.segmentation.AdhocSegment;
import io.github.algorithms.msindexpp.segmentation.SegmentMethods;
import io.github.io.DataManager;
import io.github.utils.*;
import io.github.utils.rtreemulti.Entry;
import io.github.utils.rtreemulti.Node;
import io.github.utils.rtreemulti.RTree;
import io.github.utils.rtreemulti.geometry.Geometry;
import io.github.utils.rtreemulti.internal.LeafDefault;
import io.github.utils.rtreemulti.internal.NonLeafDefault;

import java.io.*;
import java.util.*;
import java.util.logging.Logger;

import static io.github.utils.Parameters.*;
import static java.util.stream.Collectors.toList;

public class MSIndexPP extends Algorithm {
    private static final int RC_SERIES_SAMPLES = 8;
    private static final int RC_WINDOW_SAMPLES = 16;
    private static final double RC_THRESHOLD = 1.06;
    private static final double EPS = 1e-12;

    private MVMASS massEngine = null;

    final KMeans[] kMeans = new KMeans[channels];
    RTree<CandidateSegment, Geometry> tree;
    private final int treeDimensions = fourierLength();

    private boolean buildTree = true;

    // ================== DIAGNOSTICS ==================
    public static class QueryDiagnostics {
        public double rcEstimate;
        public long massTimeNanos;
        public long msIndexTimeNanos;
        public long msIndexFirstPassNanos;
        public long msIndexSecondPassNanos;
        public boolean decisionUsedMass;
        public boolean massActuallyFaster;
        public double leafRC;
        public int candidatesFromIndex;

        @Override
        public String toString() {
            return String.format(Locale.US,
                    "RC=%.4f, MASS=%.3f ms, MSIndex=%.3f ms (first=%.3f, second=%.3f), " +
                            "decisionUsedMass=%s, massActuallyFaster=%s, leafRC=%.4f, idxCandidates=%d",
                    rcEstimate,
                    massTimeNanos / 1e6,
                    msIndexTimeNanos / 1e6,
                    msIndexFirstPassNanos / 1e6,
                    msIndexSecondPassNanos / 1e6,
                    decisionUsedMass,
                    massActuallyFaster,
                    leafRC,
                    candidatesFromIndex
            );
        }
    }
    // ================== END DIAGNOSTICS ==================

    public MSIndexPP(boolean buildTree) {
        this.buildTree = buildTree;
        DFTUtils.computeKMeansClusters(kMeans);
    }

    @Override
    public String getIndexPath() {
        File folder = new File("index_caches");
        if (!folder.exists()) {
            folder.mkdir();
        }

        return "index_caches/" + String.join("_",
                algorithmType.toString(),
                datasetName,
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
        Logger.getGlobal().info("[MS-Index++] Building the R-tree " + (buildTree ? "[MAIN]" : "[AUX]"));
        tree = RTree.star().dimensions(treeDimensions).leafSize(buildTree ? indexLeafSize : Integer.MAX_VALUE).create(items);

        if (segmentMethod.equals(SegmentMethods.ADHOC)) {
            Logger.getGlobal().info("[MS-Index++] Post-index segmentation");
            AdhocSegment.postIndexSegment(tree);
        }
    }

    public void buildIndex() {
        Logger.getGlobal().info("[MS-Index++] Segmenting the fourier trails");
        final List<Entry<CandidateSegment, Geometry>> items = DFTUtils.getSegmentedFourierTrail(kMeans);
        createTree(items);
    }

    @Override
    public void saveIndex() {
        String filename = getIndexPath();
        try {
            Logger.getGlobal().info("[MS-Index++] Saving the index to file");
            FileOutputStream fileOut = new FileOutputStream(filename);
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(tree);
            out.close();
            fileOut.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public Algorithm loadIndex() {
        String filename = getIndexPath();
        try {
            Logger.getGlobal().info("[MS-Index++] Loading the index from file");
            FileInputStream fileIn = new FileInputStream(filename);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            tree = (RTree<CandidateSegment, Geometry>) in.readObject();
            in.close();
            fileIn.close();
            Logger.getGlobal().info("[MS-Index++] Index loaded");
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return this;
    }

    private static double edWindow(double[] series, int start, double[] q, boolean zNorm) {
        double sum = 0.0;
        if (!zNorm) {
            for (int i = 0; i < qLen; i++) {
                double diff = series[start + i] - q[i];
                sum += diff * diff;
            }
            return Math.sqrt(sum);
        } else {
            double muS = 0.0, muQ = 0.0;
            for (int i = 0; i < qLen; i++) {
                muS += series[start + i];
                muQ += q[i];
            }
            muS /= qLen;
            muQ /= qLen;

            double varS = 0.0, varQ = 0.0;
            for (int i = 0; i < qLen; i++) {
                double ds = series[start + i] - muS;
                double dq = q[i] - muQ;
                varS += ds * ds;
                varQ += dq * dq;
            }
            double stdS = Math.sqrt(Math.max(varS, 0.0) + EPS);
            double stdQ = Math.sqrt(Math.max(varQ, 0.0) + EPS);

            for (int i = 0; i < qLen; i++) {
                double zs = (series[start + i] - muS) / stdS;
                double zq = (q[i] - muQ) / stdQ;
                double diff = zs - zq;
                sum += diff * diff;
            }
            return Math.sqrt(sum);
        }
    }

    private double estimateRC_viaSamples(double[][] query) {
        final double[][][] data = DataManager.data;
        if (data.length == 0) return 1.0;

        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;

        for (int s = 0; s < RC_SERIES_SAMPLES; s++) {
            int tsId = random.nextInt(data.length);
            final int M = DataManager.getM(tsId);
            final int lim = Math.max(1, M - qLen + 1);

            for (int w = 0; w < RC_WINDOW_SAMPLES; w++) {
                int start = random.nextInt(lim);

                double dist2 = 0.0;
                for (int ch : selectedVariatesIdx) {
                    dist2 += Math.pow(edWindow(data[tsId][ch], start, query[ch], normalize), 2);
                }
                double d = Math.sqrt(dist2);

                if (d < min) min = d;
                if (d > max) max = d;
            }
        }

        if (!Double.isFinite(min) || !Double.isFinite(max)) return 1.0;
        return max / Math.max(min, EPS);
    }

    private MVMASS getMassEngine() {
        if (massEngine == null) {
            Logger.getGlobal().info("[MS-Index++] RC low ⇒ building MASS engine once");
            massEngine = new MVMASS();
            long t0 = System.currentTimeMillis();
            massEngine.buildIndex();
            long t1 = System.currentTimeMillis();
            Logger.getGlobal().info("[MS-Index++] MASS FFTs built in " + (t1 - t0) + "ms");
        }
        return massEngine;
    }

    public ArrayList<CandidateMVSubsequence> kNN(int k, double[][] query) {
        double rc = estimateRC_viaSamples(query);
        Logger.getGlobal().info(String.format("[MS-Index++] RC estimate: %.5f (thr=%.3f)", rc, RC_THRESHOLD));

        if (rc < RC_THRESHOLD) {
            Logger.getGlobal().info("[MS-Index++] Skipping index; falling back to MASS");
            ArrayList<CandidateMVSubsequence> res = new ArrayList<>(getMassEngine().kNN(k, query));
            subsequencesExhChecked.getAndAdd(nSubsequences);
            segmentsUnderThreshold.getAndAdd(N);
            return res;
        }

        final Tuple2<double[], LandmarkPortfolio> queryDFTs = DFTUtils.getQueryDFTs(query, kMeans);
        final double[] fftQFlat = queryDFTs._1();
        final LandmarkPortfolio queryPortfolio = queryDFTs._2();
        final MinimumBoundingRectangle queryMBR =
                new MinimumBoundingRectangle(fftQFlat, fftQFlat, kMeansClusters == 0 ? null : new LandmarkMBR(queryPortfolio));

        final ArrayList<CandidateMVSubsequence> initialTopK = firstPass(k, query, queryMBR);
        final double kThDistance = initialTopK.get(k - 1).totalDistance;
        Logger.getGlobal().info("[MS-Index++] k-th distance after initial topK: " + kThDistance);

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
        long start = System.currentTimeMillis();
        final Iterable<Entry<CandidateSegment, Geometry>> finalCandidates = tree.search(queryMBR, new RunningThreshold(threshold));

        final List<CandidateSegment> finalCandidatesList = lib.getStream(finalCandidates).map(Entry::value).collect(toList());

        segmentsUnderThreshold.getAndAdd(finalCandidatesList.size());

        Logger.getGlobal().info("[MS-Index++] Under threshold candidates: " + finalCandidatesList.size());
        indexSearchTime += System.currentTimeMillis() - start;

        start = System.currentTimeMillis();
        final ArrayList<CandidateMVSubsequence> finalTopK = postFilter(query, finalCandidatesList, k, threshold);
        exhaustiveTime += System.currentTimeMillis() - start;

        return finalTopK;
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

    private long recursiveTreeMemoryUsage(Node<CandidateSegment, Geometry> node) {
        if (node instanceof NonLeafDefault) {
            NonLeafDefault<CandidateSegment, Geometry> nonLeaf = (NonLeafDefault<CandidateSegment, Geometry>) node;
            long memoryUsage = tree.dimensions() * 8L * 2L + 4L * 2L; // mins, maxes, 2 pointers
            for (Node<CandidateSegment, Geometry> child : nonLeaf.children()) {
                memoryUsage += recursiveTreeMemoryUsage(child);
            }
            return memoryUsage;
        } else if (node instanceof LeafDefault) {
            LeafDefault<CandidateSegment, Geometry> leaf = (LeafDefault<CandidateSegment, Geometry>) node;
            long memoryUsage = tree.dimensions() * 8L * 2L; // mins, maxes
            memoryUsage += leaf.count() * tree.dimensions() * 8L * 2L + leaf.count() * (4L * 3L + (kMeansClusters == 0 ? 0 : channels * 2L * 8L + channels * 4L));
            return memoryUsage;
        }
        throw new IllegalStateException("Unknown node type " + node.getClass());
    }

    public double memoryUsage() {
        return recursiveTreeMemoryUsage(tree.root().get()) / 1_000_000d;
    }

    private double computeLeafRelativeContrastFromCandidates(List<CandidateSegment> candidates) {
        if (candidates == null || candidates.isEmpty()) return 1.0;

        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;

        for (CandidateSegment seg : candidates) {
            double d = seg.getCurrLB();
            if (d < min) min = d;
            if (d > max) max = d;
        }

        if (!Double.isFinite(min) || !Double.isFinite(max)) return 1.0;
        return max / Math.max(min, EPS);
    }

    public QueryDiagnostics debugCompareMassVsIndex(int k, double[][] query) {
        QueryDiagnostics diag = new QueryDiagnostics();

        double rc = estimateRC_viaSamples(query);
        diag.rcEstimate = rc;
        diag.decisionUsedMass = (rc < RC_THRESHOLD);

        long t0 = System.nanoTime();
        ArrayList<CandidateMVSubsequence> massRes =
                new ArrayList<>(getMassEngine().kNN(k, query));
        long t1 = System.nanoTime();
        diag.massTimeNanos = t1 - t0;

        final Tuple2<double[], LandmarkPortfolio> queryDFTs =
                DFTUtils.getQueryDFTs(query, kMeans);
        final double[] fftQFlat = queryDFTs._1();
        final LandmarkPortfolio queryPortfolio = queryDFTs._2();
        final MinimumBoundingRectangle queryMBR =
                new MinimumBoundingRectangle(
                        fftQFlat,
                        fftQFlat,
                        kMeansClusters == 0 ? null : new LandmarkMBR(queryPortfolio)
                );

        long tFirst0 = System.nanoTime();
        final List<CandidateSegment> candidateSegments = closest(queryMBR, k);
        long tFirst1 = System.nanoTime();

        long tFirst2 = System.nanoTime();
        ArrayList<CandidateMVSubsequence> initialTopK =
                postFilter(query, candidateSegments, k, Double.POSITIVE_INFINITY);
        long tFirst3 = System.nanoTime();

        double kThDistance = initialTopK.get(k - 1).totalDistance;
        long firstPassTime = (tFirst1 - tFirst0) + (tFirst3 - tFirst2);
        diag.msIndexFirstPassNanos = firstPassTime;

        long tSecond0 = System.nanoTime();
        final Iterable<Entry<CandidateSegment, Geometry>> finalCandidatesIter =
                tree.search(queryMBR, new RunningThreshold(kThDistance));
        final List<CandidateSegment> finalCandidatesList =
                lib.getStream(finalCandidatesIter).map(Entry::value).collect(toList());
        long tSecond1 = System.nanoTime();

        long tSecond2 = System.nanoTime();
        ArrayList<CandidateMVSubsequence> finalTopK =
                postFilter(query, finalCandidatesList, k, kThDistance);
        long tSecond3 = System.nanoTime();

        long secondPassTime = (tSecond1 - tSecond0) + (tSecond3 - tSecond2);
        diag.msIndexSecondPassNanos = secondPassTime;
        diag.msIndexTimeNanos = firstPassTime + secondPassTime;
        diag.candidatesFromIndex = finalCandidatesList.size();

        diag.massActuallyFaster = diag.massTimeNanos < diag.msIndexTimeNanos;

        diag.leafRC = computeLeafRelativeContrastFromCandidates(finalCandidatesList);

        Logger.getGlobal().info("[MS-Index++] Diagnostics: " + diag);
        return diag;
    }

}
