package io.github.algorithms.msindexpp;

import io.github.algorithms.Algorithm;
import io.github.algorithms.mass.MVMASS;
import io.github.algorithms.msindexpp.segmentation.AdhocSegment;
import io.github.algorithms.msindexpp.segmentation.SegmentMethods;
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

import static java.util.stream.Collectors.toList;
import static io.github.utils.Parameters.*;

public class MSIndexPP extends Algorithm {
    final KMeans[] kMeans = new KMeans[channels];
    RTree<CandidateSegment, Geometry> tree;
    RTree<CandidateSegment, Geometry> sampleTree;
    private final int treeDimensions = fourierLength();
    private final MVMASS mass = new MVMASS();

    private boolean buildTree = true;
    private int sampleSize = 0;

    private int calibSeenQueries = 0;
    private double calibSumPruneMsBetter = 0.0;
    private double calibSumPruneMassBetter = 0.0;
    private int calibCountMsBetter = 0;
    private int calibCountMassBetter = 0;
    private boolean calibFinished = false;

    public MSIndexPP(boolean buildTree) {
        this.buildTree = buildTree;
        DFTUtils.computeKMeansClusters(kMeans);
    }

    // ----------------------------------------------------------------------
    // Index path
    // ----------------------------------------------------------------------
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

    // ----------------------------------------------------------------------
    // Tree creation
    // ----------------------------------------------------------------------
    private void createTree(List<Entry<CandidateSegment, Geometry>> items) {
        Logger.getGlobal().info("[MS-Index] Building the R-tree");
        tree = RTree.star()
                .dimensions(treeDimensions)
                .leafSize(buildTree ? indexLeafSize : Integer.MAX_VALUE)
                .create(items);

        if (segmentMethod.equals(SegmentMethods.ADHOC)) {
            Logger.getGlobal().info("[MS-Index] Post-index segmentation");
            AdhocSegment.postIndexSegment(tree);
        }
    }

    private List<Entry<CandidateSegment, Geometry>> buildSampleItems(List<Entry<CandidateSegment, Geometry>> items) {
        if (hybridSampleFraction <= 0 || hybridSampleFraction >= 1) {
            sampleSize = items.size();
            return items; // trivial: όλο το index σαν sample
        }

        int target = (int) Math.max(1, Math.round(items.size() * hybridSampleFraction));
        ArrayList<Entry<CandidateSegment, Geometry>> sample = new ArrayList<>(target);

        for (Entry<CandidateSegment, Geometry> e : items) {
            if (random.nextDouble() < hybridSampleFraction) {
                sample.add(e);
            }
        }

        // Αν πήραμε πολύ λίγα, συμπληρώνουμε sequentially
        while (sample.size() < target && sample.size() < items.size()) {
            sample.add(items.get(sample.size()));
        }

        sampleSize = sample.size();
        return sample;
    }

    public void buildIndex() {
        Logger.getGlobal().info("[MS-Index] Segmenting the fourier trails");
        final List<Entry<CandidateSegment, Geometry>> items = DFTUtils.getSegmentedFourierTrail(kMeans);

        createTree(items);

        if (hybridMode) {
            Logger.getGlobal().info("[HYBRID] Building sample tree and MASS index");

            List<Entry<CandidateSegment, Geometry>> sampleItems = buildSampleItems(items);

            Logger.getGlobal().info("[HYBRID] Sample size: " + sampleSize + " / " + items.size());

            sampleTree = RTree
                    .star()
                    .dimensions(treeDimensions)
                    .leafSize(indexLeafSize * 2)
                    .create(sampleItems);

            long start = System.currentTimeMillis();
            mass.buildIndex();
            long t = System.currentTimeMillis() - start;
            Logger.getGlobal().info("[HYBRID] MASS buildIndex time: " + t + " ms");
        }
    }

    // ----------------------------------------------------------------------
    // Save / load index
    // ----------------------------------------------------------------------
    @Override
    public void saveIndex() {
        String filename = getIndexPath();

        try {
            Logger.getGlobal().info("[MS-Index] Saving the index to file");
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
            Logger.getGlobal().info("[MS-Index] Loading the index from file");
            FileInputStream fileIn = new FileInputStream(filename);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            tree = (RTree<CandidateSegment, Geometry>) in.readObject();
            in.close();
            fileIn.close();
            Logger.getGlobal().info("[MS-Index] Index loaded");
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return this;
    }

    // ----------------------------------------------------------------------
    // Hybrid expected prune estimation (sample-based)
    // ----------------------------------------------------------------------
    private double estimateExpectedPruneFromSample(MinimumBoundingRectangle queryMBR) {
        Logger logger = Logger.getGlobal();

        if (sampleTree == null || sampleSize == 0) {
            logger.info("[HYBRID][SAMPLE] No sample tree → expectedPrune=1.0");
            return 1.0;
        }

        final int maxSegmentsToCheck = 1000;
        final RunningThreshold rt = new RunningThreshold(Double.POSITIVE_INFINITY);

        int hits = 0;
        for (Entry<CandidateSegment, Geometry> entry : sampleTree.search(queryMBR, rt)) {
            hits++;
            if (hits >= maxSegmentsToCheck) break;
        }

        double sampleFraction = nSubsequences > 0
                ? sampleSize / (double) nSubsequences
                : 0.0;

        logger.info(String.format(
                "[HYBRID][SAMPLE] hits=%d (cap=%d), sampleSize=%d, nSubsequences=%d, sampleFraction=%.6f",
                hits, maxSegmentsToCheck, sampleSize, nSubsequences, sampleFraction
        ));

        if (hits == 0) {
            logger.info("[HYBRID][SAMPLE] hits=0 ⇒ expectedPrune≈1.0");
            return 1.0;
        }

        double probHitInProbedSample = hits / (double) maxSegmentsToCheck;
        double probSampleVsGlobal = sampleFraction;

        double expectedCoverageGlobal = probHitInProbedSample * probSampleVsGlobal;
        double expectedPrune = 1.0 - expectedCoverageGlobal;

        if (expectedPrune < 0.0) expectedPrune = 0.0;
        if (expectedPrune > 1.0) expectedPrune = 1.0;

        logger.info(String.format(
                "[HYBRID][SAMPLE] probHit=%.6f, probSampleVsGlobal=%.6f, expectedCoverage=%.6f, expectedPrune=%.6f",
                probHitInProbedSample, probSampleVsGlobal, expectedCoverageGlobal, expectedPrune
        ));

        return expectedPrune;
    }

    // ----------------------------------------------------------------------
    // kNN – με hybrid απόφαση + online calibration στα πρώτα queries
    // ----------------------------------------------------------------------
    @Override
    public ArrayList<CandidateMVSubsequence> kNN(int k, double[][] query) {
        Logger logger = Logger.getGlobal();

        if (!hybridMode) {
            return kNNMsIndexOnly(k, query);
        }

        final Tuple2<double[], LandmarkPortfolio> queryDFTs = DFTUtils.getQueryDFTs(query, kMeans);
        final double[] fftQFlat = queryDFTs._1();
        final LandmarkPortfolio queryPortfolio = queryDFTs._2();
        final MinimumBoundingRectangle queryMBR =
                new MinimumBoundingRectangle(fftQFlat, fftQFlat,
                        kMeansClusters == 0 ? null : new LandmarkMBR(queryPortfolio));

        double expectedPrune = estimateExpectedPruneFromSample(queryMBR);

        if (!calibFinished && hybridCalibrationQueries > 0 && calibSeenQueries < hybridCalibrationQueries) {
            calibSeenQueries++;

            long startMs = System.nanoTime();
            ArrayList<CandidateMVSubsequence> msTopK = kNNMsIndexOnly(k, query);
            long tMsNs = System.nanoTime() - startMs;
            double tMsMs = tMsNs / 1_000_000.0;

            long startMass = System.nanoTime();
            ArrayList<CandidateMVSubsequence> massTopK = new ArrayList<>(mass.kNN(k, query));
            long tMassNs = System.nanoTime() - startMass;
            double tMassMs = tMassNs / 1_000_000.0;

            boolean msBetter = tMsMs <= tMassMs;
            if (msBetter) {
                calibSumPruneMsBetter += expectedPrune;
                calibCountMsBetter++;
            } else {
                calibSumPruneMassBetter += expectedPrune;
                calibCountMassBetter++;
            }

            logger.info(String.format(
                    "[HYBRID][CALIB] q=%d/%d, expectedPrune=%.6f, tMs=%.3f ms, tMass=%.3f ms, winner=%s",
                    calibSeenQueries, hybridCalibrationQueries,
                    expectedPrune, tMsMs, tMassMs,
                    msBetter ? "MS-Index" : "MASS"
            ));

            // Όταν τελειώσει η calibration φάση, υπολογίζουμε adaptive cutoff
            if (calibSeenQueries == hybridCalibrationQueries) {
                calibFinished = true;

                double avgMs = calibCountMsBetter > 0
                        ? (calibSumPruneMsBetter / calibCountMsBetter)
                        : 1.0;
                double avgMass = calibCountMassBetter > 0
                        ? (calibSumPruneMassBetter / calibCountMassBetter)
                        : 0.0;

                double newCutoff;
                if (calibCountMsBetter == 0 && calibCountMassBetter == 0) {
                    newCutoff = hybridMinPruneSample; // δεν μάθαμε τίποτα, κράτα το παλιό
                } else if (calibCountMsBetter == 0) {
                    // μόνο MASS ήταν καλύτερος
                    newCutoff = 1.0;
                } else if (calibCountMassBetter == 0) {
                    // μόνο MS-Index ήταν καλύτερος
                    newCutoff = 0.0;
                } else {
                    // cutoff στο μέσο των δύο μέσων
                    newCutoff = 0.5 * (avgMs + avgMass);
                }

                hybridMinPruneSample = newCutoff;

                logger.info(String.format(
                        "[HYBRID][CALIB] avgPrune(MS-Index)=%.6f (n=%d), avgPrune(MASS)=%.6f (n=%d) → hybridMinPruneSample=%.6f",
                        avgMs, calibCountMsBetter, avgMass, calibCountMassBetter, hybridMinPruneSample
                ));
            }

            // Στην calibration φάση επιστρέφουμε τα αποτελέσματα του νικητή για το συγκεκριμένο query
            return (tMsMs <= tMassMs) ? msTopK : massTopK;
        }

        // ------------------------------------------------------------------
        // 2) Κανονική φάση: έχουμε cutoff, αποφασίζουμε φτηνά από expectedPrune
        // ------------------------------------------------------------------
        boolean chooseMs = expectedPrune >= hybridMinPruneSample;

        logger.info(String.format(
                "[HYBRID][DECIDE] expectedPrune=%.6f, cutoff=%.6f → choose=%s",
                expectedPrune, hybridMinPruneSample,
                chooseMs ? "MS-Index" : "MASS"
        ));

        if (chooseMs) {
            long startMs = System.nanoTime();
            ArrayList<CandidateMVSubsequence> msTopK = kNNMsIndexOnly(k, query);
            long tMsNs = System.nanoTime() - startMs;
            double tMsMs = tMsNs / 1_000_000.0;
            logger.info(String.format("[HYBRID] Chosen MS-Index, time=%.3f ms", tMsMs));
            return msTopK;
        } else {
            long startMass = System.nanoTime();
            ArrayList<CandidateMVSubsequence> massTopK = new ArrayList<>(mass.kNN(k, query));
            long tMassNs = System.nanoTime() - startMass;
            double tMassMs = tMassNs / 1_000_000.0;
            logger.info(String.format("[HYBRID] Chosen MASS, time=%.3f ms", tMassMs));
            return massTopK;
        }
    }

    // ----------------------------------------------------------------------
    // MS-Index-only kNN (όπως πριν)
    // ----------------------------------------------------------------------
    private ArrayList<CandidateMVSubsequence> kNNMsIndexOnly(int k, double[][] query) {
        // Compute query DFTs & MBR
        final Tuple2<double[], LandmarkPortfolio> queryDFTs = DFTUtils.getQueryDFTs(query, kMeans);
        final double[] fftQFlat = queryDFTs._1();
        final LandmarkPortfolio queryPortfolio = queryDFTs._2();
        final MinimumBoundingRectangle queryMBR =
                new MinimumBoundingRectangle(fftQFlat, fftQFlat,
                        kMeansClusters == 0 ? null : new LandmarkMBR(queryPortfolio));

        // 1st pass
        final ArrayList<CandidateMVSubsequence> initialTopK = firstPass(k, query, queryMBR);
        final double kThDistance = initialTopK.get(k - 1).totalDistance;
        Logger.getGlobal().info("[MS-IndexPP] k-th distance after initial topK: " + kThDistance);

        // 2nd pass
        final ArrayList<CandidateMVSubsequence> finalTopK = secondPass(k, query, queryMBR, kThDistance);

        if (finalTopK.isEmpty()) {
            Logger.getGlobal().warning("No results found for query");
        }
        return finalTopK;
    }

    private List<CandidateSegment> closest(MinimumBoundingRectangle queryMBR, int k) {
        final RunningThreshold runningThreshold = new RunningThreshold(Double.POSITIVE_INFINITY);
        final BoundedPriorityQueue<CandidateSegment> q =
                new BoundedPriorityQueue<>(k, Comparator.comparingDouble(CandidateSegment::getCurrLB));

        for (Entry<CandidateSegment, Geometry> entry : tree.search(queryMBR, runningThreshold)) {
            final CandidateSegment candidateSegment = entry.value();
            q.add(candidateSegment);

            if (q.size() == k) {
                runningThreshold.set(q.peek().getCurrLB());
            }
        }
        return q.asOrderedList();
    }

    private ArrayList<CandidateMVSubsequence> firstPass(int k,
                                                        double[][] query,
                                                        MinimumBoundingRectangle queryMBR) {
        long start = System.currentTimeMillis();
        final List<CandidateSegment> candidateSegments = closest(queryMBR, k);
        indexSearchTime += System.currentTimeMillis() - start;

        start = System.currentTimeMillis();
        ArrayList<CandidateMVSubsequence> initialTopK =
                postFilter(query, candidateSegments, k, Double.POSITIVE_INFINITY);
        exhaustiveTime += System.currentTimeMillis() - start;

        return initialTopK;
    }

    private ArrayList<CandidateMVSubsequence> secondPass(int k,
                                                         double[][] query,
                                                         MinimumBoundingRectangle queryMBR,
                                                         double threshold) {
        long start = System.currentTimeMillis();
        final Iterable<Entry<CandidateSegment, Geometry>> finalCandidates =
                tree.search(queryMBR, new RunningThreshold(threshold));

        final List<CandidateSegment> finalCandidatesList =
                lib.getStream(finalCandidates).map(Entry::value).collect(toList());

        segmentsUnderThreshold.getAndAdd(finalCandidatesList.size());

        Logger.getGlobal().info("[MS-Index] Under threshold candidates: " + finalCandidatesList.size());
        indexSearchTime += System.currentTimeMillis() - start;

        start = System.currentTimeMillis();
        final ArrayList<CandidateMVSubsequence> finalTopK =
                postFilter(query, finalCandidatesList, k, threshold);
        exhaustiveTime += System.currentTimeMillis() - start;

        return finalTopK;
    }

    private long recursiveTreeMemoryUsage(Node<CandidateSegment, Geometry> node) {
        if (node instanceof NonLeafDefault) {
            NonLeafDefault<CandidateSegment, Geometry> nonLeaf =
                    (NonLeafDefault<CandidateSegment, Geometry>) node;
            long memoryUsage = tree.dimensions() * 8L * 2L + 4L * 2L; // mins, maxes, 2 pointers
            for (Node<CandidateSegment, Geometry> child : nonLeaf.children()) {
                memoryUsage += recursiveTreeMemoryUsage(child);
            }
            return memoryUsage;
        } else if (node instanceof LeafDefault) {
            LeafDefault<CandidateSegment, Geometry> leaf =
                    (LeafDefault<CandidateSegment, Geometry>) node;
            long memoryUsage = tree.dimensions() * 8L * 2L; // mins, maxes
            memoryUsage += leaf.count() * tree.dimensions() * 8L * 2L
                    + leaf.count() * (4L * 3L
                    + (kMeansClusters == 0 ? 0
                    : channels * 2L * 8L + channels * 4L)); // metadata per entry
            return memoryUsage;
        }
        throw new IllegalStateException("Unknown node type " + node.getClass());
    }

    public double memoryUsage() {
        return recursiveTreeMemoryUsage(tree.root().get()) / 1_000_000d;
    }
}
