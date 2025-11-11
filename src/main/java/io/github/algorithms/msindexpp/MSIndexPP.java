package io.github.algorithms.msindexpp;

import io.github.algorithms.Algorithm;
import io.github.utils.*;
import io.github.utils.rtreemulti.Entry;
import io.github.utils.rtreemulti.RTree;
import io.github.utils.rtreemulti.geometry.Geometry;
import io.github.algorithms.msindexpp.OODStats;
import io.github.algorithms.msindexpp.segmentation.AdhocSegment;
import io.github.utils.FourierTrail;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.logging.Logger;

import static java.util.stream.Collectors.toList;
import static io.github.utils.Parameters.*;

public class MSIndexPP extends Algorithm {
    final KMeans[] kMeans = new KMeans[channels];
    RTree<CandidateSegment, Geometry> tree;
    private final int treeDimensions = fourierLength();

    private boolean buildTree = true;
    private OODStats oodStats; // Χτίζεται στο beforeQuerying()

    public MSIndexPP(boolean buildTree) {
        this.buildTree = buildTree;
        DFTUtils.computeKMeansClusters(kMeans);
    }

    @Override
    public String getIndexPath() {
        java.io.File folder = new java.io.File("index_caches");
        if (!folder.exists()) folder.mkdir();

        return "index_caches/" + String.join("_",
                algorithmType.toString(),
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

    private void createTree(List<Entry<CandidateSegment, Geometry>> items){
        Logger.getGlobal().info("[MS-Index++] Building the R-tree [MAIN]");
        tree = RTree.star().dimensions(treeDimensions)
                .leafSize(buildTree ? indexLeafSize : Integer.MAX_VALUE)
                .create(items);

        if (segmentMethod.equals(io.github.algorithms.msindexpp.segmentation.SegmentMethods.ADHOC)) {
            Logger.getGlobal().info("[MS-Index++] Post-index segmentation");
            io.github.algorithms.msindexpp.segmentation.AdhocSegment.postIndexSegment(tree);
        }
    }

    @Override
    public void buildIndex() {
        Logger.getGlobal().info("[MS-Index++] Segmenting the fourier trails");
        final List<Entry<CandidateSegment, Geometry>> items = DFTUtils.getSegmentedFourierTrail(kMeans);
        createTree(items);

        // OOD stats once, here:
        try {
            FourierTrail[] trailsArr = io.github.algorithms.msindexpp.segmentation.AdhocSegment.getFourierTrails();
            java.util.ArrayList<double[][]> present = new java.util.ArrayList<>();
            if (trailsArr != null) {
                for (FourierTrail ft : trailsArr) if (ft != null && ft.getTrail() != null) present.add(ft.getTrail());
            }
            if (!present.isEmpty()) {
                double[][][] trails = present.toArray(new double[0][][]);
                this.oodStats = OODStats.fromTrails(trails, fftCoefficients); // προϋποθέτει flat mean/std
                Logger.getGlobal().info("[MS-Index++] OOD stats ready over " + present.size() + " trails.");
            } else {
                this.oodStats = null;
                Logger.getGlobal().info("[MS-Index++] OOD disabled (no trails).");
            }
        } catch (Throwable t) {
            this.oodStats = null;
            Logger.getGlobal().warning("[MS-Index++] OOD disabled due to error: " + t.getMessage());
        }
    }


    @Override
    public void saveIndex() {
        String filename = getIndexPath();
        try (java.io.FileOutputStream fileOut = new java.io.FileOutputStream(filename);
             java.io.ObjectOutputStream out = new java.io.ObjectOutputStream(fileOut)) {
            Logger.getGlobal().info("[MS-Index++] Saving the index to file");
            out.writeObject(tree);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public Algorithm loadIndex() {
        String filename = getIndexPath();
        try (java.io.FileInputStream fileIn = new java.io.FileInputStream(filename);
             java.io.ObjectInputStream in = new java.io.ObjectInputStream(fileIn)) {
            Logger.getGlobal().info("[MS-Index++] Loading the index from file");
            //noinspection unchecked
            tree = (RTree<CandidateSegment, Geometry>) in.readObject();
            Logger.getGlobal().info("[MS-Index++] Index loaded");
        } catch (Exception e) {
            e.printStackTrace();
        }
        return this;
    }

    // ------------------------------------------------------------
    // OOD: Υπολογισμός στατιστικών ΜΙΑ ΦΟΡΑ πριν αρχίσει το querying
    // ------------------------------------------------------------
    @Override
    protected void beforeQuerying() {
        // Παίρνουμε τα Fourier trails που γέμισε το segmentation
        FourierTrail[] trailsArr = io.github.algorithms.msindexpp.segmentation.AdhocSegment.getFourierTrails();

        if (trailsArr == null || trailsArr.length == 0) {
            Logger.getGlobal().warning("[MS-Index++] OOD: trails array is null/empty; disabling OOD for this run.");
            this.oodStats = null;
            return;
        }

        // Μαζεύουμε ΜΟΝΟ τα μη-NULL trails (κάποια indices μπορεί να μην γέμισαν)
        java.util.ArrayList<double[][]> present = new java.util.ArrayList<>();
        int nulls = 0;
        for (int i = 0; i < trailsArr.length; i++) {
            FourierTrail ft = trailsArr[i];
            if (ft == null) { nulls++; continue; }
            double[][] t = ft.getTrail();  // [time][coeff]
            if (t != null) present.add(t);
        }

        if (present.isEmpty()) {
            Logger.getGlobal().warning("[MS-Index++] OOD: no non-null trails present; disabling OOD for this run.");
            this.oodStats = null;
            return;
        }

        if (nulls > 0) {
            Logger.getGlobal().info("[MS-Index++] OOD: ignoring " + nulls + " null Fourier trails ("
                    + present.size() + " present).");
        }

        double[][][] trails = present.toArray(new double[0][][]);
        this.oodStats = OODStats.fromTrails(trails, fftCoefficients);
    }


    // ------------------------------------------------------------
    // KNN με OOD gate + selective inflate + relaxed pruning
    // ------------------------------------------------------------
    @Override
    public ArrayList<CandidateMVSubsequence> kNN(int k, double[][] query) {
        final Tuple2<double[], LandmarkPortfolio> qDFTs = DFTUtils.getQueryDFTs(query, kMeans);
        final double[] fftQFlat = qDFTs._1().clone();
        DFTUtils.applyWhitenAndWeight(fftQFlat);

        boolean isOOD = false;
        double zMean=0, zMax=0;
        if (oodGateEnabled && this.oodStats != null) {
            double[] zBuf = new double[2];
            this.oodStats.zScores(fftQFlat, zBuf);
            zMean = zBuf[0]; zMax = zBuf[1];
            isOOD = (zMean > oodZMeanThr) || (zMax > oodZMaxThr);
            Logger.getGlobal().info(String.format("[MS-Index++] OOD score: mean=%.3f max=%.3f  -> %s",
                    zMean, zMax, isOOD ? "OOD" : "IN"));
        }

        final MinimumBoundingRectangle queryMBR = MinimumBoundingRectangle.fromPoint(fftQFlat);

        if (isOOD) {
            // adaptive inflate: όσο υπερβαίνει το threshold
            final double overMean = Math.max(0.0, zMean - oodZMeanThr);
            final double overMax  = Math.max(0.0, zMax  - oodZMaxThr);
            final double severity = Math.max(overMean / Math.max(1e-6, oodZMeanThr),
                                            overMax  / Math.max(1e-6, oodZMaxThr));
            // clamp 0..1
            final double s = Math.max(0.0, Math.min(1.0, severity));
            final double inflate = 1.0 + s * (oodInflateFactor - 1.0);

            // top-m διαστάσεις χωρίς sort
            final int m = Math.min(8, fftQFlat.length); // κάν’ το param αν θες
            int[] topIdx = topKByAbs(fftQFlat, m);
            queryMBR.inflateSelective(inflate, topIdx);
        }

        // First pass
        final ArrayList<CandidateMVSubsequence> initialTopK = firstPass(k, query, queryMBR);
        if (initialTopK.isEmpty()) return initialTopK;

        final double kThDistance = initialTopK.get(k - 1).totalDistance;
        Logger.getGlobal().info("[MS-Index++] k-th distance after initial topK: " + kThDistance);

        // Ελαφρύτερος δεύτερος γύρος αν OOD αλλά οριακά
        final boolean mildOOD = isOOD && (zMean < (oodZMeanThr + 0.4)) && (zMax < (oodZMaxThr + 0.6));
        final double eps = isOOD ? (mildOOD ? 0.015 : 0.01) : 0.03;
        final double threshold = kThDistance * (1.0 + eps);

        return secondPass(k, query, queryMBR, threshold);
    }


    // ------------------------------------------------------------
    // Passes & helpers
    // ------------------------------------------------------------
    private ArrayList<CandidateMVSubsequence> firstPass(int k, double[][] query, MinimumBoundingRectangle queryMBR) {
        long start = System.currentTimeMillis();
        final List<CandidateSegment> candidateSegments = closest(queryMBR, k);
        indexSearchTime += System.currentTimeMillis() - start;

        start = System.currentTimeMillis();
        ArrayList<CandidateMVSubsequence> initialTopK =
                postFilter(query, candidateSegments, k, Double.POSITIVE_INFINITY);
        exhaustiveTime += System.currentTimeMillis() - start;

        return initialTopK;
    }

    private ArrayList<CandidateMVSubsequence> secondPass(int k, double[][] query, MinimumBoundingRectangle queryMBR, double threshold) {
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

    // MSIndexPP.java (μέσα στην κλάση)
    private static int[] topKByAbs(double[] v, int m){
        // επιστρέφει indices των m μεγαλύτερων κατά |v|
        // απλό partial selection χωρίς allocations
        m = Math.min(m, v.length);
        int[] idx = new int[v.length];
        for (int i = 0; i < v.length; i++) idx[i] = i;
        // quickselect on |v|
        int left = 0, right = v.length - 1, kth = v.length - m;
        java.util.Random rnd = new java.util.Random(1234567);
        while (left <= right) {
            int pivot = left + rnd.nextInt(right - left + 1);
            double pv = Math.abs(v[idx[pivot]]);
            // partition
            int i = left, j = right;
            while (i <= j) {
                while (Math.abs(v[idx[i]]) < pv) i++;
                while (Math.abs(v[idx[j]]) > pv) j--;
                if (i <= j) { int t = idx[i]; idx[i]=idx[j]; idx[j]=t; i++; j--; }
            }
            if (kth < i) right = i - 1;
            if (kth > j) left  = j + 1;
            if (kth >= i && kth <= j) break;
        }
        // τα m μεγαλύτερα είναι στα τελευταία m
        int[] top = new int[m];
        System.arraycopy(idx, v.length - m, top, 0, m);
        return top;
    }

}
