package io.github.algorithms.msindexpp;

import lombok.Getter;
import lombok.Setter;
import io.github.utils.LandmarkMBR;
import io.github.utils.rtreemulti.geometry.Rectangle;

/**
 * MinimumBoundingRectangle για MS-Index++ που επεκτείνει το MBR του msindex
 * και προσθέτει υποστήριξη για LandmarkMBR + στοχευμένο inflate.
 */
public class MinimumBoundingRectangle
        extends io.github.algorithms.msindex.MinimumBoundingRectangle {

    // Πρόσθετα PP fields
    @Setter @Getter
    private LandmarkMBR landmarkMBR;

    // Προαιρετικό per-query cache
    public Double[] distanceCache;
    private static final ThreadLocal<Double> TL_KTH_SQ = ThreadLocal.withInitial(() -> Double.POSITIVE_INFINITY);
    public static void setEarlyAbandonLimitSq(double v) { TL_KTH_SQ.set(v); }

    // --- Constructors ---

    public MinimumBoundingRectangle(double[] mins, double[] maxes) {
        super(mins, maxes);
    }

    public MinimumBoundingRectangle(double[] mins, double[] maxes, LandmarkMBR landmarkMBR) {
        super(mins, maxes);
        this.landmarkMBR = landmarkMBR;
    }

    public static MinimumBoundingRectangle create(double[] mins, double[] maxes) {
        return new MinimumBoundingRectangle(mins, maxes);
    }

    public static MinimumBoundingRectangle create(double[] mins, double[] maxes, LandmarkMBR landmarkMBR) {
        return new MinimumBoundingRectangle(mins, maxes, landmarkMBR);
    }

    // --- Factory: MBR από σημείο (mins=maxes=point) ---
    public static MinimumBoundingRectangle fromPoint(double[] point) {
        double[] mins = point.clone();
        double[] maxs = point.clone();
        return new MinimumBoundingRectangle(mins, maxs, null);
    }

    // --- Covariant override για add(Rectangle) ώστε να επιστρέφει PP MBR ---
    @Override
    public io.github.algorithms.msindexpp.MinimumBoundingRectangle add(Rectangle r) {
        io.github.algorithms.msindex.MinimumBoundingRectangle base = super.add(r);
        return new io.github.algorithms.msindexpp.MinimumBoundingRectangle(
                base.mins(),
                base.maxes(),
                this.landmarkMBR
        );
    }

    /**
     * Επιλεκτικό “φούσκωμα” του MBR σε top-|z| διαστάσεις.
     *
     * @param zLike        vector (ίδιο μήκος με mins/maxes) με “ένταση” ανά διάσταση (π.χ. |z|)
     * @param inflateFact  π.χ. 1.1–1.5
     * @param topK         πόσες διαστάσεις να επηρεάσει
     */
    public void inflateSelectiveTopZ(double[] zLike, double inflateFact, int topK) {
        double[] mins = mins();
        double[] maxs = maxes();
        if (zLike == null || mins == null || maxs == null) return;

        int D = Math.min(Math.min(zLike.length, mins.length), maxs.length);
        if (D == 0) return;

        // indices 0..D-1 ταξινομημένα κατά φθίνουσα |z|
        Integer[] idx = new Integer[D];
        for (int i = 0; i < D; i++) idx[i] = i;
        java.util.Arrays.sort(idx, (a, b) -> Double.compare(Math.abs(zLike[b]), Math.abs(zLike[a])));

        int use = Math.min(Math.max(topK, 0), D);
        for (int t = 0; t < use; t++) {
            int i = idx[t];
            double mid  = 0.5 * (mins[i] + maxs[i]);
            double half = 0.5 * (maxs[i] - mins[i]);
            if (half == 0.0) half = Math.abs(mid) * 1e-9 + 1e-9; // μικρό padding
            double newHalf = half * inflateFact;
            mins[i] = mid - newHalf;
            maxs[i] = mid + newHalf;
        }
    }

    /** Γενικό inflate μόνο σε συγκεκριμένες διαστάσεις. */
    public void inflateSelective(double factor, int[] dims) {
        if (factor <= 0.0 || dims == null || dims.length == 0) return;
        double[] mins = mins();
        double[] maxs = maxes();
        for (int d : dims) {
            double c  = 0.5 * (maxs[d] + mins[d]);
            double r  = 0.5 * (maxs[d] - mins[d]);
            double nr = r * factor;
            mins[d] = c - nr;
            maxs[d] = c + nr;
        }
    }

    /** Επιστρέφει το LandmarkMBR που κουβαλά το MBR (αν έχει). */
    public io.github.utils.LandmarkMBR getLandmarkMBR() {
        return this.landmarkMBR;
    }

    public double distanceSquaredEarly(double[] q) {
        final double[] mins = mins(); // υποθέτω getters που επιστρέφουν το εσωτερικό array
        final double[] maxs = maxes();
        double acc = 0.0;
        final double kthSq = TL_KTH_SQ.get();
        final int d = mins.length;
        for (int i = 0; i < d; i++) {
            final double qi = q[i];
            double diff = 0.0;
            if (qi < mins[i])      diff = mins[i] - qi;
            else if (qi > maxs[i]) diff = qi - maxs[i];
            acc += diff * diff;
            if (acc >= kthSq) {
                // πρόωρη διακοπή: δεν γίνεται να βελτιώσει το k-th
                return acc;
            }
        }
        return acc;
    }
}
