package io.github.algorithms.msindexpp;

import lombok.Getter;
import lombok.Setter;

import io.github.utils.DistanceBound;
import io.github.utils.LandmarkMBR;
import io.github.utils.lib;
import io.github.utils.rtreemulti.geometry.Point;
import io.github.utils.rtreemulti.geometry.Rectangle;
import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;

import static io.github.utils.Parameters.*;

public class MinimumBoundingRectangle
        extends io.github.algorithms.msindex.MinimumBoundingRectangle {

    @Setter @Getter
    private LandmarkMBR landmarkMBR;

    public Double[] distanceCache;


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

    @Override
    public io.github.algorithms.msindexpp.MinimumBoundingRectangle add(Rectangle r) {
        io.github.algorithms.msindex.MinimumBoundingRectangle base = super.add(r);
        return new io.github.algorithms.msindexpp.MinimumBoundingRectangle(
                base.mins(),
                base.maxes(),
                this.landmarkMBR
        );
    }

    @Override
    public double distance(Rectangle r) {
        if (distanceCache == null) {
            distanceCache = new Double[nQueries];
        }
        Double cached = distanceCache[currQueryId];
        if (cached != null) {
            return cached;
        }

        nBoundComputations.getAndIncrement();

        if (r instanceof io.github.algorithms.msindexpp.MinimumBoundingRectangle) {
            io.github.algorithms.msindexpp.MinimumBoundingRectangle mbr =
                    (io.github.algorithms.msindexpp.MinimumBoundingRectangle) r;

            final double dftLB = dftBoundGeneric(mbr).LB();
            double LB = dftLB;

            if (this.landmarkMBR != null && mbr.landmarkMBR != null) {
                LB += this.landmarkMBR.getDistance(mbr.landmarkMBR).LB();
            }

            distanceCache[currQueryId] = LB;
            return LB;
        }

        if (r instanceof Rectangle) {
            double LB = dftBoundGeneric(r).LB();
            distanceCache[currQueryId] = LB;
            return LB;
        }

        if (r instanceof Point) {
            Point p = (Point) r;
            double[] x = p.mins();
            double dist = 0.0;

            for (int d : selectedVariatesIdx) {
                int offset = d * fourierLengthPerDimension();
                for (int i = offset; i < offset + fourierLengthPerDimension(); ++i) {
                    double maxDiff = x[i] - this.maxes()[i];
                    double minDiff = this.mins()[i] - x[i];
                    double diff = FastMath.max(0, FastMath.max(maxDiff, minDiff));
                    dist += diff * diff;
                }
            }
            distanceCache[currQueryId] = dist;
            return dist;
        }

        throw new IllegalArgumentException("Unsupported rectangle type: " + r.getClass());
    }

    public DistanceBound dftBoundGeneric(Rectangle r) {
        final int doublesPerDimension = fourierLengthPerDimension();

        final double[] a = r.mins();
        final double[] b = r.maxes();

        double LB = 0.0;
        double UB = 0.0;

        final double[] thisMins = this.mins();
        final double[] thisMaxes = this.maxes();

        for (int d : selectedVariatesIdx) {
            int offset = d * doublesPerDimension;
            for (int i = offset; i < offset + doublesPerDimension; ++i) {
                final double ay = a[i] - thisMaxes[i];
                final double bx = b[i] - thisMins[i];
                final double ayAbs = FastMath.abs(ay);
                final double bxAbs = FastMath.abs(bx);

                final double minDiff;
                if (!(ay > 0 || bx < 0)) {
                    minDiff = 0.0;
                } else {
                    minDiff = lib.pow2(FastMath.min(ayAbs, bxAbs));
                }

                final double maxDiff = lib.pow2(FastMath.max(ayAbs, bxAbs));

                LB += minDiff;
                UB += maxDiff;

                int coefId = (i - offset) / 2;
                if (fftCoefficients[coefId] != 0) {
                    LB += minDiff;
                    UB += maxDiff;
                }
            }
        }

        return new DistanceBound(LB / qLen, UB / qLen);
    }

    @Override
    public String toString() {
        return "PP-Rectangle [mins=" + Arrays.toString(this.mins()) + ", maxes=" + Arrays.toString(this.maxes()) + "]";
    }
}
