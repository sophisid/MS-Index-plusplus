package io.github.algorithms.msindex;

import io.github.utils.LandmarkMBR;
import io.github.utils.lib;
import io.github.utils.rtreemulti.geometry.Rectangle;

public class MinimumBoundingRectangle extends io.github.algorithms.msindexshared.MinimumBoundingRectangle {

    public MinimumBoundingRectangle(double[] mins, double[] maxes) {
        super(mins, maxes);
    }

    public MinimumBoundingRectangle(double[] mins, double[] maxes, LandmarkMBR landmarkMBR) {
        super(mins, maxes, landmarkMBR);
    }

    public static MinimumBoundingRectangle create(double[] mins, double[] maxes) {
        return new MinimumBoundingRectangle(mins, maxes);
    }

    public static MinimumBoundingRectangle create(double[] mins, double[] maxes, LandmarkMBR landmarkMBR) {
        return new MinimumBoundingRectangle(mins, maxes, landmarkMBR);
    }

    @Override
    public MinimumBoundingRectangle add(Rectangle r) {
        final double[] newMins = lib.minimum(this.mins(), r.mins());
        final double[] newMaxs = lib.maximum(this.maxes(), r.maxes());
        return new MinimumBoundingRectangle(newMins, newMaxs);
    }
}
