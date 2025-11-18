package io.github.algorithms.msindexpp;

import lombok.Getter;
import lombok.Setter;
import io.github.utils.LandmarkMBR;
import io.github.utils.rtreemulti.geometry.Rectangle;


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
        // Κάνε την πρόσθεση με το original super
        io.github.algorithms.msindex.MinimumBoundingRectangle base = super.add(r);
        // Τύλιξέ το σε PP instance (κρατάς και τυχόν PP metadata αν χρειάζεται)
        return new io.github.algorithms.msindexpp.MinimumBoundingRectangle(
                base.mins(),
                base.maxes(),
                this.landmarkMBR
        );
    }

}
