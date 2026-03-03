package io.github.algorithms.msindexpp.segmentation;

import io.github.algorithms.msindexpp.MinimumBoundingRectangle;
import io.github.utils.CandidateSegment;
import io.github.utils.FourierTrail;
import io.github.utils.LandmarkMBR;
import io.github.utils.LandmarkPortfolio;
import io.github.utils.rtreemulti.Entry;
import io.github.utils.rtreemulti.geometry.Geometry;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SingletonSegment extends Segmentation {
    private final boolean withMBR;

    public SingletonSegment(boolean withMBR) {
        this.withMBR = withMBR;
    }

    /**
     * Singleton segmentation of the time series. Used to be equi-depth with volume 0 but this is optimized for this case.
     */
    public List<Entry<CandidateSegment, Geometry>> segment(int idx, FourierTrail fourierTrail) {
        final double[][] trail = fourierTrail.getTrail();
        final LandmarkPortfolio[] portfolios = fourierTrail.getLandmarkPortfolios();

        if (trail.length == 0) {
            return Collections.emptyList();
        }

        final ArrayList<Entry<CandidateSegment, Geometry>> entries = new ArrayList<>(trail.length);

        for (int i = 0; i < trail.length; i++) {
            double[] point = trail[i];

            LandmarkMBR landmarkMBR = null;
            if (withMBR) landmarkMBR = new LandmarkMBR(portfolios[i]);
            MinimumBoundingRectangle mbr = MinimumBoundingRectangle.create(point, point, landmarkMBR);
            CandidateSegment segment = new CandidateSegment(idx, i, i, mbr);
            entries.add(Entry.entry(segment, mbr));
        }

        return entries;
    }

}
