package io.github.utils;
import io.github.io.DataManager;

import java.util.Comparator;

import static io.github.io.DataManager.timeSeriesStarts;
import static io.github.utils.Parameters.qLen;

public class CandidateMVSubsequence {
    public final int timeSeriesIndex;
    public final int subSequenceIndex;
    public double totalDistance = Double.MAX_VALUE;

    public double[] dimDistances;

    public CandidateMVSubsequence(int timeSeriesIndex, int subSequenceIndex) {
        this.timeSeriesIndex = timeSeriesIndex;
        this.subSequenceIndex = subSequenceIndex;
    }

    public CandidateMVSubsequence(int timeSeriesIndex, int subSequenceIndex, double totalDistance) {
        this.timeSeriesIndex = timeSeriesIndex;
        this.subSequenceIndex = subSequenceIndex;
        this.totalDistance = totalDistance;
    }

    public double totalDistance() {
        return totalDistance;
    }

    public String toString() {
        return String.format("(T_{%d, %d}, %.3f)", timeSeriesIndex, subSequenceIndex, totalDistance);
    }

    public static Comparator<CandidateMVSubsequence> compareByTotalDistance() {
        return Comparator.comparingDouble(a -> a.totalDistance);
    }

    public static Comparator<CandidateMVSubsequence> compareByTotalDistanceReversed() {
        return (a, b) -> Double.compare(b.totalDistance, a.totalDistance);
    }

    @Override
    public boolean equals(Object o) {
        if (o == this) {
            return true;
        }
        if (!(o instanceof CandidateMVSubsequence)) {
            return false;
        }
        CandidateMVSubsequence other = (CandidateMVSubsequence) o;
        // Ignore if the distance is the same as a top k with the same distances is allowed
        return other.timeSeriesIndex == timeSeriesIndex && other.subSequenceIndex == subSequenceIndex;
    }

    @Override
    public int hashCode() {
        return timeSeriesStarts[timeSeriesIndex] + subSequenceIndex;
    }
    public int length() {
        return qLen;
    }

    public CandidateSegment toCandidateSegment() {
        return new CandidateSegment(timeSeriesIndex, subSequenceIndex, subSequenceIndex, null);
    }

    public double[][] getData(){
        return DataManager.getSubSequence(timeSeriesIndex, subSequenceIndex);
    }

    public double getDimDist(int dimension) {
        if (dimDistances == null) {
            return Double.MAX_VALUE;
        }
        return dimDistances[dimension];
    }


    public void setDimDist(int dimension, double distance) {
        if (dimDistances == null) {
            dimDistances = new double[Parameters.channels];
        }
        dimDistances[dimension] = distance;
    }

    public int getTimeSeriesIndex() {
        return timeSeriesIndex;
    }

    public int getSubSequenceIndex() {
        return subSequenceIndex;
    }

    public double[] getDimDistances() {
        return dimDistances;
    }

    public void setDimDistances(double[] dimDistances) {
        this.dimDistances = dimDistances;
    }

}
