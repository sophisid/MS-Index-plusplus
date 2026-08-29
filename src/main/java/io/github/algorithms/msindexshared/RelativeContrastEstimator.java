package io.github.algorithms.msindexshared;

import io.github.io.DataManager;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import static io.github.utils.Parameters.K;
import static io.github.utils.Parameters.qLen;
import static io.github.utils.Parameters.normalize;
import static io.github.utils.Parameters.selectedVariatesIdx;

/** Estimates query relative contrast over every candidate window. */
public final class RelativeContrastEstimator {
    private static final double EPS = 1e-12;

    private RelativeContrastEstimator() {
    }

    public static double estimate(double[][] query) {
        Map<Integer, Double> estimates = estimateAll(query, K);
        return estimates.getOrDefault(K, 1.0);
    }

    /** Computes RC for several k values during one complete distance scan. */
    public static Map<Integer, Double> estimateAll(double[][] query, int... requestedKs) {
        Map<Integer, Double> result = new LinkedHashMap<>();
        if (requestedKs == null || requestedKs.length == 0) {
            return result;
        }

        int maximumRank = 1;
        for (int requestedK : requestedKs) {
            maximumRank = Math.max(maximumRank, requestedK);
        }

        double[][][] data = DataManager.data;
        if (data.length == 0) {
            for (int requestedK : requestedKs) {
                result.put(requestedK, 1.0);
            }
            return result;
        }

        PriorityQueue<Double> nearestDistances = new PriorityQueue<>(maximumRank, Comparator.reverseOrder());
        double distanceSum = 0.0;
        long candidateCount = 0;

        for (int timeSeriesId = 0; timeSeriesId < data.length; timeSeriesId++) {
            int seriesLength = DataManager.getM(timeSeriesId);
            int limit = seriesLength - qLen + 1;
            if (limit <= 0) {
                continue;
            }

            for (int start = 0; start < limit; start++) {
                double distanceSquared = 0.0;
                for (int channel : selectedVariatesIdx) {
                    double distance = windowDistance(data[timeSeriesId][channel], start, query[channel], normalize);
                    distanceSquared += distance * distance;
                }
                double distance = Math.sqrt(distanceSquared);
                if (!Double.isFinite(distance)) {
                    for (int requestedK : requestedKs) {
                        result.put(requestedK, 1.0);
                    }
                    return result;
                }
                distanceSum += distance;
                candidateCount++;

                if (nearestDistances.size() < maximumRank) {
                    nearestDistances.offer(distance);
                } else if (distance < nearestDistances.peek()) {
                    nearestDistances.poll();
                    nearestDistances.offer(distance);
                }
            }
        }

        if (candidateCount == 0 || !Double.isFinite(distanceSum) || nearestDistances.isEmpty()) {
            for (int requestedK : requestedKs) {
                result.put(requestedK, 1.0);
            }
            return result;
        }

        List<Double> nearestSorted = new ArrayList<>(nearestDistances);
        nearestSorted.sort(Comparator.naturalOrder());
        double meanDistance = distanceSum / candidateCount;
        for (int requestedK : requestedKs) {
            int rank = Math.max(requestedK, 1);
            int index = Math.min(rank, nearestSorted.size()) - 1;
            double kthNearestDistance = nearestSorted.get(index);
            result.put(requestedK, meanDistance / Math.max(kthNearestDistance, EPS));
        }
        return result;
    }

    private static double windowDistance(double[] series, int start, double[] query, boolean zNormalize) {
        if (!zNormalize) {
            double sum = 0.0;
            for (int i = 0; i < qLen; i++) {
                double difference = series[start + i] - query[i];
                sum += difference * difference;
            }
            return Math.sqrt(sum);
        }

        double seriesMean = 0.0;
        double queryMean = 0.0;
        for (int i = 0; i < qLen; i++) {
            seriesMean += series[start + i];
            queryMean += query[i];
        }
        seriesMean /= qLen;
        queryMean /= qLen;

        double seriesVariance = 0.0;
        double queryVariance = 0.0;
        for (int i = 0; i < qLen; i++) {
            double seriesDifference = series[start + i] - seriesMean;
            double queryDifference = query[i] - queryMean;
            seriesVariance += seriesDifference * seriesDifference;
            queryVariance += queryDifference * queryDifference;
        }
        seriesVariance = Math.sqrt(seriesVariance / qLen);
        queryVariance = Math.sqrt(queryVariance / qLen);

        double sum = 0.0;
        for (int i = 0; i < qLen; i++) {
            double seriesValue = (series[start + i] - seriesMean) / Math.max(seriesVariance, EPS);
            double queryValue = (query[i] - queryMean) / Math.max(queryVariance, EPS);
            double difference = seriesValue - queryValue;
            sum += difference * difference;
        }
        return Math.sqrt(sum);
    }
}
