package io.github.algorithms.msindexpp.ml;

import io.github.utils.Parameters;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class TraversalFeatureSnapshot {
    public static final List<String> FEATURE_NAMES = List.of(
            "requested_k",
            "checkpoint_id",
            "distance_computations",
            "candidate_count",
            "unique_series_count",
            "nodes_completed",
            "leafs_completed",
            "non_leaf_expansions",
            "children_enqueued",
            "entries_scanned",
            "frontier_size",
            "frontier_best_distance",
            "frontier_avg_distance",
            "frontier_worst_distance",
            "last_distance",
            "last_candidate_distance",
            "threshold",
            "threshold_minus_frontier_best",
            "candidate_rate",
            "enqueue_per_step",
            "nodes_completed_ratio",
            "leafs_completed_ratio",
            "frontier_ratio"
    );

    public final int requestedK;
    public final int checkpointId;
    public final long distanceComputations;
    public final int candidateCount;
    public final int uniqueSeriesCount;
    public final long nodesCompleted;
    public final long leafsCompleted;
    public final long nonLeafExpansions;
    public final long childrenEnqueued;
    public final long entriesScanned;
    public final int frontierSize;
    public final double frontierBestDistance;
    public final double frontierAvgDistance;
    public final double frontierWorstDistance;
    public final double lastDistance;
    public final double lastCandidateDistance;
    public final double threshold;

    public TraversalFeatureSnapshot(int requestedK, int checkpointId, long distanceComputations, int candidateCount,
                                    int uniqueSeriesCount, long nodesCompleted, long leafsCompleted,
                                    long nonLeafExpansions, long childrenEnqueued, long entriesScanned,
                                    int frontierSize, double frontierBestDistance, double frontierAvgDistance,
                                    double frontierWorstDistance, double lastDistance, double lastCandidateDistance,
                                    double threshold) {
        this.requestedK = requestedK;
        this.checkpointId = checkpointId;
        this.distanceComputations = distanceComputations;
        this.candidateCount = candidateCount;
        this.uniqueSeriesCount = uniqueSeriesCount;
        this.nodesCompleted = nodesCompleted;
        this.leafsCompleted = leafsCompleted;
        this.nonLeafExpansions = nonLeafExpansions;
        this.childrenEnqueued = childrenEnqueued;
        this.entriesScanned = entriesScanned;
        this.frontierSize = frontierSize;
        this.frontierBestDistance = frontierBestDistance;
        this.frontierAvgDistance = frontierAvgDistance;
        this.frontierWorstDistance = frontierWorstDistance;
        this.lastDistance = lastDistance;
        this.lastCandidateDistance = lastCandidateDistance;
        this.threshold = threshold;
    }

    public Map<String, Double> toFeatureMap() {
        double thresholdMinusFrontierBest = frontierBestDistance == Double.POSITIVE_INFINITY
                ? threshold
                : threshold - frontierBestDistance;
        double candidateRate = distanceComputations == 0 ? 0d : candidateCount / (double) distanceComputations;
        double enqueuePerStep = distanceComputations == 0 ? 0d : childrenEnqueued / (double) distanceComputations;
        double nodesCompletedRatio = Parameters.nNodes == 0 ? 0d : nodesCompleted / (double) Parameters.nNodes;
        double leafsCompletedRatio = Parameters.nLeafs == 0 ? 0d : leafsCompleted / (double) Parameters.nLeafs;
        double frontierRatio = Parameters.nNodes == 0 ? 0d : frontierSize / (double) Parameters.nNodes;
        LinkedHashMap<String, Double> featureMap = new LinkedHashMap<>();
        featureMap.put("requested_k", (double) requestedK);
        featureMap.put("checkpoint_id", (double) checkpointId);
        featureMap.put("distance_computations", (double) distanceComputations);
        featureMap.put("candidate_count", (double) candidateCount);
        featureMap.put("unique_series_count", (double) uniqueSeriesCount);
        featureMap.put("nodes_completed", (double) nodesCompleted);
        featureMap.put("leafs_completed", (double) leafsCompleted);
        featureMap.put("non_leaf_expansions", (double) nonLeafExpansions);
        featureMap.put("children_enqueued", (double) childrenEnqueued);
        featureMap.put("entries_scanned", (double) entriesScanned);
        featureMap.put("frontier_size", (double) frontierSize);
        featureMap.put("frontier_best_distance", frontierBestDistance);
        featureMap.put("frontier_avg_distance", frontierAvgDistance);
        featureMap.put("frontier_worst_distance", frontierWorstDistance);
        featureMap.put("last_distance", lastDistance);
        featureMap.put("last_candidate_distance", lastCandidateDistance);
        featureMap.put("threshold", threshold);
        featureMap.put("threshold_minus_frontier_best", thresholdMinusFrontierBest);
        featureMap.put("candidate_rate", candidateRate);
        featureMap.put("enqueue_per_step", enqueuePerStep);
        featureMap.put("nodes_completed_ratio", nodesCompletedRatio);
        featureMap.put("leafs_completed_ratio", leafsCompletedRatio);
        featureMap.put("frontier_ratio", frontierRatio);
        return featureMap;
    }
}
