package io.github.algorithms.msindexpp.ml;

import io.github.utils.CandidateSegment;
import io.github.utils.Parameters;
import io.github.utils.rtreemulti.Entry;
import io.github.utils.rtreemulti.Leaf;
import io.github.utils.rtreemulti.Node;
import io.github.utils.rtreemulti.NonLeaf;
import io.github.utils.rtreemulti.RTree;
import io.github.utils.rtreemulti.geometry.Geometry;
import io.github.utils.rtreemulti.geometry.Rectangle;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.PriorityQueue;
import java.util.Set;

public class SecondPassTraversal {

    public static TraversalResult traverse(RTree<CandidateSegment, Geometry> tree,
                                           Rectangle queryMBR,
                                           double threshold,
                                           int requestedK,
                                           int checkpointInterval,
                                           CheckpointDecider checkpointDecider,
                                           boolean collectCheckpoints) {
        Optional<? extends Node<CandidateSegment, Geometry>> root = tree.root();
        if (root.isEmpty()) {
            return new TraversalResult(new ArrayList<>(), new ArrayList<>(), false, -1d, 0);
        }

        PriorityQueue<SearchNodeState> frontier = new PriorityQueue<>(Comparator.comparingDouble(SearchNodeState::priority));
        frontier.add(new SearchNodeState(root.get(), 0, 0d));

        TraversalStats stats = new TraversalStats(threshold, requestedK);
        List<CandidateSegment> acceptedCandidates = new ArrayList<>();
        List<CheckpointRecord> checkpoints = new ArrayList<>();

        int safeCheckpointInterval = Math.max(1, checkpointInterval);
        while (!frontier.isEmpty()) {
            SearchNodeState current = frontier.peek();
            if (current.priority() > threshold) {
                break;
            }

            if (!current.hasRemaining()) {
                frontier.poll();
                stats.completeNode(current.node().isLeaf());
                continue;
            }

            if (current.node().isLeaf()) {
                Leaf<CandidateSegment, Geometry> leaf = (Leaf<CandidateSegment, Geometry>) current.node();
                Entry<CandidateSegment, Geometry> entry = leaf.entry(current.position());
                current.advance();
                double distance = entry.geometry().distance(queryMBR);
                stats.recordEntryScan(distance);
                if (distance <= threshold) {
                    acceptedCandidates.add(entry.value());
                    stats.recordAcceptedCandidate(entry.value(), distance);
                }
            } else {
                NonLeaf<CandidateSegment, Geometry> nonLeaf = (NonLeaf<CandidateSegment, Geometry>) current.node();
                Node<CandidateSegment, Geometry> child = nonLeaf.child(current.position());
                current.advance();
                double distance = child.geometry().distance(queryMBR);
                stats.recordNonLeafExpansion(distance);
                if (distance <= threshold) {
                    frontier.add(new SearchNodeState(child, 0, distance));
                    stats.recordEnqueuedChild();
                }
            }

            if (stats.distanceComputations >= safeCheckpointInterval
                    && stats.distanceComputations % safeCheckpointInterval == 0) {
                TraversalFeatureSnapshot snapshot = stats.toSnapshot(frontier);
                if (collectCheckpoints) {
                    checkpoints.add(new CheckpointRecord(snapshot, acceptedCandidates.size()));
                }
                if (checkpointDecider != null && acceptedCandidates.size() >= requestedK) {
                    double score = checkpointDecider.score(snapshot);
                    if (checkpointDecider.shouldStop(snapshot, score)) {
                        return new TraversalResult(acceptedCandidates, checkpoints, true, score, snapshot.checkpointId);
                    }
                }
            }
        }

        return new TraversalResult(acceptedCandidates, checkpoints, false, -1d, stats.checkpointId);
    }

    public interface CheckpointDecider {
        double score(TraversalFeatureSnapshot snapshot);

        boolean shouldStop(TraversalFeatureSnapshot snapshot, double score);
    }

    public static class TraversalResult {
        public final List<CandidateSegment> acceptedCandidates;
        public final List<CheckpointRecord> checkpoints;
        public final boolean stoppedEarly;
        public final double stopScore;
        public final int lastCheckpointId;

        public TraversalResult(List<CandidateSegment> acceptedCandidates, List<CheckpointRecord> checkpoints,
                               boolean stoppedEarly, double stopScore, int lastCheckpointId) {
            this.acceptedCandidates = acceptedCandidates;
            this.checkpoints = checkpoints;
            this.stoppedEarly = stoppedEarly;
            this.stopScore = stopScore;
            this.lastCheckpointId = lastCheckpointId;
        }
    }

    public static class CheckpointRecord {
        public final TraversalFeatureSnapshot snapshot;
        public final int candidateCountAtCheckpoint;

        public CheckpointRecord(TraversalFeatureSnapshot snapshot, int candidateCountAtCheckpoint) {
            this.snapshot = snapshot;
            this.candidateCountAtCheckpoint = candidateCountAtCheckpoint;
        }
    }

    private static final class SearchNodeState {
        private final Node<CandidateSegment, Geometry> node;
        private int position;
        private final double priority;

        private SearchNodeState(Node<CandidateSegment, Geometry> node, int position, double priority) {
            this.node = node;
            this.position = position;
            this.priority = priority;
        }

        private Node<CandidateSegment, Geometry> node() {
            return node;
        }

        private int position() {
            return position;
        }

        private void advance() {
            position++;
        }

        private boolean hasRemaining() {
            return position < node.count();
        }

        private double priority() {
            return priority;
        }
    }

    private static final class TraversalStats implements Serializable {
        private final double threshold;
        private final int requestedK;
        private final Set<Integer> uniqueSeries = new HashSet<>();

        private long distanceComputations;
        private long nodesCompleted;
        private long leafsCompleted;
        private long nonLeafExpansions;
        private long childrenEnqueued;
        private long entriesScanned;
        private int candidateCount;
        private double lastDistance = Double.POSITIVE_INFINITY;
        private double lastCandidateDistance = Double.POSITIVE_INFINITY;
        private int checkpointId;

        private TraversalStats(double threshold, int requestedK) {
            this.threshold = threshold;
            this.requestedK = requestedK;
        }

        private void recordEntryScan(double distance) {
            distanceComputations++;
            entriesScanned++;
            lastDistance = distance;
        }

        private void recordAcceptedCandidate(CandidateSegment candidateSegment, double distance) {
            candidateCount++;
            uniqueSeries.add(candidateSegment.getTimeSeriesIndex());
            lastCandidateDistance = distance;
        }

        private void recordNonLeafExpansion(double distance) {
            distanceComputations++;
            nonLeafExpansions++;
            lastDistance = distance;
        }

        private void recordEnqueuedChild() {
            childrenEnqueued++;
        }

        private void completeNode(boolean leaf) {
            if (leaf) {
                leafsCompleted++;
                Parameters.nLeafsVisited++;
            } else {
                nodesCompleted++;
                Parameters.nNodesVisited++;
            }
        }

        private TraversalFeatureSnapshot toSnapshot(PriorityQueue<SearchNodeState> frontier) {
            checkpointId++;
            double best = frontier.isEmpty() ? Double.POSITIVE_INFINITY : frontier.peek().priority();
            double sum = 0d;
            double worst = Double.NEGATIVE_INFINITY;
            for (SearchNodeState state : frontier) {
                sum += state.priority();
                worst = Math.max(worst, state.priority());
            }
            double avg = frontier.isEmpty() ? Double.POSITIVE_INFINITY : sum / frontier.size();
            if (frontier.isEmpty()) {
                worst = Double.POSITIVE_INFINITY;
            }

            return new TraversalFeatureSnapshot(
                    requestedK,
                    checkpointId,
                    distanceComputations,
                    candidateCount,
                    uniqueSeries.size(),
                    nodesCompleted,
                    leafsCompleted,
                    nonLeafExpansions,
                    childrenEnqueued,
                    entriesScanned,
                    frontier.size(),
                    best,
                    avg,
                    worst,
                    lastDistance,
                    lastCandidateDistance,
                    threshold
            );
        }
    }
}
