package io.github.algorithms.msindexpp.ml;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.github.utils.Parameters;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;

public class TrainingDataWriter {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    public static void appendRows(String outputPath, int queryId, List<TrainingRow> rows) throws IOException {
        if (rows.isEmpty()) {
            return;
        }

        File outputFile = new File(outputPath);
        File parent = outputFile.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }

        try (BufferedWriter writer = Files.newBufferedWriter(outputFile.toPath(), StandardCharsets.UTF_8,
                Files.exists(outputFile.toPath()) ? java.nio.file.StandardOpenOption.APPEND : java.nio.file.StandardOpenOption.CREATE)) {
            for (TrainingRow row : rows) {
                ObjectNode node = MAPPER.createObjectNode();
                node.put("dataset", Parameters.datasetName);
                node.put("query_id", queryId);
                node.put("label", row.label ? 1 : 0);
                node.put("achieved_accuracy", row.achievedAccuracy);
                node.put("actual_recall", row.achievedAccuracy);
                node.put("candidate_count_at_checkpoint", row.candidateCountAtCheckpoint);
                node.put("final_candidate_count", row.finalCandidateCount);
                for (var entry : row.snapshot.toFeatureMap().entrySet()) {
                    node.put(entry.getKey(), entry.getValue());
                }
                writer.write(MAPPER.writeValueAsString(node));
                writer.newLine();
            }
        }
    }

    public static class TrainingRow {
        public final TraversalFeatureSnapshot snapshot;
        public final boolean label;
        public final double achievedAccuracy;
        public final int candidateCountAtCheckpoint;
        public final int finalCandidateCount;

        public TrainingRow(TraversalFeatureSnapshot snapshot, boolean label, double achievedAccuracy,
                           int candidateCountAtCheckpoint, int finalCandidateCount) {
            this.snapshot = snapshot;
            this.label = label;
            this.achievedAccuracy = achievedAccuracy;
            this.candidateCountAtCheckpoint = candidateCountAtCheckpoint;
            this.finalCandidateCount = finalCandidateCount;
        }
    }
}
