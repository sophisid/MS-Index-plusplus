package io.github.algorithms.msindexpp.decision;

import io.github.algorithms.AlgorithmType;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

public final class DecisionEvaluationCsvWriter {
    private static final String[] EXTRA_COLUMNS = {
            "query_id",
            "predicted_algorithm",
            "actual_best_algorithm",
            "prediction_correct",
            "predicted_time_ms",
            "mass_time_ms",
            "msindex_time_ms"
    };

    private DecisionEvaluationCsvWriter() {
    }

    public static synchronized void append(
            Path path,
            int queryId,
            DecisionFeatureVector featureVector,
            AlgorithmType predictedAlgorithm,
            AlgorithmType actualBestAlgorithm,
            long predictedTimeMs,
            long massTimeMs,
            long msIndexTimeMs
    ) {
        try {
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }

            if (!Files.exists(path)) {
                Files.writeString(path, header() + "\n", StandardCharsets.UTF_8, StandardOpenOption.CREATE_NEW);
            }

            final List<String> row = new ArrayList<>();
            row.add(Integer.toString(queryId));
            featureVector.asMap().values().forEach(value -> row.add(Double.toString(value)));
            row.add(predictedAlgorithm.name());
            row.add(actualBestAlgorithm.name());
            row.add(predictedAlgorithm == actualBestAlgorithm ? "1" : "0");
            row.add(Long.toString(predictedTimeMs));
            row.add(Long.toString(massTimeMs));
            row.add(Long.toString(msIndexTimeMs));

            Files.writeString(
                    path,
                    String.join(",", row) + "\n",
                    StandardCharsets.UTF_8,
                    StandardOpenOption.APPEND
            );
        } catch (IOException e) {
            throw new RuntimeException("Failed to append evaluation row to " + path, e);
        }
    }

    private static String header() {
        final List<String> columns = new ArrayList<>();
        columns.add(EXTRA_COLUMNS[0]);
        for (String featureName : DecisionFeatureVector.FEATURE_NAMES) {
            columns.add(featureName);
        }
        for (int i = 1; i < EXTRA_COLUMNS.length; i++) {
            columns.add(EXTRA_COLUMNS[i]);
        }
        return String.join(",", columns);
    }
}
